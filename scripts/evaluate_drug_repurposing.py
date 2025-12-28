#!/usr/bin/env python
"""Evaluate drug repurposing predictions with additional metrics for signed edge experiments.

Metrics computed:
- Mean indication score (across all drug-disease pairs)
- Mean contraindication score (across all drug-disease pairs)
- Known indication drugs mean score
- Known contraindication drugs mean score
- Recall@500 for indications
"""

import logging

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from tqdm import tqdm

from src.config import conf
from src.constants import TORCH_DEVICE
from src.dataloaders import load_graph
from src.models import HGT

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
_logger = logging.getLogger(__name__)


def get_pred_ranks(preds):
    """Convert prediction scores to ranks (1 = highest score)."""
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    sorted_indices = np.argsort(preds)
    return len(preds) - np.argsort(sorted_indices)


def compute_recall_at_k(df: pd.DataFrame, ground_truth_col: str, rank_col: str, k: int) -> float:
    """Compute recall at k."""
    if df[ground_truth_col].sum() == 0:
        return 0.0
    top_k = df[df[rank_col] <= k]
    true_positives = top_k[ground_truth_col].sum()
    total_positives = df[ground_truth_col].sum()
    return true_positives / total_positives if total_positives > 0 else 0.0


def main():
    _logger.info("=" * 60)
    _logger.info("Drug Repurposing Evaluation")
    _logger.info("=" * 60)

    # Check edge signs config
    _logger.info(f"Edge signs enabled: {conf.edge_signs.enabled}")
    if conf.edge_signs.enabled:
        _logger.info(f"Edge sign mapping has {len(conf.edge_signs.mapping)} entries")

    # Load data
    _logger.info("Loading nodes and edges...")
    nodes = pd.read_csv(conf.paths.kg.nodes_path, dtype={"node_index": int}, low_memory=False)
    edges = pd.read_csv(
        conf.paths.kg.edges_path, dtype={"edge_index": int, "x_index": int, "y_index": int}, low_memory=False
    )
    _logger.info(f"Loaded {len(nodes)} nodes and {len(edges) // 2} edges")

    # Load model and embeddings
    _logger.info("Loading model and embeddings...")
    pl.seed_everything(conf.seed, workers=True)
    kg = load_graph(nodes, edges)

    pretrain_model = HGT.load_from_checkpoint(
        checkpoint_path=str(conf.paths.checkpoint.checkpoint_path),
        kg=kg,
        strict=False,
    )
    pretrain_model.eval()
    pretrain_model = pretrain_model.to(TORCH_DEVICE)

    embeddings = torch.load(conf.paths.checkpoint.embeddings_path)
    _logger.info(f"Loaded embeddings with shape: {embeddings.shape}")

    # Get drug and disease nodes
    drug_nodes = nodes[nodes["node_type"] == "drug"].copy()
    disease_nodes = nodes[nodes["node_type"] == "disease"].copy()
    _logger.info(f"Found {len(drug_nodes)} drugs and {len(disease_nodes)} diseases")

    # Get indication and contraindication edges
    indications = edges[edges["relation"] == "indication"]
    contraindications = edges[edges["relation"] == "contraindication"]
    _logger.info(
        f"Found {len(indications) // 2} indication edges and {len(contraindications) // 2} contraindication edges"
    )

    # Create ground truth mappings
    indication_pairs = set(zip(indications["x_index"], indications["y_index"], strict=False))
    contraindication_pairs = set(zip(contraindications["x_index"], contraindications["y_index"], strict=False))

    # =========================================================================
    # Compute metrics for a subset of diseases (neurological disorders from test sets)
    # =========================================================================
    test_sets = [
        42049,  # Alzheimer disease
        39579,  # Parkinson disease
        39528,  # bipolar disorder
        41591,  # multiple sclerosis
        39348,  # epilepsy
        32652,  # schizophrenia
    ]

    results = []
    all_indication_scores = []
    all_contraindication_scores = []
    known_indication_scores = []
    known_contraindication_scores = []

    _logger.info("\n" + "=" * 60)
    _logger.info("Computing predictions for neurological diseases...")
    _logger.info("=" * 60)

    for disease_index in tqdm(test_sets, desc="Evaluating diseases"):
        disease_name = nodes[nodes["node_index"] == disease_index]["node_name"].values[0]

        # Get scores
        src_ids = [disease_index] * len(drug_nodes)
        dst_ids = drug_nodes["node_index"].values.tolist()

        # Indication scores
        indication_scores = (
            pretrain_model.get_scores_from_embeddings(
                src_ids, dst_ids, ("disease", "indication", "drug"), embeddings=embeddings, query_kg=kg, use_cache=False
            )
            .cpu()
            .numpy()
        )

        # Contraindication scores
        contraindication_scores = (
            pretrain_model.get_scores_from_embeddings(
                src_ids,
                dst_ids,
                ("disease", "contraindication", "drug"),
                embeddings=embeddings,
                query_kg=kg,
                use_cache=False,
            )
            .cpu()
            .numpy()
        )

        # Build evaluation dataframe
        eval_df = drug_nodes[["node_index", "node_name"]].copy()
        eval_df["indication_score"] = indication_scores
        eval_df["contraindication_score"] = contraindication_scores
        eval_df["indication_rank"] = get_pred_ranks(indication_scores)
        eval_df["contraindication_rank"] = get_pred_ranks(contraindication_scores)

        # Add ground truth
        eval_df["is_known_indication"] = eval_df["node_index"].apply(
            lambda x, di=disease_index: 1 if (di, x) in indication_pairs else 0
        )
        eval_df["is_known_contraindication"] = eval_df["node_index"].apply(
            lambda x, di=disease_index: 1 if (di, x) in contraindication_pairs else 0
        )

        # Collect scores for aggregation
        all_indication_scores.extend(indication_scores)
        all_contraindication_scores.extend(contraindication_scores)

        known_ind_mask = eval_df["is_known_indication"] == 1
        known_contra_mask = eval_df["is_known_contraindication"] == 1

        if known_ind_mask.sum() > 0:
            known_indication_scores.extend(eval_df.loc[known_ind_mask, "indication_score"].tolist())
        if known_contra_mask.sum() > 0:
            known_contraindication_scores.extend(eval_df.loc[known_contra_mask, "contraindication_score"].tolist())

        # Compute per-disease metrics
        disease_result = {
            "disease_index": disease_index,
            "disease_name": disease_name,
            "n_drugs": len(drug_nodes),
            "n_known_indications": known_ind_mask.sum(),
            "n_known_contraindications": known_contra_mask.sum(),
            "mean_indication_score": np.mean(indication_scores),
            "mean_contraindication_score": np.mean(contraindication_scores),
            "known_indication_mean_score": eval_df.loc[known_ind_mask, "indication_score"].mean()
            if known_ind_mask.sum() > 0
            else np.nan,
            "known_contraindication_mean_score": eval_df.loc[known_contra_mask, "contraindication_score"].mean()
            if known_contra_mask.sum() > 0
            else np.nan,
            "recall_at_100_indication": compute_recall_at_k(eval_df, "is_known_indication", "indication_rank", 100),
            "recall_at_500_indication": compute_recall_at_k(eval_df, "is_known_indication", "indication_rank", 500),
            "recall_at_1000_indication": compute_recall_at_k(eval_df, "is_known_indication", "indication_rank", 1000),
        }
        results.append(disease_result)

    # =========================================================================
    # Aggregate results
    # =========================================================================
    results_df = pd.DataFrame(results)

    _logger.info("\n" + "=" * 60)
    _logger.info("EVALUATION RESULTS")
    _logger.info("=" * 60)

    # Print per-disease results
    _logger.info("\nPer-Disease Results:")
    _logger.info("-" * 60)
    for _, row in results_df.iterrows():
        _logger.info(f"\n{row['disease_name']}:")
        _logger.info(f"  Known indications: {row['n_known_indications']}")
        _logger.info(f"  Known contraindications: {row['n_known_contraindications']}")
        _logger.info(f"  Mean indication score: {row['mean_indication_score']:.4f}")
        _logger.info(f"  Mean contraindication score: {row['mean_contraindication_score']:.4f}")
        if not np.isnan(row["known_indication_mean_score"]):
            _logger.info(f"  Known indication drugs mean score: {row['known_indication_mean_score']:.4f}")
        if not np.isnan(row["known_contraindication_mean_score"]):
            _logger.info(f"  Known contraindication drugs mean score: {row['known_contraindication_mean_score']:.4f}")
        _logger.info(f"  Recall@100 (indication): {row['recall_at_100_indication']:.4f}")
        _logger.info(f"  Recall@500 (indication): {row['recall_at_500_indication']:.4f}")
        _logger.info(f"  Recall@1000 (indication): {row['recall_at_1000_indication']:.4f}")

    # Print aggregate results
    _logger.info("\n" + "=" * 60)
    _logger.info("AGGREGATE METRICS")
    _logger.info("=" * 60)

    agg_metrics = {
        "edge_signs_enabled": conf.edge_signs.enabled,
        "n_diseases_evaluated": len(results_df),
        "mean_indication_score_overall": np.mean(all_indication_scores),
        "mean_contraindication_score_overall": np.mean(all_contraindication_scores),
        "known_indication_drugs_mean_score": np.mean(known_indication_scores) if known_indication_scores else np.nan,
        "known_contraindication_drugs_mean_score": np.mean(known_contraindication_scores)
        if known_contraindication_scores
        else np.nan,
        "mean_recall_at_100_indication": results_df["recall_at_100_indication"].mean(),
        "mean_recall_at_500_indication": results_df["recall_at_500_indication"].mean(),
        "mean_recall_at_1000_indication": results_df["recall_at_1000_indication"].mean(),
    }

    _logger.info(f"\nEdge signs enabled: {agg_metrics['edge_signs_enabled']}")
    _logger.info(f"Diseases evaluated: {agg_metrics['n_diseases_evaluated']}")
    _logger.info("\n--- Score Metrics ---")
    _logger.info(f"Mean indication score (all pairs): {agg_metrics['mean_indication_score_overall']:.4f}")
    _logger.info(f"Mean contraindication score (all pairs): {agg_metrics['mean_contraindication_score_overall']:.4f}")
    _logger.info(f"Known indication drugs mean score: {agg_metrics['known_indication_drugs_mean_score']:.4f}")
    _logger.info(
        f"Known contraindication drugs mean score: {agg_metrics['known_contraindication_drugs_mean_score']:.4f}"
    )
    _logger.info("\n--- Recall Metrics ---")
    _logger.info(f"Mean Recall@100 (indication): {agg_metrics['mean_recall_at_100_indication']:.4f}")
    _logger.info(f"Mean Recall@500 (indication): {agg_metrics['mean_recall_at_500_indication']:.4f}")
    _logger.info(f"Mean Recall@1000 (indication): {agg_metrics['mean_recall_at_1000_indication']:.4f}")

    # Save results
    output_dir = conf.paths.notebooks.drug_repurposing_dir / "signed_edge_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "baseline" if not conf.edge_signs.enabled else "signed"
    results_df.to_csv(output_dir / f"per_disease_results_{suffix}.csv", index=False)

    agg_df = pd.DataFrame([agg_metrics])
    agg_df.to_csv(output_dir / f"aggregate_metrics_{suffix}.csv", index=False)

    _logger.info(f"\nResults saved to: {output_dir}")

    return agg_metrics


if __name__ == "__main__":
    main()
