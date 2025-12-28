#!/usr/bin/env python
"""Compare drug repurposing rankings between baseline and signed edge modes.

This script:
1. Runs predictions for all neurological diseases in BOTH modes
2. Compares Top-K drug rankings before/after signed edge modulation
3. Identifies drugs that rose or fell significantly in rankings
4. Generates comparison reports and visualizations
"""

import logging
from pathlib import Path

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


def compute_scores_for_disease(
    pretrain_model, drug_nodes, disease_index, embeddings, kg, edge_type: str = "indication"
):
    """Compute scores for all drugs for a given disease and edge type."""
    src_ids = [disease_index] * len(drug_nodes)
    dst_ids = drug_nodes["node_index"].values.tolist()

    scores = (
        pretrain_model.get_scores_from_embeddings(
            src_ids, dst_ids, ("disease", edge_type, "drug"), embeddings=embeddings, query_kg=kg, use_cache=False
        )
        .cpu()
        .numpy()
    )

    return scores


def run_predictions(edge_signs_enabled: bool, nodes, edges, test_diseases):
    """Run predictions with specified edge_signs setting."""

    # Temporarily modify the config
    original_enabled = conf.edge_signs.enabled
    conf.edge_signs.enabled = edge_signs_enabled

    _logger.info(f"\n{'=' * 60}")
    _logger.info(f"Running predictions with edge_signs.enabled = {edge_signs_enabled}")
    _logger.info(f"{'=' * 60}")

    # Load model (needs to be reloaded to pick up config change)
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

    drug_nodes = nodes[nodes["node_type"] == "drug"].copy()

    # Get indication and contraindication edges for ground truth
    indications = edges[edges["relation"] == "indication"]
    contraindications = edges[edges["relation"] == "contraindication"]
    indication_pairs = set(zip(indications["x_index"], indications["y_index"], strict=False))
    contraindication_pairs = set(zip(contraindications["x_index"], contraindications["y_index"], strict=False))

    all_results = []

    for disease_index in tqdm(test_diseases, desc="Evaluating diseases"):
        disease_name = nodes[nodes["node_index"] == disease_index]["node_name"].values[0]

        # Get indication scores
        indication_scores = compute_scores_for_disease(
            pretrain_model, drug_nodes, disease_index, embeddings, kg, "indication"
        )

        # Get contraindication scores
        contraindication_scores = compute_scores_for_disease(
            pretrain_model, drug_nodes, disease_index, embeddings, kg, "contraindication"
        )

        # Build result dataframe
        result_df = drug_nodes[["node_index", "node_name"]].copy()
        result_df["disease_index"] = disease_index
        result_df["disease_name"] = disease_name
        result_df["indication_score"] = indication_scores
        result_df["contraindication_score"] = contraindication_scores
        result_df["indication_rank"] = get_pred_ranks(indication_scores)
        result_df["contraindication_rank"] = get_pred_ranks(contraindication_scores)

        # Add ground truth
        result_df["is_known_indication"] = result_df["node_index"].apply(
            lambda x, di=disease_index: 1 if (di, x) in indication_pairs else 0
        )
        result_df["is_known_contraindication"] = result_df["node_index"].apply(
            lambda x, di=disease_index: 1 if (di, x) in contraindication_pairs else 0
        )

        all_results.append(result_df)

    # Restore original config
    conf.edge_signs.enabled = original_enabled

    return pd.concat(all_results, ignore_index=True)


def compare_rankings(baseline_df: pd.DataFrame, signed_df: pd.DataFrame, top_k: int = 100):
    """Compare rankings between baseline and signed predictions."""

    # Merge on drug + disease
    merged = baseline_df.merge(
        signed_df,
        on=[
            "node_index",
            "node_name",
            "disease_index",
            "disease_name",
            "is_known_indication",
            "is_known_contraindication",
        ],
        suffixes=("_baseline", "_signed"),
    )

    # Calculate rank changes
    merged["indication_rank_change"] = merged["indication_rank_baseline"] - merged["indication_rank_signed"]
    merged["contraindication_rank_change"] = (
        merged["contraindication_rank_baseline"] - merged["contraindication_rank_signed"]
    )

    return merged


def generate_report(comparison_df: pd.DataFrame, output_dir: Path, top_k: int = 100):
    """Generate comparison reports."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Report 1: Drugs that moved into Top-K (rose in ranking)
    # =========================================================================
    drugs_rose = comparison_df[
        (comparison_df["indication_rank_signed"] <= top_k) & (comparison_df["indication_rank_baseline"] > top_k)
    ].copy()
    drugs_rose = drugs_rose.sort_values("indication_rank_change", ascending=False)

    # =========================================================================
    # Report 2: Drugs that fell out of Top-K
    # =========================================================================
    drugs_fell = comparison_df[
        (comparison_df["indication_rank_baseline"] <= top_k) & (comparison_df["indication_rank_signed"] > top_k)
    ].copy()
    drugs_fell = drugs_fell.sort_values("indication_rank_change", ascending=True)

    # =========================================================================
    # Report 3: Contraindicated drugs that were correctly suppressed
    # =========================================================================
    contras_suppressed = comparison_df[(comparison_df["is_known_contraindication"] == 1)].copy()
    contras_suppressed["rank_improvement"] = (
        contras_suppressed["contraindication_rank_signed"] - contras_suppressed["contraindication_rank_baseline"]
    )
    contras_suppressed = contras_suppressed.sort_values("rank_improvement", ascending=False)

    # =========================================================================
    # Report 4: Top movers per disease
    # =========================================================================
    top_movers = []
    for disease_name in comparison_df["disease_name"].unique():
        disease_df = comparison_df[comparison_df["disease_name"] == disease_name]

        # Top 5 risers
        risers = disease_df.nlargest(5, "indication_rank_change")[
            [
                "node_name",
                "disease_name",
                "indication_rank_baseline",
                "indication_rank_signed",
                "indication_rank_change",
                "is_known_indication",
                "is_known_contraindication",
            ]
        ].copy()
        risers["movement"] = "rose"

        # Top 5 fallers
        fallers = disease_df.nsmallest(5, "indication_rank_change")[
            [
                "node_name",
                "disease_name",
                "indication_rank_baseline",
                "indication_rank_signed",
                "indication_rank_change",
                "is_known_indication",
                "is_known_contraindication",
            ]
        ].copy()
        fallers["movement"] = "fell"

        top_movers.append(risers)
        top_movers.append(fallers)

    top_movers_df = pd.concat(top_movers, ignore_index=True)

    # =========================================================================
    # Report 5: Summary statistics
    # =========================================================================
    summary = {
        "total_drug_disease_pairs": len(comparison_df),
        "unique_diseases": comparison_df["disease_name"].nunique(),
        "unique_drugs": comparison_df["node_name"].nunique(),
        f"drugs_entering_top_{top_k}": len(drugs_rose),
        f"drugs_leaving_top_{top_k}": len(drugs_fell),
        "known_indications_in_top_100_baseline": len(
            comparison_df[
                (comparison_df["is_known_indication"] == 1) & (comparison_df["indication_rank_baseline"] <= 100)
            ]
        ),
        "known_indications_in_top_100_signed": len(
            comparison_df[
                (comparison_df["is_known_indication"] == 1) & (comparison_df["indication_rank_signed"] <= 100)
            ]
        ),
        "known_contraindications_in_top_100_baseline": len(
            comparison_df[
                (comparison_df["is_known_contraindication"] == 1)
                & (comparison_df["contraindication_rank_baseline"] <= 100)
            ]
        ),
        "known_contraindications_in_top_100_signed": len(
            comparison_df[
                (comparison_df["is_known_contraindication"] == 1)
                & (comparison_df["contraindication_rank_signed"] <= 100)
            ]
        ),
    }

    # =========================================================================
    # Save reports
    # =========================================================================
    comparison_df.to_csv(output_dir / "full_comparison.csv", index=False)
    drugs_rose.to_csv(output_dir / f"drugs_entering_top_{top_k}.csv", index=False)
    drugs_fell.to_csv(output_dir / f"drugs_leaving_top_{top_k}.csv", index=False)
    contras_suppressed.to_csv(output_dir / "contraindications_suppressed.csv", index=False)
    top_movers_df.to_csv(output_dir / "top_movers_per_disease.csv", index=False)

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / "summary_statistics.csv", index=False)

    return summary, top_movers_df


def main():
    _logger.info("=" * 70)
    _logger.info("DRUG RANKING COMPARISON: Baseline vs Signed Edge Modulation")
    _logger.info("=" * 70)

    # Load data once
    _logger.info("Loading nodes and edges...")
    nodes = pd.read_csv(conf.paths.kg.nodes_path, dtype={"node_index": int}, low_memory=False)
    edges = pd.read_csv(
        conf.paths.kg.edges_path, dtype={"edge_index": int, "x_index": int, "y_index": int}, low_memory=False
    )
    _logger.info(f"Loaded {len(nodes)} nodes and {len(edges) // 2} edges")

    # Test diseases - neurological disorders
    test_diseases = [
        42049,  # Alzheimer disease
        39579,  # Parkinson disease
        39528,  # bipolar disorder
        41591,  # multiple sclerosis
        39348,  # epilepsy
        32652,  # schizophrenia
        40310,  # autism spectrum disorder
        40341,  # attention deficit hyperactivity disorder
        40366,  # Huntington disease
        39326,  # amyotrophic lateral sclerosis
        39342,  # anxiety disorder
        42188,  # stroke disorder
        41524,  # Down syndrome
        41571,  # meningitis
        39364,  # encephalitis
        41508,  # diabetic neuropathy
        39330,  # Charcot-Marie-Tooth disease
    ]

    # Filter to diseases that exist
    disease_nodes = nodes[nodes["node_type"] == "disease"]
    test_diseases = [d for d in test_diseases if d in disease_nodes["node_index"].values]
    _logger.info(f"Evaluating {len(test_diseases)} neurological diseases")

    # Run baseline predictions
    baseline_df = run_predictions(edge_signs_enabled=False, nodes=nodes, edges=edges, test_diseases=test_diseases)

    # Run signed predictions
    signed_df = run_predictions(edge_signs_enabled=True, nodes=nodes, edges=edges, test_diseases=test_diseases)

    # Compare rankings
    _logger.info("\nComparing rankings...")
    comparison_df = compare_rankings(baseline_df, signed_df, top_k=100)

    # Generate reports
    output_dir = conf.paths.notebooks.drug_repurposing_dir / "ranking_comparison"
    summary, top_movers = generate_report(comparison_df, output_dir, top_k=100)

    # Print summary
    _logger.info("\n" + "=" * 70)
    _logger.info("SUMMARY")
    _logger.info("=" * 70)
    for key, value in summary.items():
        _logger.info(f"  {key}: {value}")

    # Print top movers for first disease
    _logger.info("\n" + "=" * 70)
    _logger.info("SAMPLE: Top Movers for First Disease")
    _logger.info("=" * 70)
    first_disease = top_movers["disease_name"].iloc[0]
    sample_movers = top_movers[top_movers["disease_name"] == first_disease]

    _logger.info(f"\n{first_disease}:")
    _logger.info("-" * 60)

    risers = sample_movers[sample_movers["movement"] == "rose"]
    _logger.info("\n  Drugs that ROSE in ranking (now more favored):")
    for _, row in risers.iterrows():
        marker = ""
        if row["is_known_indication"] == 1:
            marker = " [KNOWN INDICATION]"
        elif row["is_known_contraindication"] == 1:
            marker = " [KNOWN CONTRAINDICATION]"
        _logger.info(
            f"    {row['node_name']}: {int(row['indication_rank_baseline'])} -> "
            f"{int(row['indication_rank_signed'])} (Δ{int(row['indication_rank_change'])}){marker}"
        )

    fallers = sample_movers[sample_movers["movement"] == "fell"]
    _logger.info("\n  Drugs that FELL in ranking (now less favored):")
    for _, row in fallers.iterrows():
        marker = ""
        if row["is_known_indication"] == 1:
            marker = " [KNOWN INDICATION]"
        elif row["is_known_contraindication"] == 1:
            marker = " [KNOWN CONTRAINDICATION]"
        _logger.info(
            f"    {row['node_name']}: {int(row['indication_rank_baseline'])} -> "
            f"{int(row['indication_rank_signed'])} (Δ{int(row['indication_rank_change'])}){marker}"
        )

    _logger.info(f"\n\nAll reports saved to: {output_dir}")
    _logger.info("Files generated:")
    _logger.info("  - full_comparison.csv")
    _logger.info("  - drugs_entering_top_100.csv")
    _logger.info("  - drugs_leaving_top_100.csv")
    _logger.info("  - contraindications_suppressed.csv")
    _logger.info("  - top_movers_per_disease.csv")
    _logger.info("  - summary_statistics.csv")


if __name__ == "__main__":
    main()
