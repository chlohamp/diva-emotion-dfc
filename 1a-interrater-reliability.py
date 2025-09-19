import pandas as pd
import numpy as np
import pingouin as pg
from pathlib import Path


def fisher_z_transform(r):
    """Apply Fisher z-transformation to correlation coefficients."""
    return np.arctanh(r)


def calculate_icc(rater_a1_data, rater_a2_data):
    """Calculate ICC(2,1) for valence and arousal ratings."""

    n_trials = len(rater_a1_data)

    # Prepare data for ICC calculation - long format
    # Valence data
    valence_data = pd.DataFrame(
        {
            "subject": list(range(n_trials)) * 2,
            "rater": ["A1"] * n_trials + ["A2"] * n_trials,
            "valence": (
                list(rater_a1_data["valence"]) + list(rater_a2_data["valence"])
            ),
        }
    )

    # Arousal data
    arousal_data = pd.DataFrame(
        {
            "subject": list(range(n_trials)) * 2,
            "rater": ["A1"] * n_trials + ["A2"] * n_trials,
            "arousal": (
                list(rater_a1_data["arousal"]) + list(rater_a2_data["arousal"])
            ),
        }
    )

    # Calculate ICC(2,1) for valence
    icc_valence = pg.intraclass_corr(
        data=valence_data, targets="subject", raters="rater", ratings="valence"
    )

    # Calculate ICC(2,1) for arousal
    icc_arousal = pg.intraclass_corr(
        data=arousal_data, targets="subject", raters="rater", ratings="arousal"
    )

    # Extract ICC(2,1) values - ICC2 with single rater
    icc2_valence = icc_valence[icc_valence["Type"] == "ICC2"]
    icc2_arousal = icc_arousal[icc_arousal["Type"] == "ICC2"]

    # Get single rater ICC values and p-values - ICC2 single random raters
    val_row = icc2_valence[icc2_valence["Description"].str.contains("Single")]
    aro_row = icc2_arousal[icc2_arousal["Description"].str.contains("Single")]

    val_icc = val_row["ICC"].iloc[0]
    val_pval = val_row["pval"].iloc[0]
    val_ci = val_row["CI95%"].iloc[0]

    aro_icc = aro_row["ICC"].iloc[0]
    aro_pval = aro_row["pval"].iloc[0]
    aro_ci = aro_row["CI95%"].iloc[0]

    return val_icc, val_pval, val_ci, aro_icc, aro_pval, aro_ci


def detect_outlier_timepoints(rater_a1_data, rater_a2_data):
    """
    Detect specific timepoints where raters strongly disagree.
    Returns detailed information about outlier timepoints.
    """
    n_trials = len(rater_a1_data)
    outlier_details = {
        "timepoints": [],
        "valence_disagreements": [],
        "arousal_disagreements": [],
        "summary": {},
    }

    # Calculate difference scores for each timepoint
    val_diff = np.abs(rater_a1_data["valence"] - rater_a2_data["valence"])
    aro_diff = np.abs(rater_a1_data["arousal"] - rater_a2_data["arousal"])

    # Define disagreement thresholds
    large_disagreement_threshold = 2.0  # 2+ scale points difference
    moderate_disagreement_threshold = 1.5  # 1.5+ scale points difference

    # Check each timepoint
    for i in range(n_trials):
        val_disagreement = val_diff.iloc[i]
        aro_disagreement = aro_diff.iloc[i]

        # Check if this timepoint has large disagreements
        is_val_outlier = val_disagreement > large_disagreement_threshold
        is_aro_outlier = aro_disagreement > large_disagreement_threshold

        if is_val_outlier or is_aro_outlier:
            timepoint_info = {
                "timepoint_index": i,
                "onset_time": rater_a1_data.iloc[i].get("onset", i),
                "valence_rater1": rater_a1_data["valence"].iloc[i],
                "valence_rater2": rater_a2_data["valence"].iloc[i],
                "valence_difference": val_disagreement,
                "arousal_rater1": rater_a1_data["arousal"].iloc[i],
                "arousal_rater2": rater_a2_data["arousal"].iloc[i],
                "arousal_difference": aro_disagreement,
                "valence_outlier": is_val_outlier,
                "arousal_outlier": is_aro_outlier,
            }
            outlier_details["timepoints"].append(timepoint_info)

    # Calculate summary statistics
    large_val_disagreements = (val_diff > large_disagreement_threshold).sum()
    large_aro_disagreements = (aro_diff > large_disagreement_threshold).sum()
    moderate_val_disagreements = (val_diff > moderate_disagreement_threshold).sum()
    moderate_aro_disagreements = (aro_diff > moderate_disagreement_threshold).sum()

    outlier_details["summary"] = {
        "total_timepoints": n_trials,
        "large_valence_disagreements": large_val_disagreements,
        "large_arousal_disagreements": large_aro_disagreements,
        "moderate_valence_disagreements": moderate_val_disagreements,
        "moderate_arousal_disagreements": moderate_aro_disagreements,
        "pct_large_valence": (large_val_disagreements / n_trials) * 100,
        "pct_large_arousal": (large_aro_disagreements / n_trials) * 100,
        "mean_valence_difference": val_diff.mean(),
        "mean_arousal_difference": aro_diff.mean(),
        "max_valence_difference": val_diff.max(),
        "max_arousal_difference": aro_diff.max(),
    }

    return outlier_details


def calculate_agreement_metrics(rater_a1_data, rater_a2_data):
    """Calculate additional agreement metrics beyond ICC."""

    metrics = {}

    for emotion in ["valence", "arousal"]:
        r1_ratings = rater_a1_data[emotion]
        r2_ratings = rater_a2_data[emotion]

        # Mean absolute difference
        mad = np.mean(np.abs(r1_ratings - r2_ratings))

        # Root mean square difference
        rmsd = np.sqrt(np.mean((r1_ratings - r2_ratings) ** 2))

        # Proportion of ratings within 1 scale point
        within_1_point = np.mean(np.abs(r1_ratings - r2_ratings) <= 1)

        # Proportion of ratings within 0.5 scale points
        within_half_point = np.mean(np.abs(r1_ratings - r2_ratings) <= 0.5)

        metrics[f"{emotion}_MAD"] = mad
        metrics[f"{emotion}_RMSD"] = rmsd
        metrics[f"{emotion}_within_1pt"] = within_1_point
        metrics[f"{emotion}_within_0.5pt"] = within_half_point

    return metrics


def main():
    print("Inter-rater Reliability Analysis")
    print("=" * 40)

    # Load the two rater files
    try:
        rater_a1_file = (
            "derivatives/simulated/ses-01_task-strangerthings_acq-A1_run-1_events.tsv"
        )
        rater_a2_file = (
            "derivatives/simulated/ses-01_task-strangerthings_acq-A2_run-1_events.tsv"
        )

        rater_a1_data = pd.read_csv(rater_a1_file, sep="\t")
        rater_a2_data = pd.read_csv(rater_a2_file, sep="\t")

        print(f"Loaded {rater_a1_file}: {len(rater_a1_data)} trials")
        print(f"Loaded {rater_a2_file}: {len(rater_a2_data)} trials")

    except FileNotFoundError as e:
        print(f"Error: Could not find one of the rater files: {e}")
        return

    # Check that both files have the same number of trials
    if len(rater_a1_data) != len(rater_a2_data):
        print("Error: Rater files have different numbers of trials!")
        return

    # Calculate ICC(2,1) - Most important for absolute agreement
    print("\nCalculating ICC(2,1)...")
    try:
        icc_results = calculate_icc(rater_a1_data, rater_a2_data)
        valence_icc, valence_pval, valence_ci = icc_results[0:3]
        arousal_icc, arousal_pval, arousal_ci = icc_results[3:6]
    except Exception as e:
        print(f"Error calculating ICC: {e}")
        return

    # Calculate agreement metrics
    print("\nCalculating agreement metrics...")
    try:
        agreement_metrics = calculate_agreement_metrics(rater_a1_data, rater_a2_data)
    except Exception as e:
        print(f"Error calculating agreement metrics: {e}")
        return

    print("\n" + "=" * 60)
    print("KEY METRICS FOR ABSOLUTE AGREEMENT")
    print("=" * 60)

    # Display ICC results with p-values and significance
    alpha = 0.05  # significance level

    print(f"ICC(2,1) for Valence: {valence_icc:.4f}")
    print(f"  p-value: {valence_pval:.4f}")
    if valence_pval < alpha:
        print(f"  *** STATISTICALLY SIGNIFICANT (p < {alpha})")
    else:
        print(f"  Not statistically significant (p >= {alpha})")
    print(f"  95% CI: {valence_ci}")

    print()
    print(f"ICC(2,1) for Arousal: {arousal_icc:.4f}")
    print(f"  p-value: {arousal_pval:.4f}")
    if arousal_pval < alpha:
        print(f"  *** STATISTICALLY SIGNIFICANT (p < {alpha})")
    else:
        print(f"  Not statistically significant (p >= {alpha})")
    print(f"  95% CI: {arousal_ci}")

    print()
    print("Agreement within thresholds:")
    val_within_1 = agreement_metrics["valence_within_1pt"]
    aro_within_1 = agreement_metrics["arousal_within_1pt"]
    print(f"  Valence within 1.0 point: {val_within_1*100:.1f}%")
    print(f"  Arousal within 1.0 point: {aro_within_1*100:.1f}%")
    val_mad = agreement_metrics["valence_MAD"]
    aro_mad = agreement_metrics["arousal_MAD"]
    print(f"  Mean difference - Valence: {val_mad:.2f}")
    print(f"  Mean difference - Arousal: {aro_mad:.2f}")
    print()

    # Enhanced recommendation considering p-values
    print("RECOMMENDATION:")
    val_significant = valence_pval < alpha
    aro_significant = arousal_pval < alpha

    if (valence_icc > 0.5 and val_significant) or (
        arousal_icc > 0.5 and aro_significant
    ):
        print(
            "✓ Adequate reliability with statistical significance - proceed with brain analysis"
        )
    elif valence_icc > 0.5 or arousal_icc > 0.5:
        print("~ Adequate ICC but not statistically significant - interpret cautiously")
    elif val_within_1 > 0.7 and aro_within_1 > 0.7:
        print("~ Marginal reliability - consider averaging raters")
    else:
        print("✗ Poor reliability - improve rating protocol first")

    # Apply Fisher z-transformation
    print("\nApplying Fisher z-transformation...")
    valence_z = fisher_z_transform(valence_icc)
    arousal_z = fisher_z_transform(arousal_icc)

    print(f"Fisher z-transformed Valence ICC: {valence_z:.4f}")
    print(f"Fisher z-transformed Arousal ICC: {arousal_z:.4f}")

    # Store results for aggregation
    results = {
        "run": ["run-1"],
        "valence_icc": [valence_icc],
        "arousal_icc": [arousal_icc],
        "valence_pval": [valence_pval],
        "arousal_pval": [arousal_pval],
        "valence_significant": [valence_pval < alpha],
        "arousal_significant": [arousal_pval < alpha],
        "valence_z": [valence_z],
        "arousal_z": [arousal_z],
    }

    results_df = pd.DataFrame(results)

    print("\nResults Summary:")
    print("-" * 50)
    print(results_df.to_string(index=False))

    # Aggregate results (mean of z-transformed values)
    print("\nAggregated Results:")
    print("-" * 30)
    mean_valence_z = np.mean(results_df["valence_z"])
    mean_arousal_z = np.mean(results_df["arousal_z"])

    # Transform back from Fisher z to get aggregated ICC
    aggregated_valence_icc = np.tanh(mean_valence_z)
    aggregated_arousal_icc = np.tanh(mean_arousal_z)

    print(f"Aggregated Valence ICC: {aggregated_valence_icc:.4f}")
    print(f"Aggregated Arousal ICC: {aggregated_arousal_icc:.4f}")

    # Interpretation guidelines
    print("\nICC Interpretation Guidelines:")
    print("< 0.50: Poor reliability")
    print("0.50-0.75: Moderate reliability")
    print("0.75-0.90: Good reliability")
    print("> 0.90: Excellent reliability")
    print()
    print("Statistical Significance Guidelines:")
    print("p < 0.05: Statistically significant reliability")
    print("p >= 0.05: No significant evidence of reliability")
    print("Note: ICC can be positive but not statistically significant")
    print("      indicating insufficient evidence for true reliability")

    # Save results to file
    output_dir = Path("derivatives/caps/interrater")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "interrater_reliability_results.tsv"
    results_df.to_csv(results_file, sep="\t", index=False)
    print(f"\nResults saved to: {results_file}")

    # Outlier timepoint detection
    print("\nDetecting outlier timepoints...")
    try:
        outlier_details = detect_outlier_timepoints(rater_a1_data, rater_a2_data)

        # Print summary
        summary = outlier_details["summary"]
        print(f"Total timepoints analyzed: {summary['total_timepoints']}")
        print(
            f"Large valence disagreements (>2 points): "
            f"{summary['large_valence_disagreements']} "
            f"({summary['pct_large_valence']:.1f}%)"
        )
        print(
            f"Large arousal disagreements (>2 points): "
            f"{summary['large_arousal_disagreements']} "
            f"({summary['pct_large_arousal']:.1f}%)"
        )
        print(f"Mean valence difference: {summary['mean_valence_difference']:.2f}")
        print(f"Mean arousal difference: {summary['mean_arousal_difference']:.2f}")
        print(f"Max valence difference: {summary['max_valence_difference']:.2f}")
        print(f"Max arousal difference: {summary['max_arousal_difference']:.2f}")

        # Print details of outlier timepoints
        if outlier_details["timepoints"]:
            print(f"\nOutlier timepoints (n={len(outlier_details['timepoints'])}):")
            print("-" * 80)
            for tp in outlier_details["timepoints"]:
                print(
                    f"Timepoint {tp['timepoint_index']} "
                    f"(onset: {tp['onset_time']}):"
                )
                if tp["valence_outlier"]:
                    print(
                        f"  Valence: Rater1={tp['valence_rater1']:.1f}, "
                        f"Rater2={tp['valence_rater2']:.1f}, "
                        f"Diff={tp['valence_difference']:.1f}"
                    )
                if tp["arousal_outlier"]:
                    print(
                        f"  Arousal: Rater1={tp['arousal_rater1']:.1f}, "
                        f"Rater2={tp['arousal_rater2']:.1f}, "
                        f"Diff={tp['arousal_difference']:.1f}"
                    )
        else:
            print(
                "\nNo major outlier timepoints detected (all disagreements <2 points)"
            )

    except Exception as e:
        print(f"Error in outlier detection: {e}")


if __name__ == "__main__":
    main()
