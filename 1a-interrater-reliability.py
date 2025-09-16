import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg


def fisher_z_transform(r):
    """Apply Fisher z-transformation to correlation coefficients."""
    return np.arctanh(r)


def calculate_pearson_correlation(rater_a1_data, rater_a2_data):
    """Calculate Pearson correlation for valence and arousal ratings."""

    # Calculate correlations
    valence_corr, valence_p = stats.pearsonr(
        rater_a1_data["valence"], rater_a2_data["valence"]
    )

    arousal_corr, arousal_p = stats.pearsonr(
        rater_a1_data["arousal"], rater_a2_data["arousal"]
    )

    return valence_corr, arousal_corr, valence_p, arousal_p


def calculate_icc(rater_a1_data, rater_a2_data):
    """Calculate ICC(2,1) using pingouin for valence and arousal ratings."""

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

    # Get single rater ICC values - ICC2 single random raters
    val_icc = icc2_valence[icc2_valence["Description"].str.contains("Single")][
        "ICC"
    ].iloc[0]
    aro_icc = icc2_arousal[icc2_arousal["Description"].str.contains("Single")][
        "ICC"
    ].iloc[0]

    return val_icc, aro_icc


def main():
    print("Inter-rater Reliability Analysis")
    print("=" * 40)

    # Load the two rater files
    try:
        rater_a1_file = "ses-01_task-strangerthings_acq-A1_run-1_events.tsv"
        rater_a2_file = "ses-01_task-strangerthings_acq-A2_run-1_events.tsv"

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

    # Calculate correlations first
    print("\nCalculating Pearson correlations...")
    try:
        val_corr, aro_corr, val_p, aro_p = calculate_pearson_correlation(
            rater_a1_data, rater_a2_data
        )

        print(f"Valence correlation: r = {val_corr:.4f}, p = {val_p:.4f}")
        print(f"Arousal correlation: r = {aro_corr:.4f}, p = {aro_p:.4f}")

    except Exception as e:
        print(f"Error calculating correlations: {e}")
        return

    # Calculate ICC(2,1) using pingouin
    print("\nCalculating ICC(2,1) using pingouin...")
    try:
        valence_icc, arousal_icc = calculate_icc(rater_a1_data, rater_a2_data)

        print(f"ICC(2,1) for Valence: {valence_icc:.4f}")
        print(f"ICC(2,1) for Arousal: {arousal_icc:.4f}")

    except Exception as e:
        print(f"Error calculating ICC: {e}")
        return

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
        "valence_z": [valence_z],
        "arousal_z": [arousal_z],
        "valence_corr": [val_corr],
        "arousal_corr": [aro_corr],
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

    # Save results to file
    results_file = "interrater_reliability_results.tsv"
    results_df.to_csv(results_file, sep="\t", index=False)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
