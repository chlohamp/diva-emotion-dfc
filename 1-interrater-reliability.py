import pandas as pd
import numpy as np
import pingouin as pg
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def fisher_z_transform(r):
    """Apply Fisher z-transformation to correlation coefficients."""
    return np.arctanh(r)


def create_aggregated_time_series(rater_a1_data, rater_a2_data, valence_reliable, arousal_reliable):
    """
    Create aggregated emotion time series for reliable runs.
    Z-score and average ratings across raters for reliable emotions.
    """
    results = {}
    
    if valence_reliable:
        # Z-score valence ratings for each rater
        val_r1_z = stats.zscore(rater_a1_data["valence"])
        val_r2_z = stats.zscore(rater_a2_data["valence"])
        
        # Average z-scored ratings
        valence_aggregated = (val_r1_z + val_r2_z) / 2
        results['valence_timeseries'] = valence_aggregated
        print("✓ Created aggregated valence time series (z-scored and averaged)")
    else:
        results['valence_timeseries'] = None
        print("✗ Valence not reliable - no aggregated time series created")
    
    if arousal_reliable:
        # Z-score arousal ratings for each rater
        aro_r1_z = stats.zscore(rater_a1_data["arousal"])
        aro_r2_z = stats.zscore(rater_a2_data["arousal"])
        
        # Average z-scored ratings
        arousal_aggregated = (aro_r1_z + aro_r2_z) / 2
        results['arousal_timeseries'] = arousal_aggregated
        print("✓ Created aggregated arousal time series (z-scored and averaged)")
    else:
        results['arousal_timeseries'] = None
        print("✗ Arousal not reliable - no aggregated time series created")
    
    return results


def compute_valence_arousal_correlation(valence_ts, arousal_ts):
    """
    Compute Spearman correlation between valence and arousal time series.
    """
    if valence_ts is None or arousal_ts is None:
        return None, None, "Cannot compute correlation - one or both time series not reliable"
    
    # Compute Spearman correlation
    correlation, p_value = stats.spearmanr(valence_ts, arousal_ts)
    
    # Interpret correlation strength
    abs_corr = abs(correlation)
    if abs_corr < 0.1:
        strength = "negligible"
    elif abs_corr < 0.3:
        strength = "weak"
    elif abs_corr < 0.5:
        strength = "moderate"
    elif abs_corr < 0.7:
        strength = "strong"
    else:
        strength = "very strong"
    
    direction = "positive" if correlation > 0 else "negative"
    interpretation = f"{strength} {direction} correlation"
    
    return correlation, p_value, interpretation


def assess_emotional_dynamics(valence_ts, arousal_ts, onset_times):
    """
    Assess whether emotional changes align with expected affective dynamics.
    Identify potential atypical patterns.
    """
    assessment = {
        'patterns_detected': [],
        'concern_flags': [],
        'overall_assessment': '',
        'exclusion_recommended': False
    }
    
    if valence_ts is None or arousal_ts is None:
        assessment['overall_assessment'] = "Cannot assess - insufficient reliable data"
        assessment['exclusion_recommended'] = True
        return assessment
    
    # Check for excessive volatility (rapid changes)
    val_changes = np.abs(np.diff(valence_ts))
    aro_changes = np.abs(np.diff(arousal_ts))
    
    val_volatility = np.mean(val_changes)
    aro_volatility = np.mean(aro_changes)
    
    # Check for patterns that might indicate problems
    
    # 1. Excessive volatility
    volatility_threshold = 1.5  # z-score units
    if val_volatility > volatility_threshold:
        assessment['concern_flags'].append(f"High valence volatility (avg change: {val_volatility:.2f})")
    if aro_volatility > volatility_threshold:
        assessment['concern_flags'].append(f"High arousal volatility (avg change: {aro_volatility:.2f})")
    
    # 2. Flat/unchanging ratings (potential disengagement)
    val_range = np.ptp(valence_ts)  # peak-to-peak (range)
    aro_range = np.ptp(arousal_ts)
    
    flat_threshold = 0.5  # z-score units
    if val_range < flat_threshold:
        assessment['concern_flags'].append(f"Very flat valence ratings (range: {val_range:.2f})")
    if aro_range < flat_threshold:
        assessment['concern_flags'].append(f"Very flat arousal ratings (range: {aro_range:.2f})")
    
    # 3. Check for obvious outliers
    val_outliers = np.abs(valence_ts) > 3  # beyond 3 standard deviations
    aro_outliers = np.abs(arousal_ts) > 3
    
    if np.any(val_outliers):
        n_outliers = np.sum(val_outliers)
        assessment['concern_flags'].append(f"Extreme valence values detected ({n_outliers} timepoints)")
    if np.any(aro_outliers):
        n_outliers = np.sum(aro_outliers)
        assessment['concern_flags'].append(f"Extreme arousal values detected ({n_outliers} timepoints)")
    
    # Overall assessment
    n_concerns = len(assessment['concern_flags'])
    if n_concerns == 0:
        assessment['overall_assessment'] = "Normal emotional dynamics - no concerns detected"
        assessment['exclusion_recommended'] = False
    elif n_concerns <= 2:
        assessment['overall_assessment'] = "Minor concerns detected - monitor but likely usable"
        assessment['exclusion_recommended'] = False
    else:
        assessment['overall_assessment'] = "Multiple concerns detected - consider exclusion"
        assessment['exclusion_recommended'] = True
    
    return assessment


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
    print("Following Shrout & Fleiss (1979) ICC(2,1) methodology")
    print("Significance threshold: p < 0.05")
    print()

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

    print(f"\nAnalyzing {len(rater_a1_data)} TR-length clips for reliability...")

    # STEP 1: Calculate ICC(2,1) for absolute agreement
    print("\nSTEP 1: Calculating ICC(2,1) for absolute agreement...")
    print("-" * 50)
    try:
        icc_results = calculate_icc(rater_a1_data, rater_a2_data)
        valence_icc, valence_pval, valence_ci = icc_results[0:3]
        arousal_icc, arousal_pval, arousal_ci = icc_results[3:6]
    except Exception as e:
        print(f"Error calculating ICC: {e}")
        return

    # Determine reliability (p < 0.05)
    alpha = 0.05
    valence_reliable = valence_pval < alpha
    arousal_reliable = arousal_pval < alpha

    print(f"Valence ICC(2,1): {valence_icc:.4f} (p = {valence_pval:.4f})")
    if valence_reliable:
        print("  ✓ SIGNIFICANT RELIABILITY - proceed with aggregation")
    else:
        print("  ✗ NOT RELIABLE - exclude from further analysis")
    
    print(f"Arousal ICC(2,1): {arousal_icc:.4f} (p = {arousal_pval:.4f})")
    if arousal_reliable:
        print("  ✓ SIGNIFICANT RELIABILITY - proceed with aggregation")
    else:
        print("  ✗ NOT RELIABLE - exclude from further analysis")

    # STEP 2: Create aggregated time series for reliable emotions
    print("\nSTEP 2: Creating aggregated emotion time series...")
    print("-" * 50)
    
    if not (valence_reliable or arousal_reliable):
        print("❌ NO RELIABLE EMOTIONS - RUN SHOULD BE EXCLUDED")
        print("Recommendation: This run lacks sufficient inter-rater reliability")
        print("for both valence and arousal and should be excluded from analysis.")
        return
    
    aggregated_series = create_aggregated_time_series(
        rater_a1_data, rater_a2_data, valence_reliable, arousal_reliable
    )

    # STEP 3: Compute Spearman correlation between valence and arousal
    print("\nSTEP 3: Computing valence-arousal correlation...")
    print("-" * 50)
    
    correlation, corr_pvalue, interpretation = compute_valence_arousal_correlation(
        aggregated_series['valence_timeseries'], 
        aggregated_series['arousal_timeseries']
    )
    
    if correlation is not None:
        print(f"Spearman correlation: r = {correlation:.4f} (p = {corr_pvalue:.4f})")
        print(f"Interpretation: {interpretation}")
        
        if corr_pvalue < 0.05:
            print("  ✓ Statistically significant covariation detected")
        else:
            print("  - No significant covariation between valence and arousal")
    else:
        print("Cannot compute correlation - insufficient reliable data")

    # STEP 4: Assess emotional dynamics and identify atypical patterns
    print("\nSTEP 4: Assessing emotional dynamics...")
    print("-" * 50)
    
    onset_times = rater_a1_data.get('onset', range(len(rater_a1_data)))
    dynamics_assessment = assess_emotional_dynamics(
        aggregated_series['valence_timeseries'],
        aggregated_series['arousal_timeseries'],
        onset_times
    )
    
    print(f"Overall assessment: {dynamics_assessment['overall_assessment']}")
    
    if dynamics_assessment['concern_flags']:
        print("\nConcerns detected:")
        for concern in dynamics_assessment['concern_flags']:
            print(f"  ⚠️  {concern}")
    else:
        print("✓ No concerning patterns detected")
    
    # STEP 5: Final recommendation
    print("\nSTEP 5: Final recommendation...")
    print("=" * 50)
    
    reliable_emotions = []
    if valence_reliable:
        reliable_emotions.append("valence")
    if arousal_reliable:
        reliable_emotions.append("arousal")
    
    if dynamics_assessment['exclusion_recommended']:
        final_recommendation = "EXCLUDE RUN"
        reason = "Atypical emotional dynamics detected"
    elif not reliable_emotions:
        final_recommendation = "EXCLUDE RUN"
        reason = "No reliable emotion ratings"
    elif len(reliable_emotions) == 1:
        final_recommendation = "PROCEED WITH CAUTION"
        reason = f"Only {reliable_emotions[0]} ratings are reliable"
    else:
        final_recommendation = "PROCEED WITH ANALYSIS"
        reason = "Both emotion dimensions show adequate reliability"
    
    print(f"RECOMMENDATION: {final_recommendation}")
    print(f"Reason: {reason}")
    
    if reliable_emotions:
        print(f"Reliable emotions for analysis: {', '.join(reliable_emotions)}")

    # Save detailed results
    print("\nSaving results...")
    print("-" * 30)
    
    # Prepare comprehensive results
    results = {
        "run": ["run-1"],
        "n_clips": [len(rater_a1_data)],
        "valence_icc": [valence_icc],
        "valence_pval": [valence_pval],
        "valence_reliable": [valence_reliable],
        "arousal_icc": [arousal_icc],
        "arousal_pval": [arousal_pval],
        "arousal_reliable": [arousal_reliable],
        "valence_arousal_correlation": [correlation if correlation is not None else np.nan],
        "valence_arousal_corr_pval": [corr_pvalue if corr_pvalue is not None else np.nan],
        "n_concerns": [len(dynamics_assessment['concern_flags'])],
        "exclusion_recommended": [dynamics_assessment['exclusion_recommended']],
        "final_recommendation": [final_recommendation],
        "reliable_emotions": [', '.join(reliable_emotions) if reliable_emotions else 'none']
    }

    results_df = pd.DataFrame(results)

    # Save results to file
    output_dir = Path("derivatives/caps/interrater")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "interrater_reliability_results.tsv"
    results_df.to_csv(results_file, sep="\t", index=False)
    print(f"Results saved to: {results_file}")

    # Save aggregated time series if available
    if aggregated_series['valence_timeseries'] is not None or aggregated_series['arousal_timeseries'] is not None:
        timeseries_data = {
            'onset': onset_times,
            'valence_aggregated': aggregated_series['valence_timeseries'] if aggregated_series['valence_timeseries'] is not None else [np.nan] * len(onset_times),
            'arousal_aggregated': aggregated_series['arousal_timeseries'] if aggregated_series['arousal_timeseries'] is not None else [np.nan] * len(onset_times),
            'valence_reliable': [valence_reliable] * len(onset_times),
            'arousal_reliable': [arousal_reliable] * len(onset_times)
        }
        
        timeseries_df = pd.DataFrame(timeseries_data)
        timeseries_file = output_dir / "aggregated_emotion_timeseries.tsv"
        timeseries_df.to_csv(timeseries_file, sep="\t", index=False)
        print(f"Aggregated time series saved to: {timeseries_file}")

    print("\n" + "=" * 60)
    print("INTER-RATER RELIABILITY ANALYSIS COMPLETE")
    print("=" * 60)
    print("Next steps:")
    if final_recommendation == "PROCEED WITH ANALYSIS":
        print("✓ This run can proceed to brain analysis")
        print("✓ Use aggregated emotion time series for further analysis")
    elif final_recommendation == "PROCEED WITH CAUTION":
        print("⚠️  Proceed with caution - only partial emotion data reliable")
    else:
        print("❌ Exclude this run from further analysis")
        print("❌ Consider obtaining additional ratings from new annotators")


if __name__ == "__main__":
    main()
