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


def calculate_icc_multi_rater(data, valence_cols, arousal_cols):
    """Calculate ICC(2,1) for valence and arousal ratings with multiple raters."""
    
    n_trials = len(data)
    n_val_raters = len(valence_cols)
    n_aro_raters = len(arousal_cols)
    
    # Prepare valence data for ICC calculation - long format
    # First collect all valence data per rater for standardization
    valence_by_rater = {}
    for i, col in enumerate(valence_cols):
        rater_vals = []
        for trial in range(n_trials):
            val = data[col].iloc[trial]
            try:
                val_numeric = pd.to_numeric(val, errors='coerce')
                if pd.notna(val_numeric):
                    rater_vals.append(val_numeric)
            except:
                continue
        valence_by_rater[f"R{i+1}"] = rater_vals
    
    # Apply standardization: center around 4, then z-score each rater
    valence_data_list = []
    for rater_id, vals in valence_by_rater.items():
        if len(vals) > 0:
            # Center valence around 4 (neutral point on 1-7 scale)
            centered_vals = np.array(vals) - 4
            # Z-score to standardize scale usage
            if np.std(centered_vals) > 0:
                standardized_vals = (centered_vals - np.mean(centered_vals)) / np.std(centered_vals)
            else:
                standardized_vals = centered_vals  # Keep as-is if no variance
            
            # Add to data list
            for trial_idx, std_val in enumerate(standardized_vals):
                valence_data_list.append({
                    "subject": trial_idx,
                    "rater": rater_id,
                    "valence": std_val
                })
    
    valence_data = pd.DataFrame(valence_data_list)
    
    # Prepare arousal data for ICC calculation - long format
    # First collect all arousal data per rater for standardization
    arousal_by_rater = {}
    for i, col in enumerate(arousal_cols):
        rater_vals = []
        for trial in range(n_trials):
            val = data[col].iloc[trial]
            try:
                val_numeric = pd.to_numeric(val, errors='coerce')
                if pd.notna(val_numeric):
                    rater_vals.append(val_numeric)
            except:
                continue
        arousal_by_rater[f"R{i+1}"] = rater_vals
    
    # Apply standardization: z-score each rater's arousal ratings
    arousal_data_list = []
    for rater_id, vals in arousal_by_rater.items():
        if len(vals) > 0:
            # Z-score to standardize scale usage
            vals_array = np.array(vals)
            if np.std(vals_array) > 0:
                standardized_vals = (vals_array - np.mean(vals_array)) / np.std(vals_array)
            else:
                standardized_vals = vals_array - np.mean(vals_array)  # Center if no variance
            
            # Add to data list
            for trial_idx, std_val in enumerate(standardized_vals):
                arousal_data_list.append({
                    "subject": trial_idx,
                    "rater": rater_id,
                    "arousal": std_val
                })
    
    arousal_data = pd.DataFrame(arousal_data_list)
    
    # Calculate ICC(2,1) for valence
    try:
        # Check for sufficient data and variance
        if len(valence_data) == 0:
            print("Error calculating valence ICC: No valid data")
            val_icc, val_pval, val_ci = np.nan, np.nan, [np.nan, np.nan]
        elif valence_data['valence'].nunique() <= 1:
            print("Error calculating valence ICC: Zero variance in ratings (all raters gave same scores)")
            val_icc, val_pval, val_ci = np.nan, np.nan, [np.nan, np.nan]
        elif len(valence_data.groupby('subject')) < 2:
            print("Error calculating valence ICC: Insufficient subjects with complete data")
            val_icc, val_pval, val_ci = np.nan, np.nan, [np.nan, np.nan]
        else:
            icc_valence = pg.intraclass_corr(
                data=valence_data, targets="subject", raters="rater", ratings="valence"
            )
            # Extract ICC(2,1) values - ICC2 with single rater
            icc2_valence = icc_valence[icc_valence["Type"] == "ICC2"]
            val_row = icc2_valence[icc2_valence["Description"].str.contains("Single")]
            val_icc = val_row["ICC"].iloc[0]
            val_pval = val_row["pval"].iloc[0]
            val_ci = val_row["CI95%"].iloc[0]
    except Exception as e:
        print(f"Error calculating valence ICC: {e}")
        val_icc = np.nan
        val_pval = np.nan
        val_ci = [np.nan, np.nan]

    # Calculate ICC(2,1) for arousal
    try:
        # Check for sufficient data and variance
        if len(arousal_data) == 0:
            print("Error calculating arousal ICC: No valid data")
            aro_icc, aro_pval, aro_ci = np.nan, np.nan, [np.nan, np.nan]
        elif arousal_data['arousal'].nunique() <= 1:
            print("Error calculating arousal ICC: Zero variance in ratings (all raters gave same scores)")
            aro_icc, aro_pval, aro_ci = np.nan, np.nan, [np.nan, np.nan]
        elif len(arousal_data.groupby('subject')) < 2:
            print("Error calculating arousal ICC: Insufficient subjects with complete data")
            aro_icc, aro_pval, aro_ci = np.nan, np.nan, [np.nan, np.nan]
        else:
            icc_arousal = pg.intraclass_corr(
                data=arousal_data, targets="subject", raters="rater", ratings="arousal"
            )
            # Extract ICC(2,1) values - ICC2 with single rater
            icc2_arousal = icc_arousal[icc_arousal["Type"] == "ICC2"]
            aro_row = icc2_arousal[icc2_arousal["Description"].str.contains("Single")]
            aro_icc = aro_row["ICC"].iloc[0]
            aro_pval = aro_row["pval"].iloc[0]
            aro_ci = aro_row["CI95%"].iloc[0]
    except Exception as e:
        print(f"Error calculating arousal ICC: {e}")
        aro_icc = np.nan
        aro_pval = np.nan
        aro_ci = [np.nan, np.nan]

    return val_icc, val_pval, val_ci, aro_icc, aro_pval, aro_ci, n_val_raters, n_aro_raters


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


def load_combined_annotation_files():
    """Load all combined annotation CSV files."""
    annotation_dir = Path("derivatives/annotations")
    annotation_files = list(annotation_dir.glob("S*.csv"))
    
    if not annotation_files:
        print("No combined annotation files found in derivatives/annotations/")
        return []
    
    return sorted(annotation_files)


def create_aggregated_time_series_multi_rater(data, valence_cols, arousal_cols, valence_reliable, arousal_reliable):
    """
    Create aggregated emotion time series for reliable runs with multiple raters.
    Z-score and average ratings across all available raters for reliable emotions.
    """
    results = {}
    
    if valence_reliable and len(valence_cols) >= 2:
        # Z-score valence ratings for each rater and average
        val_z_scores = []
        for col in valence_cols:
            if not data[col].isna().all():  # Skip columns that are all NaN
                val_z = stats.zscore(data[col].dropna())
                # Pad with NaN if needed to match original length
                if len(val_z) < len(data):
                    val_full = np.full(len(data), np.nan)
                    val_full[~data[col].isna()] = val_z
                    val_z_scores.append(val_full)
                else:
                    val_z_scores.append(val_z)
        
        if val_z_scores:
            # Average z-scored ratings across raters (ignoring NaN)
            valence_aggregated = np.nanmean(val_z_scores, axis=0)
            results['valence_timeseries'] = valence_aggregated
            print(f"✓ Created aggregated valence time series (z-scored and averaged across {len(valence_cols)} raters)")
        else:
            results['valence_timeseries'] = None
            print("✗ Valence reliable but no valid data - no aggregated time series created")
    else:
        results['valence_timeseries'] = None
        print("✗ Valence not reliable - no aggregated time series created")
    
    if arousal_reliable and len(arousal_cols) >= 2:
        # Z-score arousal ratings for each rater and average
        aro_z_scores = []
        for col in arousal_cols:
            if not data[col].isna().all():  # Skip columns that are all NaN
                aro_z = stats.zscore(data[col].dropna())
                # Pad with NaN if needed to match original length
                if len(aro_z) < len(data):
                    aro_full = np.full(len(data), np.nan)
                    aro_full[~data[col].isna()] = aro_z
                    aro_z_scores.append(aro_full)
                else:
                    aro_z_scores.append(aro_z)
        
        if aro_z_scores:
            # Average z-scored ratings across raters (ignoring NaN)
            arousal_aggregated = np.nanmean(aro_z_scores, axis=0)
            results['arousal_timeseries'] = arousal_aggregated
            print(f"✓ Created aggregated arousal time series (z-scored and averaged across {len(arousal_cols)} raters)")
        else:
            results['arousal_timeseries'] = None
            print("✗ Arousal reliable but no valid data - no aggregated time series created")
    else:
        results['arousal_timeseries'] = None
        print("✗ Arousal not reliable - no aggregated time series created")
    
    return results


def process_single_run(csv_file):
    """Process a single combined annotation CSV file for IRR."""
    print(f"\nProcessing: {csv_file.name}")
    print("-" * 50)
    
    try:
        data = pd.read_csv(csv_file)
        print(f"Loaded {len(data)} timepoints")
        
        # Check if we have at least 2 raters for both valence and arousal
        valence_cols = [col for col in data.columns if col.startswith('valence_')]
        arousal_cols = [col for col in data.columns if col.startswith('arousal_')]
        
        if len(valence_cols) < 2 and len(arousal_cols) < 2:
            print(f"⚠️  Insufficient raters: {len(valence_cols)} valence, {len(arousal_cols)} arousal raters")
            print("Need at least 2 raters for one emotion dimension")
            return None
        
        print(f"Found {len(valence_cols)} valence raters, {len(arousal_cols)} arousal raters")
        
        return {
            'run_name': csv_file.stem,
            'data': data,
            'valence_cols': valence_cols,
            'arousal_cols': arousal_cols,
            'n_timepoints': len(data),
            'n_valence_raters': len(valence_cols),
            'n_arousal_raters': len(arousal_cols)
        }
        
    except Exception as e:
        print(f"Error processing {csv_file.name}: {e}")
        return None


def main():
    print("Inter-rater Reliability Analysis for Combined Annotations")
    print("=" * 60)
    print("Following Shrout & Fleiss (1979) ICC(2,1) methodology")
    print("Significance threshold: p < 0.05")
    print()

    # Load all combined annotation files
    annotation_files = load_combined_annotation_files()
    if not annotation_files:
        print("No annotation files found to process.")
        return
    
    print(f"Found {len(annotation_files)} combined annotation files to process")
    
    # Process each file and collect results
    all_results = []
    
    for csv_file in annotation_files:
        run_data = process_single_run(csv_file)
        if run_data is None:
            continue
        
        run_name = run_data['run_name']
        data = run_data['data']
        valence_cols = run_data['valence_cols']
        arousal_cols = run_data['arousal_cols']
        
        print(f"\nAnalyzing {run_data['n_timepoints']} timepoints for reliability...")

        # STEP 1: Calculate ICC(2,1) for absolute agreement
        print("\nSTEP 1: Calculating ICC(2,1) for absolute agreement...")
        try:
            icc_results = calculate_icc_multi_rater(data, valence_cols, arousal_cols)
            valence_icc, valence_pval, valence_ci, arousal_icc, arousal_pval, arousal_ci, n_val_raters, n_aro_raters = icc_results
        except Exception as e:
            print(f"Error calculating ICC: {e}")
            continue

        # Determine reliability based on ICC thresholds for averaging
        alpha = 0.05
        
        # ICC interpretation for averaging purposes
        def interpret_icc_for_averaging(icc_val, p_val):
            if pd.isna(icc_val) or pd.isna(p_val):
                return "Cannot assess", "NA - data issues"
            elif p_val >= alpha:
                return "Poor", "Not significant - don't average"
            elif icc_val >= 0.75:
                return "Excellent", "Definitely average"
            elif icc_val >= 0.60:
                return "Good", "Safe to average"
            elif icc_val >= 0.40:
                return "Fair", "Use caution - averaging questionable"
            else:
                return "Poor", "Don't average"
        
        val_reliability, val_recommendation = interpret_icc_for_averaging(valence_icc, valence_pval)
        aro_reliability, aro_recommendation = interpret_icc_for_averaging(arousal_icc, arousal_pval)
        
        # Simple binary for downstream analysis
        valence_suitable_for_averaging = val_reliability in ["Excellent", "Good"] 
        arousal_suitable_for_averaging = aro_reliability in ["Excellent", "Good"]

        print(f"Valence ICC(2,1): {valence_icc:.4f} (p = {valence_pval:.4f}) [{n_val_raters} raters]")
        print(f"  Reliability: {val_reliability} - {val_recommendation}")
        
        print(f"Arousal ICC(2,1): {arousal_icc:.4f} (p = {arousal_pval:.4f}) [{n_aro_raters} raters]")
        print(f"  Reliability: {aro_reliability} - {aro_recommendation}")

        # Overall recommendation for this run
        if valence_suitable_for_averaging and arousal_suitable_for_averaging:
            run_recommendation = "AVERAGE BOTH EMOTIONS"
            reason = "Both valence and arousal have good/excellent reliability"
        elif valence_suitable_for_averaging:
            run_recommendation = "AVERAGE VALENCE ONLY"
            reason = "Only valence has sufficient reliability for averaging"
        elif arousal_suitable_for_averaging:
            run_recommendation = "AVERAGE AROUSAL ONLY"
            reason = "Only arousal has sufficient reliability for averaging"
        else:
            run_recommendation = "DON'T AVERAGE"
            reason = "Neither emotion dimension has sufficient reliability"

        print(f"\nRUN RECOMMENDATION: {run_recommendation}")
        print(f"Reason: {reason}")

        # STEP 2: Create aggregated time series for suitable emotions
        print("\nSTEP 2: Creating aggregated emotion time series...")
        
        if not (valence_suitable_for_averaging or arousal_suitable_for_averaging):
            print("❌ NO SUITABLE EMOTIONS FOR AVERAGING")
            correlation = np.nan
            corr_pvalue = np.nan
            n_concerns = 0
            exclusion_recommended = True
        else:
            aggregated_series = create_aggregated_time_series_multi_rater(
                data, valence_cols, arousal_cols, valence_suitable_for_averaging, arousal_suitable_for_averaging
            )

            # STEP 3: Compute Spearman correlation between valence and arousal
            print("\nSTEP 3: Computing valence-arousal correlation...")
            
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
            
            onset_times = range(len(data))  # Use index as onset times
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
            
            n_concerns = len(dynamics_assessment['concern_flags'])
            exclusion_recommended = dynamics_assessment['exclusion_recommended']

        # Collect results for this run
        emotions_for_averaging = []
        if valence_suitable_for_averaging:
            emotions_for_averaging.append("valence")
        if arousal_suitable_for_averaging:
            emotions_for_averaging.append("arousal")

        run_results = {
            "run": run_name,
            "n_clips": run_data['n_timepoints'],
            "n_valence_raters": run_data['n_valence_raters'],
            "n_arousal_raters": run_data['n_arousal_raters'],
            "valence_icc": valence_icc if not pd.isna(valence_icc) else "NA",
            "valence_pval": valence_pval if not pd.isna(valence_pval) else "NA",
            "valence_reliability": val_reliability,
            "valence_suitable_for_averaging": valence_suitable_for_averaging,
            "arousal_icc": arousal_icc if not pd.isna(arousal_icc) else "NA",
            "arousal_pval": arousal_pval if not pd.isna(arousal_pval) else "NA",
            "arousal_reliability": aro_reliability,
            "arousal_suitable_for_averaging": arousal_suitable_for_averaging,
            "valence_arousal_correlation": correlation if not pd.isna(correlation) else "NA",
            "valence_arousal_corr_pval": corr_pvalue if not pd.isna(corr_pvalue) else "NA",
            "run_recommendation": run_recommendation,
            "emotions_for_averaging": ', '.join(emotions_for_averaging) if emotions_for_averaging else 'none'
        }
        
        all_results.append(run_results)

    # Save comprehensive results for all runs
    print(f"\n{'='*60}")
    print("SAVING COMPREHENSIVE RESULTS")
    print("="*60)
    
    if not all_results:
        print("No runs were successfully processed.")
        return
    
    results_df = pd.DataFrame(all_results)

    # Save results to file
    output_dir = Path("derivatives/caps/interrater")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "interrater_reliability_results.tsv"
    results_df.to_csv(results_file, sep="\t", index=False)
    print(f"Results saved to: {results_file}")

    # Print summary statistics
    print(f"\nSUMMARY - CAN I AVERAGE THE RATERS? ({len(all_results)} RUNS)")
    print("-" * 60)
    
    suitable_valence = results_df['valence_suitable_for_averaging'].sum()
    suitable_arousal = results_df['arousal_suitable_for_averaging'].sum()
    suitable_both = (results_df['valence_suitable_for_averaging'] & results_df['arousal_suitable_for_averaging']).sum()
    
    average_both = (results_df['run_recommendation'] == 'AVERAGE BOTH EMOTIONS').sum()
    average_valence_only = (results_df['run_recommendation'] == 'AVERAGE VALENCE ONLY').sum()
    average_arousal_only = (results_df['run_recommendation'] == 'AVERAGE AROUSAL ONLY').sum()
    dont_average = (results_df['run_recommendation'] == "DON'T AVERAGE").sum()
    
    print(f"✅ AVERAGE BOTH emotions: {average_both}/{len(all_results)} runs ({average_both/len(all_results)*100:.1f}%)")
    print(f"⚠️  AVERAGE VALENCE only: {average_valence_only}/{len(all_results)} runs ({average_valence_only/len(all_results)*100:.1f}%)")
    print(f"⚠️  AVERAGE AROUSAL only: {average_arousal_only}/{len(all_results)} runs ({average_arousal_only/len(all_results)*100:.1f}%)")
    print(f"❌ DON'T AVERAGE either: {dont_average}/{len(all_results)} runs ({dont_average/len(all_results)*100:.1f}%)")
    
    print(f"\nBy emotion dimension:")
    print(f"Valence suitable for averaging: {suitable_valence}/{len(all_results)} ({suitable_valence/len(all_results)*100:.1f}%)")
    print(f"Arousal suitable for averaging: {suitable_arousal}/{len(all_results)} ({suitable_arousal/len(all_results)*100:.1f}%)")

    print("\n" + "=" * 60)
    print("INTER-RATER RELIABILITY ANALYSIS COMPLETE")
    print("=" * 60)
    print("BOTTOM LINE:")
    if average_both > 0:
        print(f"✅ You can safely average BOTH emotions for {average_both} runs")
    if average_valence_only > 0:
        print(f"⚠️  You can average VALENCE only for {average_valence_only} additional runs")
    if average_arousal_only > 0:
        print(f"⚠️  You can average AROUSAL only for {average_arousal_only} additional runs")
    if dont_average > 0:
        print(f"❌ DON'T average either emotion for {dont_average} runs (poor reliability)")


if __name__ == "__main__":
    main()
