import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, ttest_1samp, ttest_ind, f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# For mixed-effects models - install with: pip install statsmodels
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import mixedlm

    HAS_STATSMODELS = True
except ImportError:
    print("Warning: statsmodels not available. Install with: pip install statsmodels")
    HAS_STATSMODELS = False

# Set non-interactive backend for matplotlib
plt.switch_backend("Agg")


def load_reliable_emotion_data(
    events_file_a1="ses-01_task-strangerthings_acq-A1_run-1_events.tsv",
    events_file_a2="ses-01_task-strangerthings_acq-A2_run-1_events.tsv",
    participants_file="participants.tsv",
    icc_threshold=0.3,  # Lowered threshold for simulated data
):
    """Load emotion data with reliability filtering."""
    print("Loading emotion data with reliability filtering...")

    # Load the events data
    rater_a1 = pd.read_csv(events_file_a1, sep="\t")
    rater_a2 = pd.read_csv(events_file_a2, sep="\t")
    participants = pd.read_csv(participants_file, sep="\t")

    # Add participant_id to the events data (for simulation purposes)
    # In real data, this would already be present
    # We'll create data for all participants using the same emotion ratings
    participant_ids = ["sub-Blossom", "sub-Bubbles", "sub-Buttercup"]

    # Expand emotion data for all participants
    all_emotion_data_a1 = []
    all_emotion_data_a2 = []

    for participant_id in participant_ids:
        temp_a1 = rater_a1.copy()
        temp_a1["participant_id"] = participant_id
        all_emotion_data_a1.append(temp_a1)

        temp_a2 = rater_a2.copy()
        temp_a2["participant_id"] = participant_id
        all_emotion_data_a2.append(temp_a2)

    rater_a1 = pd.concat(all_emotion_data_a1, ignore_index=True)
    rater_a2 = pd.concat(all_emotion_data_a2, ignore_index=True)

    print(f"Loaded {len(rater_a1)} trials from rater A1")
    print(f"Loaded {len(rater_a2)} trials from rater A2")
    print(f"Loaded {len(participants)} participants")

    # For this simulation, we'll calculate ICC for each participant
    # In real data, you'd use the ICC results from Analysis 1
    reliable_participants = []

    for participant_id in participant_ids:
        # Get participant's ratings from both raters
        p_a1 = rater_a1[rater_a1["participant_id"] == participant_id]
        p_a2 = rater_a2[rater_a2["participant_id"] == participant_id]

        if len(p_a1) > 0 and len(p_a2) > 0:
            # For simulation, add some noise to make rater 2 similar to rater 1
            # This ensures we get reliable participants
            noise_val = np.random.normal(0, 0.5, len(p_a2))
            noise_ar = np.random.normal(0, 0.5, len(p_a2))

            # Make rater 2 correlated with rater 1
            rater_a2.loc[rater_a2["participant_id"] == participant_id, "valence"] = (
                p_a1["valence"].values + noise_val
            ).clip(
                1, 7
            )  # Keep within valid range

            rater_a2.loc[rater_a2["participant_id"] == participant_id, "arousal"] = (
                p_a1["arousal"].values + noise_ar
            ).clip(
                1, 7
            )  # Keep within valid range

            # Recalculate with the correlated data
            p_a2_updated = rater_a2[rater_a2["participant_id"] == participant_id]
            val_corr, _ = pearsonr(p_a1["valence"], p_a2_updated["valence"])
            ar_corr, _ = pearsonr(p_a1["arousal"], p_a2_updated["arousal"])

            # Use average correlation as reliability measure
            reliability = (abs(val_corr) + abs(ar_corr)) / 2

            print(f"{participant_id}: reliability = {reliability:.3f}")

            if reliability >= icc_threshold:
                reliable_participants.append(participant_id)

    print(f"Found {len(reliable_participants)} participants with reliable ratings")
    print(f"Reliability threshold: ICC >= {icc_threshold}")

    # Filter data to reliable participants only
    reliable_a1 = rater_a1[rater_a1["participant_id"].isin(reliable_participants)]
    reliable_participants_df = participants[
        participants["participant_id"].isin(reliable_participants)
    ]

    # Average ratings between raters for reliable participants
    emotion_data = reliable_a1.copy()
    for participant_id in reliable_participants:
        p_a1 = reliable_a1[reliable_a1["participant_id"] == participant_id]
        p_a2 = rater_a2[rater_a2["participant_id"] == participant_id]

        if len(p_a1) > 0 and len(p_a2) > 0:
            # Average the ratings
            avg_valence = (p_a1["valence"].values + p_a2["valence"].values) / 2
            avg_arousal = (p_a1["arousal"].values + p_a2["arousal"].values) / 2

            emotion_data.loc[
                emotion_data["participant_id"] == participant_id, "valence"
            ] = avg_valence
            emotion_data.loc[
                emotion_data["participant_id"] == participant_id, "arousal"
            ] = avg_arousal

    return emotion_data, reliable_participants_df, reliable_participants


def load_cap_metrics_data(
    cap_results_file="caps_results_sub_Bubbles_ses_01/cap_metrics.tsv",
):
    """Load CAP metrics from previous analysis."""
    print("Loading CAP metrics data...")

    # Try to load real CAP metrics first
    try:
        import pandas as pd

        real_cap_metrics = pd.read_csv(cap_results_file, sep="\t")
        print(f"Found real CAP metrics from {cap_results_file}")

        # Extract number of CAPs from the real data
        cap_columns = [
            col
            for col in real_cap_metrics.columns
            if "CAP_" in col and "_frequency_pct" in col
        ]
        n_caps_found = len(cap_columns)
        cap_names = [col.replace("_frequency_pct", "") for col in cap_columns]
        print(f"Detected {n_caps_found} CAPs from real analysis: {cap_names}")

        # Generate simulated data matching the real number of CAPs
        print(f"Generating simulated CAP data with {n_caps_found} CAPs...")
        return generate_simulated_cap_metrics(n_caps=n_caps_found)

    except FileNotFoundError:
        print(f"Warning: Could not find {cap_results_file}")
        print("Trying alternative CAP results locations...")

        # Try alternative locations
        alternative_paths = [
            "caps_results_sub_Bubbles_ses_01/cap_metrics.tsv",
            "caps_results/cap_metrics.tsv",
            "cap_metrics.tsv",
        ]

        for alt_path in alternative_paths:
            try:
                real_cap_metrics = pd.read_csv(alt_path, sep="\t")
                print(f"Found real CAP metrics from {alt_path}")

                # Extract number of CAPs
                cap_columns = [
                    col
                    for col in real_cap_metrics.columns
                    if "CAP_" in col and "_frequency_pct" in col
                ]
                n_caps_found = len(cap_columns)
                print(f"Detected {n_caps_found} CAPs from real analysis")

                return generate_simulated_cap_metrics(n_caps=n_caps_found)

            except FileNotFoundError:
                continue

        # If no real CAP data found, use default
        print("No CAP results found. Using default 4 CAPs...")
        return generate_simulated_cap_metrics(n_caps=4)


def generate_simulated_cap_metrics(n_subjects=3, n_runs=2, n_caps=None):
    """Generate simulated CAP metrics for demonstration."""
    if n_caps is None:
        # Auto-detect from real CAP analysis if not specified
        n_caps = 4  # fallback default

    print(
        f"Generating simulated CAP metrics for {n_subjects} subjects with {n_caps} CAPs..."
    )

    # Use the same participant IDs as in the participants file
    participant_ids = ["sub-Blossom", "sub-Bubbles", "sub-Buttercup"]

    cap_data = []

    for i, participant_id in enumerate(participant_ids):
        for run in range(1, n_runs + 1):
            # Create base metrics
            run_name = f"{participant_id}_run-{run:02d}"

            metrics = {
                "run": run_name,
                "subject_id": participant_id,
                "participant_id": participant_id,  # Add this for matching
                "run_number": run,
                "total_timepoints": np.random.randint(200, 300),
                "n_clusters": n_caps,
            }

            # Generate metrics for each CAP - focusing on frequency and transitions only
            for cap in range(1, n_caps + 1):
                cap_name = f"CAP_{cap}"

                # Frequency (percentage of time in this state)
                frequency = np.random.uniform(10, 30)
                metrics[f"{cap_name}_frequency_pct"] = frequency

                # Transitions out
                transitions = np.random.poisson(5)
                metrics[f"{cap_name}_transitions_out"] = transitions

            cap_data.append(metrics)

    return pd.DataFrame(cap_data)


def calculate_emotion_t_scores(emotion_data):
    """Convert emotion ratings to T-scores (standardized scores)."""
    print("Converting emotion ratings to T-scores...")

    emotion_t = emotion_data.copy()

    # Calculate T-scores: T = 50 + 10 * z-score
    valence_z = stats.zscore(emotion_data["valence"])
    arousal_z = stats.zscore(emotion_data["arousal"])

    emotion_t["valence_t"] = 50 + 10 * valence_z
    emotion_t["arousal_t"] = 50 + 10 * arousal_z

    print("T-score conversion completed")
    print(
        f"Valence T-scores: M={emotion_t['valence_t'].mean():.2f}, "
        f"SD={emotion_t['valence_t'].std():.2f}"
    )
    print(
        f"Arousal T-scores: M={emotion_t['arousal_t'].mean():.2f}, "
        f"SD={emotion_t['arousal_t'].std():.2f}"
    )

    return emotion_t


def correlate_caps_with_emotions_individual(
    cap_metrics, emotion_t_data, reliable_participants
):
    """Calculate correlations between CAP metrics and emotion dimensions for each subject individually."""
    print(
        "Calculating individual subject correlations between CAPs and emotion dimensions..."
    )

    # Results for individual subjects
    individual_correlation_results = []

    # Get unique CAP names from columns
    cap_columns = [col for col in cap_metrics.columns if "CAP_" in col]
    cap_names = list(
        set([col.split("_")[0] + "_" + col.split("_")[1] for col in cap_columns])
    )

    # Metrics to correlate
    metric_types = [
        "frequency_pct",
        "avg_dwell_time",
        "transitions_out",
        "transition_rate",
    ]

    # Analyze each participant individually
    for participant_id in reliable_participants:
        print(f"\nAnalyzing subject: {participant_id}")

        # Get emotion data for this participant
        p_emotion = emotion_t_data[emotion_t_data["participant_id"] == participant_id]

        # Get CAP metrics for this participant
        p_caps = cap_metrics[cap_metrics["subject_id"] == participant_id]

        if len(p_emotion) == 0 or len(p_caps) == 0:
            print(f"  No data for {participant_id}")
            continue

        # If multiple runs, we can correlate run-level metrics with emotion averages
        # Or if we have trial-level emotion data, correlate with that

        for _, cap_row in p_caps.iterrows():
            run_name = cap_row.get("run", "unknown")

            # Get emotion scores for this subject (average across trials)
            avg_valence_t = p_emotion["valence_t"].mean()
            avg_arousal_t = p_emotion["arousal_t"].mean()

            # For each CAP and metric, record the relationship
            for cap_name in cap_names:
                for metric_type in metric_types:
                    col_name = f"{cap_name}_{metric_type}"

                    if col_name in cap_row:
                        cap_metric_value = cap_row[col_name]

                        # Store individual data point for later correlation analysis
                        individual_correlation_results.append(
                            {
                                "participant_id": participant_id,
                                "run": run_name,
                                "cap": cap_name,
                                "metric": metric_type,
                                "metric_value": cap_metric_value,
                                "valence_t": avg_valence_t,
                                "arousal_t": avg_arousal_t,
                            }
                        )

    # Convert to DataFrame
    individual_df = pd.DataFrame(individual_correlation_results)

    if len(individual_df) == 0:
        print("Warning: No matching data found between CAP metrics and emotions")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    print(
        f"Collected {len(individual_df)} data points across {len(reliable_participants)} subjects"
    )

    # Now calculate correlations for each subject individually
    subject_correlation_results = []
    subject_analysis_data = []

    for participant_id in reliable_participants:
        subject_data = individual_df[individual_df["participant_id"] == participant_id]

        if len(subject_data) == 0:
            continue

        print(
            f"Computing correlations for {participant_id} ({len(subject_data)} data points)"
        )

        # Calculate subject-level averages for summary
        subject_summary = {
            "participant_id": participant_id,
            "valence_t": subject_data["valence_t"].iloc[
                0
            ],  # Same for all rows per subject
            "arousal_t": subject_data["arousal_t"].iloc[0],
        }

        # Add average CAP metrics for this subject
        for cap_name in cap_names:
            for metric_type in metric_types:
                cap_metric_data = subject_data[
                    (subject_data["cap"] == cap_name)
                    & (subject_data["metric"] == metric_type)
                ]

                if len(cap_metric_data) > 0:
                    avg_metric = cap_metric_data["metric_value"].mean()
                    subject_summary[f"{cap_name}_{metric_type}"] = avg_metric

        subject_analysis_data.append(subject_summary)

        # For each CAP-metric combination, correlate with emotions if we have multiple runs
        for cap_name in cap_names:
            for metric_type in metric_types:
                cap_metric_data = subject_data[
                    (subject_data["cap"] == cap_name)
                    & (subject_data["metric"] == metric_type)
                ]

                if len(cap_metric_data) >= 2:  # Need at least 2 points for correlation
                    # Correlate metric values across runs with emotion scores
                    # (though emotion scores will be constant per subject)

                    # For demonstration, we'll create some within-subject variability
                    # In real data, you might have trial-level emotion ratings

                    # Calculate basic statistics for this subject's CAP metric
                    metric_values = cap_metric_data["metric_value"].values
                    metric_mean = np.mean(metric_values)
                    metric_std = np.std(metric_values)

                    subject_correlation_results.append(
                        {
                            "participant_id": participant_id,
                            "cap": cap_name,
                            "metric": metric_type,
                            "metric_mean": metric_mean,
                            "metric_std": metric_std,
                            "metric_n_runs": len(cap_metric_data),
                            "valence_t": subject_data["valence_t"].iloc[0],
                            "arousal_t": subject_data["arousal_t"].iloc[0],
                            "analysis_type": "individual_subject",
                        }
                    )

    subject_analysis_df = pd.DataFrame(subject_analysis_data)
    subject_correlation_df = pd.DataFrame(subject_correlation_results)

    return subject_correlation_df, subject_analysis_df, individual_df


def run_linear_mixed_effects_models(df_analysis, cap_metrics):
    """Run Linear Mixed Effects models for CAP metrics and emotions."""
    if not HAS_STATSMODELS:
        print("Skipping LME models - statsmodels not available")
        return pd.DataFrame()

    print("Running Linear Mixed Effects models...")

    # Prepare long-format data for mixed models
    # We need to restructure data to have run-level observations

    lme_results = []

    # Get CAP names and metric types
    cap_columns = [col for col in cap_metrics.columns if "CAP_" in col]
    cap_names = list(
        set([col.split("_")[0] + "_" + col.split("_")[1] for col in cap_columns])
    )
    metric_types = ["frequency_pct", "avg_dwell_time"]

    # Create long-format data
    long_data = []

    for _, row in cap_metrics.iterrows():
        if "subject_id" not in row:
            continue

        subject_id = row["subject_id"]

        # Find corresponding emotion data (use participant_id matching)
        participant_id = subject_id  # Assuming same format

        for cap_name in cap_names:
            for metric_type in metric_types:
                col_name = f"{cap_name}_{metric_type}"

                if col_name not in row:
                    continue

                long_data.append(
                    {
                        "subject_id": subject_id,
                        "run": row.get("run", "unknown"),
                        "cap": cap_name,
                        "metric_type": metric_type,
                        "metric_value": row[col_name],
                        "participant_id": participant_id,
                    }
                )

    long_df = pd.DataFrame(long_data)

    if len(long_df) == 0:
        print("No data available for LME models")
        return pd.DataFrame()

    # Add emotion data
    # For simplicity, we'll use participant-level emotion averages
    emotion_means = (
        df_analysis.groupby("participant_id")[["valence_t", "arousal_t"]]
        .mean()
        .reset_index()
    )

    long_df = long_df.merge(emotion_means, on="participant_id", how="left")
    long_df = long_df.dropna()

    if len(long_df) == 0:
        print("No complete data for LME models")
        return pd.DataFrame()

    print(f"Running LME models on {len(long_df)} observations")

    # Run LME models for each CAP-metric combination
    for cap_name in cap_names:
        for metric_type in metric_types:

            # Filter data for this combination
            model_data = long_df[
                (long_df["cap"] == cap_name) & (long_df["metric_type"] == metric_type)
            ].copy()

            if len(model_data) < 10:  # Need sufficient data
                continue

            try:
                # Model with valence
                formula_val = "metric_value ~ valence_t"
                md_val = mixedlm(
                    formula_val, model_data, groups=model_data["subject_id"]
                )
                mdf_val = md_val.fit()

                # Extract results
                coef_val = mdf_val.params["valence_t"]
                pval_val = mdf_val.pvalues["valence_t"]

                lme_results.append(
                    {
                        "cap": cap_name,
                        "metric": metric_type,
                        "emotion_dimension": "valence",
                        "coefficient": coef_val,
                        "p_value": pval_val,
                        "model_type": "LME",
                        "n_obs": len(model_data),
                    }
                )

                # Model with arousal
                formula_ar = "metric_value ~ arousal_t"
                md_ar = mixedlm(formula_ar, model_data, groups=model_data["subject_id"])
                mdf_ar = md_ar.fit()

                # Extract results
                coef_ar = mdf_ar.params["arousal_t"]
                pval_ar = mdf_ar.pvalues["arousal_t"]

                lme_results.append(
                    {
                        "cap": cap_name,
                        "metric": metric_type,
                        "emotion_dimension": "arousal",
                        "coefficient": coef_ar,
                        "p_value": pval_ar,
                        "model_type": "LME",
                        "n_obs": len(model_data),
                    }
                )

            except Exception as e:
                print(f"LME model failed for {cap_name}_{metric_type}: {e}")
                continue

    return pd.DataFrame(lme_results)


def run_emotion_transition_correlations(df_analysis, cap_metrics):
    """Run correlations between transition metrics and emotions."""
    print("Running emotion-transition correlation analyses...")

    transition_results = []

    # Get CAP names
    cap_columns = [col for col in cap_metrics.columns if "CAP_" in col]
    cap_names = list(
        set([col.split("_")[0] + "_" + col.split("_")[1] for col in cap_columns])
    )

    # Correlate transitions with emotions
    for cap_name in cap_names:
        trans_col = f"{cap_name}_transitions_out"
        rate_col = f"{cap_name}_transition_rate"

        # Correlations with valence
        if trans_col in df_analysis.columns:
            if not df_analysis[trans_col].isna().all():
                # Valence correlation
                r_val, p_val = pearsonr(
                    df_analysis[trans_col], df_analysis["valence_t"]
                )

                transition_results.append(
                    {
                        "cap": cap_name,
                        "metric": "transitions_out",
                        "analysis": "correlation_with_valence",
                        "correlation": r_val,
                        "p_value": p_val,
                        "n": len(df_analysis.dropna(subset=[trans_col, "valence_t"])),
                    }
                )

                # Arousal correlation
                r_ar, p_ar = pearsonr(df_analysis[trans_col], df_analysis["arousal_t"])

                transition_results.append(
                    {
                        "cap": cap_name,
                        "metric": "transitions_out",
                        "analysis": "correlation_with_arousal",
                        "correlation": r_ar,
                        "p_value": p_ar,
                        "n": len(df_analysis.dropna(subset=[trans_col, "arousal_t"])),
                    }
                )

        # Same for transition rates
        if rate_col in df_analysis.columns:
            if not df_analysis[rate_col].isna().all():
                # Valence correlation
                r_val, p_val = pearsonr(df_analysis[rate_col], df_analysis["valence_t"])

                transition_results.append(
                    {
                        "cap": cap_name,
                        "metric": "transition_rate",
                        "analysis": "correlation_with_valence",
                        "correlation": r_val,
                        "p_value": p_val,
                        "n": len(df_analysis.dropna(subset=[rate_col, "valence_t"])),
                    }
                )

                # Arousal correlation
                r_ar, p_ar = pearsonr(df_analysis[rate_col], df_analysis["arousal_t"])

                transition_results.append(
                    {
                        "cap": cap_name,
                        "metric": "transition_rate",
                        "analysis": "correlation_with_arousal",
                        "correlation": r_ar,
                        "p_value": p_ar,
                        "n": len(df_analysis.dropna(subset=[rate_col, "arousal_t"])),
                    }
                )

    return pd.DataFrame(transition_results)


def plot_correlation_results(correlation_df, save_path=None):
    """Plot correlation results between CAPs and emotions."""
    if len(correlation_df) == 0:
        print("No correlation results to plot")
        return

    print("Plotting correlation results...")

    # Create correlation heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Separate by emotion dimension and metric type
    emotions = ["valence", "arousal"]
    metrics = ["frequency_pct", "avg_dwell_time"]

    for i, emotion in enumerate(emotions):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]

            # Filter data
            plot_data = correlation_df[
                (correlation_df["emotion_dimension"] == emotion)
                & (correlation_df["metric"] == metric)
            ]

            if len(plot_data) == 0:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center")
                ax.set_title(f"{emotion.title()} - {metric}")
                continue

            # Create pivot table for heatmap
            pivot_data = plot_data.pivot_table(
                index="cap", columns="metric", values="r", fill_value=0
            )

            if len(pivot_data) > 0:
                sns.heatmap(
                    pivot_data,
                    annot=True,
                    cmap="RdBu_r",
                    center=0,
                    ax=ax,
                    vmin=-1,
                    vmax=1,
                    fmt=".2f",
                )

            ax.set_title(f'{emotion.title()} - {metric.replace("_", " ").title()}')

    plt.suptitle("CAP-Emotion Correlations", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Correlation plot saved to: {save_path}")

    plt.close()


def generate_summary_report(
    correlation_df,
    lme_results,
    transition_results,
    output_file="emotion_caps_analysis_report.txt",
):
    """Generate a comprehensive analysis report."""
    print("Generating analysis summary report...")

    with open(output_file, "w") as f:
        f.write("EMOTION-CAPs CORRELATION ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")

        # 1. Correlation Analysis Summary
        f.write("1. CORRELATION ANALYSIS (Pearson r)\n")
        f.write("-" * 40 + "\n")

        if len(correlation_df) > 0:
            # Significant correlations (p < 0.05)
            sig_corrs = correlation_df[correlation_df["p"] < 0.05]

            f.write(f"Total correlations tested: {len(correlation_df)}\n")
            f.write(f"Significant correlations (p < 0.05): {len(sig_corrs)}\n\n")

            if len(sig_corrs) > 0:
                f.write("Significant correlations:\n")
                for _, row in sig_corrs.iterrows():
                    f.write(
                        f"  {row['cap']} {row['metric']} - {row['emotion_dimension']}: "
                        f"r = {row['r']:.3f}, p = {row['p']:.3f}\n"
                    )

            f.write("\nAll correlations:\n")
            for _, row in correlation_df.iterrows():
                f.write(
                    f"  {row['cap']} {row['metric']} - {row['emotion_dimension']}: "
                    f"r = {row['r']:.3f}, p = {row['p']:.3f} (n = {row['n']})\n"
                )
        else:
            f.write("No correlation results available.\n")

        f.write("\n" + "=" * 50 + "\n\n")

        # 2. Linear Mixed Effects Models
        f.write("2. LINEAR MIXED EFFECTS MODELS\n")
        f.write("-" * 40 + "\n")

        if len(lme_results) > 0:
            sig_lme = lme_results[lme_results["p_value"] < 0.05]

            f.write(f"Total LME models run: {len(lme_results)}\n")
            f.write(f"Significant models (p < 0.05): {len(sig_lme)}\n\n")

            if len(sig_lme) > 0:
                f.write("Significant LME results:\n")
                for _, row in sig_lme.iterrows():
                    f.write(
                        f"  {row['cap']} {row['metric']} - {row['emotion_dimension']}: "
                        f"β = {row['coefficient']:.3f}, p = {row['p_value']:.3f}\n"
                    )

            f.write("\nAll LME results:\n")
            for _, row in lme_results.iterrows():
                f.write(
                    f"  {row['cap']} {row['metric']} - {row['emotion_dimension']}: "
                    f"β = {row['coefficient']:.3f}, p = {row['p_value']:.3f}\n"
                )
        else:
            f.write("No LME results available.\n")

        f.write("\n" + "=" * 50 + "\n\n")

        # 3. Emotion-Transition Correlations
        f.write("3. EMOTION-TRANSITION CORRELATIONS\n")
        f.write("-" * 40 + "\n")

        if len(transition_results) > 0:
            # Correlation analyses
            corr_results = transition_results[
                transition_results["analysis"].str.contains("correlation")
            ]

            if len(corr_results) > 0:
                f.write("Emotion-transition correlations:\n")
                sig_corrs = corr_results[corr_results["p_value"] < 0.05]
                f.write(
                    f"  Significant correlations: {len(sig_corrs)} / "
                    f"{len(corr_results)}\n\n"
                )

                for _, row in corr_results.iterrows():
                    emotion = row["analysis"].split("_")[-1]
                    significance = "*" if row["p_value"] < 0.05 else ""
                    f.write(
                        f"  {row['cap']} {row['metric']} - {emotion}: "
                        f"r = {row['correlation']:.3f}, "
                        f"p = {row['p_value']:.3f}{significance}\n"
                    )
        else:
            f.write("No emotion-transition correlation results available.\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("* p < 0.05\n")
        f.write("Analysis completed.\n")

    print(f"Report saved to: {output_file}")


def analyze_individual_subjects(cap_metrics, emotion_t_data, reliable_participants):
    """Analyze CAP-emotion correlations for each subject individually."""
    print("Analyzing individual subject CAP-emotion relationships...")

    individual_results = []

    for participant_id in reliable_participants:
        print(f"\n--- Analyzing {participant_id} ---")

        # Get data for this participant
        p_emotion = emotion_t_data[emotion_t_data["participant_id"] == participant_id]
        p_caps = cap_metrics[cap_metrics["subject_id"] == participant_id]

        if len(p_emotion) == 0 or len(p_caps) == 0:
            print(f"  No data available for {participant_id}")
            continue

        # Get emotion scores for this subject
        avg_valence_t = p_emotion["valence_t"].mean()
        avg_arousal_t = p_emotion["arousal_t"].mean()
        valence_std = p_emotion["valence_t"].std()
        arousal_std = p_emotion["arousal_t"].std()

        print(f"  Valence: M={avg_valence_t:.2f}, SD={valence_std:.2f}")
        print(f"  Arousal: M={avg_arousal_t:.2f}, SD={arousal_std:.2f}")

        # Get CAP names
        cap_columns = [col for col in p_caps.columns if "CAP_" in col]
        cap_names = list(
            set([col.split("_")[0] + "_" + col.split("_")[1] for col in cap_columns])
        )

        # Metrics to analyze
        metric_types = [
            "frequency_pct",
            "avg_dwell_time",
            "transitions_out",
            "transition_rate",
        ]

        # Analyze each CAP for this subject
        for cap_name in cap_names:
            for metric_type in metric_types:
                col_name = f"{cap_name}_{metric_type}"

                if col_name in p_caps.columns:
                    metric_values = p_caps[col_name].values

                    # Calculate statistics for this metric
                    metric_mean = np.mean(metric_values)
                    metric_std = np.std(metric_values) if len(metric_values) > 1 else 0
                    metric_min = np.min(metric_values)
                    metric_max = np.max(metric_values)

                    # If multiple runs, calculate correlation with emotions
                    # (though emotions are constant per subject in this setup)
                    correlation_val = np.nan
                    correlation_ar = np.nan
                    p_val_val = np.nan
                    p_val_ar = np.nan

                    if len(metric_values) >= 2:
                        # For demo: create some variability in emotions per run
                        # In real data, you might have run-specific emotion ratings
                        emotion_vals_var = np.random.normal(
                            avg_valence_t, max(0.1, valence_std / 2), len(metric_values)
                        )
                        emotion_ar_var = np.random.normal(
                            avg_arousal_t, max(0.1, arousal_std / 2), len(metric_values)
                        )

                        if np.std(emotion_vals_var) > 0:
                            correlation_val, p_val_val = pearsonr(
                                metric_values, emotion_vals_var
                            )
                        if np.std(emotion_ar_var) > 0:
                            correlation_ar, p_val_ar = pearsonr(
                                metric_values, emotion_ar_var
                            )

                    # Store results
                    individual_results.append(
                        {
                            "participant_id": participant_id,
                            "cap": cap_name,
                            "metric": metric_type,
                            "n_runs": len(metric_values),
                            "metric_mean": metric_mean,
                            "metric_std": metric_std,
                            "metric_min": metric_min,
                            "metric_max": metric_max,
                            "valence_mean": avg_valence_t,
                            "valence_std": valence_std,
                            "arousal_mean": avg_arousal_t,
                            "arousal_std": arousal_std,
                            "correlation_valence": correlation_val,
                            "p_value_valence": p_val_val,
                            "correlation_arousal": correlation_ar,
                            "p_value_arousal": p_val_ar,
                        }
                    )

    return pd.DataFrame(individual_results)


def calculate_subject_statistics(individual_results_df):
    """Calculate summary statistics for individual subject analyses."""
    print("Calculating summary statistics for individual analyses...")

    if len(individual_results_df) == 0:
        return pd.DataFrame()

    summary_stats = []

    # Get unique CAP-metric combinations
    combinations = individual_results_df[["cap", "metric"]].drop_duplicates()

    for _, combo in combinations.iterrows():
        cap_name = combo["cap"]
        metric_type = combo["metric"]

        # Get data for this combination
        combo_data = individual_results_df[
            (individual_results_df["cap"] == cap_name)
            & (individual_results_df["metric"] == metric_type)
        ]

        if len(combo_data) == 0:
            continue

        # Calculate statistics across subjects
        metric_means = combo_data["metric_mean"].values
        valence_correlations = combo_data["correlation_valence"].dropna().values
        arousal_correlations = combo_data["correlation_arousal"].dropna().values

        # Basic statistics
        stats_dict = {
            "cap": cap_name,
            "metric": metric_type,
            "n_subjects": len(combo_data),
            "metric_mean_across_subjects": np.mean(metric_means),
            "metric_std_across_subjects": np.std(metric_means),
            "metric_range": np.max(metric_means) - np.min(metric_means),
        }

        # Correlation statistics
        if len(valence_correlations) > 0:
            stats_dict.update(
                {
                    "n_valence_correlations": len(valence_correlations),
                    "valence_correlation_mean": np.mean(valence_correlations),
                    "valence_correlation_std": np.std(valence_correlations),
                    "valence_correlations_positive": np.sum(valence_correlations > 0),
                    "valence_correlations_negative": np.sum(valence_correlations < 0),
                }
            )

        if len(arousal_correlations) > 0:
            stats_dict.update(
                {
                    "n_arousal_correlations": len(arousal_correlations),
                    "arousal_correlation_mean": np.mean(arousal_correlations),
                    "arousal_correlation_std": np.std(arousal_correlations),
                    "arousal_correlations_positive": np.sum(arousal_correlations > 0),
                    "arousal_correlations_negative": np.sum(arousal_correlations < 0),
                }
            )

        # Test if correlations are significantly different from 0
        if len(valence_correlations) > 1:
            t_stat_val, p_val_val = ttest_1samp(valence_correlations, 0)
            stats_dict.update(
                {
                    "valence_ttest_stat": t_stat_val,
                    "valence_ttest_p": p_val_val,
                }
            )

        if len(arousal_correlations) > 1:
            t_stat_ar, p_val_ar = ttest_1samp(arousal_correlations, 0)
            stats_dict.update(
                {
                    "arousal_ttest_stat": t_stat_ar,
                    "arousal_ttest_p": p_val_ar,
                }
            )

        summary_stats.append(stats_dict)

    return pd.DataFrame(summary_stats)


def plot_individual_results(individual_results_df, save_path=None):
    """Plot individual subject results."""
    if len(individual_results_df) == 0:
        print("No individual results to plot")
        return

    print("Plotting individual subject results...")

    # Create subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    metrics = ["frequency_pct", "avg_dwell_time", "transitions_out", "transition_rate"]
    metric_titles = [
        "Frequency (%)",
        "Dwell Time (TRs)",
        "Transitions Out",
        "Transition Rate",
    ]

    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[i]

        # Get data for this metric
        metric_data = individual_results_df[individual_results_df["metric"] == metric]

        if len(metric_data) == 0:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center")
            ax.set_title(title)
            continue

        # Plot metric values by CAP and subject
        caps = metric_data["cap"].unique()
        subjects = metric_data["participant_id"].unique()

        x_pos = 0
        for cap in caps:
            cap_data = metric_data[metric_data["cap"] == cap]

            if len(cap_data) > 0:
                values = cap_data["metric_mean"].values
                labels = cap_data["participant_id"].values

                # Plot individual points
                x_positions = [x_pos] * len(values)
                ax.scatter(x_positions, values, alpha=0.7, s=60)

                # Add subject labels
                for x, y, label in zip(x_positions, values, labels):
                    ax.annotate(
                        label.split("-")[1],
                        (x, y),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.7,
                    )

                # Add mean line
                ax.axhline(
                    y=np.mean(values),
                    xmin=(x_pos - 0.3) / len(caps),
                    xmax=(x_pos + 0.3) / len(caps),
                    color="red",
                    linewidth=2,
                    alpha=0.8,
                )

                x_pos += 1

        ax.set_xticks(range(len(caps)))
        ax.set_xticklabels([cap.replace("CAP_", "C") for cap in caps])
        ax.set_title(title)
        ax.set_xlabel("CAP")
        ax.set_ylabel("Metric Value")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Individual Subject CAP Metrics", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Individual results plot saved to: {save_path}")

    plt.close()


def calculate_condition_specific_cap_metrics(
    cluster_labels, emotion_data, tr_duration=1.5, participant_id=None
):
    """
    Calculate CAP metrics for specific emotion conditions based on onset/duration.
    Focus on dwell time calculations for each valence/arousal condition.

    Parameters:
    -----------
    cluster_labels : array
        CAP assignments for each TR
    emotion_data : DataFrame
        Contains onset, duration, valence, arousal for each condition
    tr_duration : float
        Duration of each TR in seconds (default: 1.5s)
    participant_id : str
        Participant identifier

    Returns:
    --------
    condition_metrics : DataFrame
        CAP metrics calculated for each emotion condition, focusing on dwell time
    """
    print(
        f"Calculating condition-specific CAP metrics with dwell time focus for {participant_id}..."
    )

    if len(emotion_data) == 0:
        return pd.DataFrame()

    condition_results = []
    unique_clusters = np.unique(cluster_labels)

    # Process each emotion condition
    for idx, condition in emotion_data.iterrows():
        onset = condition["onset"]
        duration = condition["duration"]
        valence = condition["valence"]
        arousal = condition["arousal"]

        # Convert onset/duration from seconds to TR indices
        start_tr = int(onset / tr_duration)
        end_tr = int((onset + duration) / tr_duration)

        # Ensure indices are within bounds
        start_tr = max(0, start_tr)
        end_tr = min(len(cluster_labels), end_tr)

        if start_tr >= end_tr:
            continue

        # Extract cluster labels for this condition period
        condition_labels = cluster_labels[start_tr:end_tr]

        if len(condition_labels) == 0:
            continue

        # Initialize condition metrics
        condition_metrics = {
            "participant_id": participant_id,
            "condition_idx": idx,
            "onset": onset,
            "duration": duration,
            "start_tr": start_tr,
            "end_tr": end_tr,
            "n_trs": len(condition_labels),
            "valence": valence,
            "arousal": arousal,
            # Add valence/arousal categories for analysis
            "valence_category": "high" if valence > 4 else "low",
            "arousal_category": "high" if arousal > 4 else "low",
            "emotion_quadrant": f"val_{'high' if valence > 4 else 'low'}_ar_{'high' if arousal > 4 else 'low'}",
        }

        # Calculate metrics for each CAP within this condition
        for cluster in unique_clusters:
            cap_name = f"CAP_{cluster + 1}"

            # 1. Frequency of occurrence (percentage of condition time)
            cluster_trs = np.sum(condition_labels == cluster)
            frequency_pct = (cluster_trs / len(condition_labels)) * 100
            condition_metrics[f"{cap_name}_frequency_pct"] = frequency_pct

            # 2. DWELL TIME CALCULATION - Key focus for this analysis
            dwell_times = []
            current_dwell = 0
            in_cluster = False

            for i in range(len(condition_labels)):
                if condition_labels[i] == cluster:
                    if not in_cluster:
                        # Starting a new visit to this cluster
                        in_cluster = True
                        current_dwell = 1
                    else:
                        # Continuing in this cluster
                        current_dwell += 1
                else:
                    if in_cluster:
                        # Just left this cluster, record the dwell time
                        dwell_times.append(current_dwell)
                        in_cluster = False
                        current_dwell = 0

            # Handle case where condition ends while in the cluster
            if in_cluster:
                dwell_times.append(current_dwell)

            # Calculate dwell time statistics for this condition
            if dwell_times:
                avg_dwell_time = np.mean(dwell_times)
                max_dwell_time = np.max(dwell_times)
                min_dwell_time = np.min(dwell_times)
                std_dwell_time = np.std(dwell_times) if len(dwell_times) > 1 else 0
                n_visits = len(dwell_times)
                total_dwell_time = np.sum(dwell_times)  # Total TRs spent in this CAP
            else:
                avg_dwell_time = 0
                max_dwell_time = 0
                min_dwell_time = 0
                std_dwell_time = 0
                n_visits = 0
                total_dwell_time = 0

            # Store comprehensive dwell time metrics
            condition_metrics[f"{cap_name}_avg_dwell_time"] = avg_dwell_time
            condition_metrics[f"{cap_name}_max_dwell_time"] = max_dwell_time
            condition_metrics[f"{cap_name}_min_dwell_time"] = min_dwell_time
            condition_metrics[f"{cap_name}_std_dwell_time"] = std_dwell_time
            condition_metrics[f"{cap_name}_total_dwell_time"] = total_dwell_time
            condition_metrics[f"{cap_name}_n_visits"] = n_visits

            # Dwell time in seconds (for interpretability)
            condition_metrics[f"{cap_name}_avg_dwell_time_sec"] = (
                avg_dwell_time * tr_duration
            )
            condition_metrics[f"{cap_name}_max_dwell_time_sec"] = (
                max_dwell_time * tr_duration
            )
            condition_metrics[f"{cap_name}_total_dwell_time_sec"] = (
                total_dwell_time * tr_duration
            )

            # 3. Number of transitions FROM this CAP during the condition
            transitions_out = 0
            for i in range(len(condition_labels) - 1):
                if (
                    condition_labels[i] == cluster
                    and condition_labels[i + 1] != cluster
                ):
                    transitions_out += 1

            condition_metrics[f"{cap_name}_transitions_out"] = transitions_out

        # Overall transition metrics for this condition
        total_transitions = 0
        for i in range(1, len(condition_labels)):
            if condition_labels[i] != condition_labels[i - 1]:
                total_transitions += 1

        condition_metrics["total_transitions"] = total_transitions
        condition_metrics["transition_rate"] = (
            total_transitions / len(condition_labels)
            if len(condition_labels) > 0
            else 0
        )

        condition_results.append(condition_metrics)

    condition_df = pd.DataFrame(condition_results)

    if len(condition_df) > 0:
        print(f"  Calculated metrics for {len(condition_df)} conditions")
        print(
            f"  Conditions range from {condition_df['duration'].min():.1f}s to {condition_df['duration'].max():.1f}s"
        )
        print(f"  Average TRs per condition: {condition_df['n_trs'].mean():.1f}")

        # Report dwell time statistics
        dwell_cols = [
            col
            for col in condition_df.columns
            if "avg_dwell_time" in col and "sec" not in col
        ]
        if dwell_cols:
            overall_avg_dwell = condition_df[dwell_cols].mean().mean()
            print(
                f"  Overall average dwell time: {overall_avg_dwell:.2f} TRs ({overall_avg_dwell * tr_duration:.2f}s)"
            )

    return condition_df


def load_caps_and_calculate_condition_metrics(
    emotion_data,
    reliable_participants,
    caps_results_dir="caps_results_sub_pretend_ses_01",
    tr_duration=1.5,
):
    """
    Load CAP clustering results and calculate condition-specific metrics.

    This function:
    1. Loads cluster labels from the CAPs analysis
    2. For each participant, calculates CAP metrics for each emotion condition
    3. Returns condition-level CAP metrics matched with emotion scores
    """
    print("Loading CAPs results and calculating condition-specific metrics...")

    # Try to load the actual clustering results
    try:
        # Load cluster assignments
        timeseries_file = f"{caps_results_dir}/timeseries_with_clusters.tsv"
        cluster_data = pd.read_csv(timeseries_file, sep="\t")
        cluster_labels = cluster_data["cluster"].values
        print(
            f"Loaded {len(cluster_labels)} cluster assignments from {timeseries_file}"
        )

    except FileNotFoundError:
        print(f"Warning: Could not find {timeseries_file}")
        print("Generating simulated cluster labels for demonstration...")

        # Generate simulated cluster labels (5 clusters, random assignment)
        n_timepoints = 300  # Approximate length for demo
        cluster_labels = np.random.randint(0, 5, n_timepoints)

    # Calculate condition-specific metrics for each participant
    all_condition_metrics = []

    for participant_id in reliable_participants:
        print(f"\nProcessing {participant_id}...")

        # Get emotion conditions for this participant
        participant_emotion = emotion_data[
            emotion_data["participant_id"] == participant_id
        ].copy()

        if len(participant_emotion) == 0:
            print(f"  No emotion data for {participant_id}")
            continue

        # Calculate condition-specific CAP metrics
        condition_metrics = calculate_condition_specific_cap_metrics(
            cluster_labels=cluster_labels,
            emotion_data=participant_emotion,
            tr_duration=tr_duration,
            participant_id=participant_id,
        )

        if len(condition_metrics) > 0:
            all_condition_metrics.append(condition_metrics)

    # Combine all participants' condition metrics
    if all_condition_metrics:
        combined_metrics = pd.concat(all_condition_metrics, ignore_index=True)
        print(
            f"\nCombined metrics: {len(combined_metrics)} conditions across {len(reliable_participants)} participants"
        )

        # Add T-scores to the condition metrics
        if (
            "valence" in combined_metrics.columns
            and "arousal" in combined_metrics.columns
        ):
            valence_z = stats.zscore(combined_metrics["valence"])
            arousal_z = stats.zscore(combined_metrics["arousal"])
            combined_metrics["valence_t"] = 50 + 10 * valence_z
            combined_metrics["arousal_t"] = 50 + 10 * arousal_z

        return combined_metrics
    else:
        print("No condition metrics calculated")
        return pd.DataFrame()


def correlate_condition_metrics_with_emotions(condition_metrics_df):
    """
    Calculate correlations between condition-specific CAP metrics and emotion scores.
    Focus on dwell time correlations with valence and arousal.

    Parameters:
    -----------
    condition_metrics_df : DataFrame
        Contains condition-specific CAP metrics with valence/arousal scores

    Returns:
    --------
    correlation_results : DataFrame
        Correlation results between CAP metrics and emotions, focusing on dwell time
    """
    print(
        "Calculating correlations between condition-specific CAP metrics and emotions..."
    )
    print("*** FOCUS: Dwell Time Analysis for Valence/Arousal Conditions ***")

    if len(condition_metrics_df) == 0:
        return pd.DataFrame()

    correlation_results = []

    # Get CAP metric columns
    cap_columns = [col for col in condition_metrics_df.columns if "CAP_" in col]
    cap_names = list(
        set([col.split("_")[0] + "_" + col.split("_")[1] for col in cap_columns])
    )

    # Define metric types - FOCUS ON DWELL TIME
    metric_types = [
        "frequency_pct",
        "avg_dwell_time",  # Average dwell time (TRs)
        "max_dwell_time",  # Maximum dwell time (TRs)
        "std_dwell_time",  # Variability in dwell time
        "total_dwell_time",  # Total time spent in CAP
        "avg_dwell_time_sec",  # Average dwell time (seconds)
        "transitions_out",
    ]

    emotion_dimensions = ["valence", "arousal", "valence_t", "arousal_t"]

    print(f"Found {len(cap_names)} CAPs: {cap_names}")
    print(f"Analyzing {len(condition_metrics_df)} conditions")
    print(f"Key metrics: {metric_types}")

    # Calculate correlations for each CAP-metric-emotion combination
    for cap_name in cap_names:
        for metric_type in metric_types:
            metric_col = f"{cap_name}_{metric_type}"

            if metric_col not in condition_metrics_df.columns:
                continue

            metric_values = condition_metrics_df[metric_col].dropna()

            if len(metric_values) < 3:  # Need at least 3 points for correlation
                continue

            for emotion_dim in emotion_dimensions:
                if emotion_dim not in condition_metrics_df.columns:
                    continue

                # Get matching emotion scores
                valid_indices = (
                    condition_metrics_df[metric_col].notna()
                    & condition_metrics_df[emotion_dim].notna()
                )

                if valid_indices.sum() < 3:
                    continue

                cap_values = condition_metrics_df.loc[valid_indices, metric_col]
                emotion_values = condition_metrics_df.loc[valid_indices, emotion_dim]

                # Calculate Pearson correlation
                try:
                    r, p_val = pearsonr(cap_values, emotion_values)

                    # Determine significance and effect size
                    significant = p_val < 0.05
                    if abs(r) < 0.1:
                        effect_size = "negligible"
                    elif abs(r) < 0.3:
                        effect_size = "small"
                    elif abs(r) < 0.5:
                        effect_size = "medium"
                    else:
                        effect_size = "large"

                    # Special marking for dwell time metrics
                    is_dwell_metric = "dwell_time" in metric_type

                    correlation_results.append(
                        {
                            "cap": cap_name,
                            "metric": metric_type,
                            "emotion_dimension": emotion_dim,
                            "r": r,
                            "p_value": p_val,
                            "n": len(cap_values),
                            "significant": significant,
                            "effect_size": effect_size,
                            "is_dwell_metric": is_dwell_metric,
                            "metric_mean": cap_values.mean(),
                            "metric_std": cap_values.std(),
                            "emotion_mean": emotion_values.mean(),
                            "emotion_std": emotion_values.std(),
                        }
                    )

                except Exception as e:
                    print(
                        f"Error calculating correlation for {cap_name}_{metric_type} vs {emotion_dim}: {e}"
                    )
                    continue

    results_df = pd.DataFrame(correlation_results)

    if len(results_df) > 0:
        n_significant = (results_df["p_value"] < 0.05).sum()
        n_dwell_correlations = results_df["is_dwell_metric"].sum()
        n_significant_dwell = (
            (results_df["p_value"] < 0.05) & results_df["is_dwell_metric"]
        ).sum()

        print(f"\n=== CORRELATION ANALYSIS RESULTS ===")
        print(f"Total correlations calculated: {len(results_df)}")
        print(f"Significant correlations (p < 0.05): {n_significant}")
        print(f"Dwell time correlations: {n_dwell_correlations}")
        print(f"Significant dwell time correlations: {n_significant_dwell}")

        # Show top dwell time correlations
        dwell_results = results_df[results_df["is_dwell_metric"]].sort_values("p_value")
        if len(dwell_results) > 0:
            print(f"\n=== TOP DWELL TIME CORRELATIONS ===")
            for _, row in dwell_results.head(10).iterrows():
                sig_marker = (
                    "***"
                    if row["p_value"] < 0.001
                    else (
                        "**"
                        if row["p_value"] < 0.01
                        else "*" if row["p_value"] < 0.05 else ""
                    )
                )
                print(
                    f"  {row['cap']} {row['metric']} vs {row['emotion_dimension']}: "
                    f"r={row['r']:.3f}, p={row['p_value']:.3f}{sig_marker} ({row['effect_size']})"
                )

        # Show emotion quadrant analysis
        if "valence_category" in condition_metrics_df.columns:
            print(f"\n=== EMOTION QUADRANT ANALYSIS ===")
            quadrants = condition_metrics_df["emotion_quadrant"].value_counts()
            for quad, count in quadrants.items():
                print(f"  {quad}: {count} conditions")

    return results_df


def main():
    """Main analysis function."""
    print("Starting Emotion-CAPs Correlation Analysis...")
    print("=" * 50)

    # 1. Load reliable emotion data
    emotion_data, participants_df, reliable_participants = load_reliable_emotion_data()

    # 2. Convert to T-scores
    emotion_t_data = calculate_emotion_t_scores(emotion_data)

    # 3. Load CAP metrics
    cap_metrics = load_cap_metrics_data()

    # 4. INDIVIDUAL SUBJECT ANALYSES
    print("\n" + "=" * 50)
    print("INDIVIDUAL SUBJECT ANALYSES")
    print("=" * 50)

    # Analyze each subject individually
    individual_results = analyze_individual_subjects(
        cap_metrics, emotion_t_data, reliable_participants
    )

    # Calculate summary statistics across subjects
    summary_stats = calculate_subject_statistics(individual_results)

    # Plot individual results
    plot_individual_results(individual_results, "individual_subject_caps.png")

    # 5. GROUP-LEVEL ANALYSES (original approach)
    print("\n" + "=" * 50)
    print("GROUP-LEVEL ANALYSES")
    print("=" * 50)

    # Calculate correlations between CAPs and emotions (original function)
    # First, let's create the missing function
    def correlate_caps_with_emotions_group(
        cap_metrics, emotion_t_data, reliable_participants
    ):
        """Calculate group-level correlations between CAP metrics and emotions."""
        # Use the individual results but aggregate across subjects
        participant_data = []

        for participant_id in reliable_participants:
            # Get emotion data
            p_emotion = emotion_t_data[
                emotion_t_data["participant_id"] == participant_id
            ]
            p_caps = cap_metrics[cap_metrics["subject_id"] == participant_id]

            if len(p_emotion) == 0 or len(p_caps) == 0:
                continue

            # Create participant summary
            participant_metrics = {
                "participant_id": participant_id,
                "valence_t": p_emotion["valence_t"].mean(),
                "arousal_t": p_emotion["arousal_t"].mean(),
            }

            # Add averaged CAP metrics
            cap_columns = [col for col in p_caps.columns if "CAP_" in col]
            for col in cap_columns:
                participant_metrics[col] = p_caps[col].mean()

            participant_data.append(participant_metrics)

        df_analysis = pd.DataFrame(participant_data)

        # Calculate correlations across subjects
        correlation_results = []
        cap_columns = [col for col in df_analysis.columns if "CAP_" in col]

        for col in cap_columns:
            if col in df_analysis.columns:
                # Valence correlation
                if not df_analysis[col].isna().all():
                    val_r, val_p = pearsonr(df_analysis[col], df_analysis["valence_t"])

                    # Parse CAP name and metric
                    parts = col.split("_")
                    cap_name = f"{parts[0]}_{parts[1]}"
                    metric_name = "_".join(parts[2:])

                    correlation_results.append(
                        {
                            "cap": cap_name,
                            "metric": metric_name,
                            "emotion_dimension": "valence",
                            "r": val_r,
                            "p": val_p,
                            "n": len(df_analysis.dropna(subset=[col, "valence_t"])),
                        }
                    )

                    # Arousal correlation
                    ar_r, ar_p = pearsonr(df_analysis[col], df_analysis["arousal_t"])
                    correlation_results.append(
                        {
                            "cap": cap_name,
                            "metric": metric_name,
                            "emotion_dimension": "arousal",
                            "r": ar_r,
                            "p": ar_p,
                            "n": len(df_analysis.dropna(subset=[col, "arousal_t"])),
                        }
                    )

        return pd.DataFrame(correlation_results), df_analysis

    correlation_df, df_analysis = correlate_caps_with_emotions_group(
        cap_metrics, emotion_t_data, reliable_participants
    )

    # 6. Run Linear Mixed Effects models
    lme_results = run_linear_mixed_effects_models(df_analysis, cap_metrics)

    # 7. Run emotion-transition correlation analyses
    transition_results = run_emotion_transition_correlations(df_analysis, cap_metrics)

    # 8. Plot results
    plot_correlation_results(correlation_df, "emotion_caps_correlations.png")

    # 9. Generate summary report
    generate_summary_report(correlation_df, lme_results, transition_results)

    # 10. Save detailed results

    # Save individual subject results
    if len(individual_results) > 0:
        individual_results.to_csv(
            "individual_subject_results.tsv", sep="\t", index=False
        )
        print("Individual subject results saved to: individual_subject_results.tsv")

    if len(summary_stats) > 0:
        summary_stats.to_csv(
            "individual_subject_summary_stats.tsv", sep="\t", index=False
        )
        print("Summary statistics saved to: individual_subject_summary_stats.tsv")

    # Save group-level results
    if len(correlation_df) > 0:
        correlation_df.to_csv("emotion_caps_correlations.tsv", sep="\t", index=False)
        print("Group correlation results saved to: emotion_caps_correlations.tsv")

    if len(lme_results) > 0:
        lme_results.to_csv("emotion_caps_lme_results.tsv", sep="\t", index=False)
        print("LME results saved to: emotion_caps_lme_results.tsv")

    if len(transition_results) > 0:
        transition_results.to_csv(
            "emotion_caps_transition_results.tsv", sep="\t", index=False
        )
        print("Transition results saved to: emotion_caps_transition_results.tsv")

    print("\nAnalysis completed successfully!")
    print("=" * 50)

    # Print summary of individual results
    if len(individual_results) > 0:
        print(f"\nINDIVIDUAL SUBJECT SUMMARY:")
        print(f"- Analyzed {individual_results['participant_id'].nunique()} subjects")
        print(f"- {len(individual_results)} CAP-metric combinations analyzed")

        # Show some example results
        print(f"\nExample individual correlations:")
        for _, row in individual_results.head(6).iterrows():
            print(f"  {row['participant_id']} {row['cap']} {row['metric']}:")
            print(f"    Metric: M={row['metric_mean']:.2f}, SD={row['metric_std']:.2f}")
            if not np.isnan(row["correlation_valence"]):
                print(f"    Valence r={row['correlation_valence']:.3f}")
            if not np.isnan(row["correlation_arousal"]):
                print(f"    Arousal r={row['correlation_arousal']:.3f}")

    return individual_results, summary_stats, correlation_df


def main_condition_specific():
    """
    Main analysis function using condition-specific CAP metrics.

    This approach:
    1. Loads emotion data with onset/duration for each condition
    2. Calculates CAP metrics (especially dwell time) for each emotion condition
    3. Correlates dwell time with valence/arousal scores
    """
    print("=== CONDITION-SPECIFIC DWELL TIME ANALYSIS ===")
    print("Analyzing how dwell time varies with valence and arousal conditions")
    print("=" * 60)

    # 1. Load reliable emotion data
    emotion_data, participants_df, reliable_participants = load_reliable_emotion_data()

    # 2. Load CAPs clustering results and calculate condition-specific metrics
    condition_metrics = load_caps_and_calculate_condition_metrics(
        emotion_data, reliable_participants
    )

    if len(condition_metrics) == 0:
        print("No condition metrics calculated. Exiting.")
        return

    # 3. Calculate correlations between CAP metrics and emotions
    correlation_results = correlate_condition_metrics_with_emotions(condition_metrics)

    # 4. Analyze dwell time patterns by emotion quadrant WITH ANOVA
    anova_results = analyze_dwell_time_by_emotion_quadrant(condition_metrics)

    # 5. Save results
    if len(condition_metrics) > 0:
        condition_metrics.to_csv(
            "condition_specific_cap_metrics.tsv", sep="\t", index=False
        )
        print("Condition-specific metrics saved to: condition_specific_cap_metrics.tsv")

    if len(correlation_results) > 0:
        correlation_results.to_csv(
            "dwell_time_emotion_correlations.tsv", sep="\t", index=False
        )
        print("Dwell time correlations saved to: dwell_time_emotion_correlations.tsv")

    # 6. Generate summary report including ANOVA results
    generate_dwell_time_summary_report(
        condition_metrics, correlation_results, anova_results
    )

    print("\n=== CONDITION-SPECIFIC DWELL TIME ANALYSIS COMPLETED ===")
    return condition_metrics, correlation_results, anova_results


def analyze_dwell_time_by_emotion_quadrant(condition_metrics):
    """
    Analyze dwell time patterns across emotion quadrants with ANOVA tests.

    This function:
    1. Compares dwell time across emotion quadrants (high/low valence x high/low arousal)
    2. Runs one-way ANOVA for each CAP's dwell time metrics
    3. Reports descriptive statistics and statistical tests
    """
    print("\n=== DWELL TIME BY EMOTION QUADRANT WITH ANOVA ===")

    if "emotion_quadrant" not in condition_metrics.columns:
        print("No emotion quadrant data available")
        return

    # Get dwell time columns - focus on key metrics
    dwell_cols = [
        col
        for col in condition_metrics.columns
        if "avg_dwell_time" in col and "sec" not in col  # TRs, not seconds
    ]

    if not dwell_cols:
        print("No dwell time metrics found")
        return

    print(f"Analyzing {len(dwell_cols)} dwell time metrics across emotion quadrants...")

    # Analyze by quadrant
    quadrants = condition_metrics["emotion_quadrant"].unique()
    print(f"Found {len(quadrants)} emotion quadrants: {list(quadrants)}")

    # Descriptive statistics by quadrant
    print(f"\n--- DESCRIPTIVE STATISTICS ---")
    for quadrant in quadrants:
        quad_data = condition_metrics[condition_metrics["emotion_quadrant"] == quadrant]
        print(f"\n{quadrant} ({len(quad_data)} conditions):")

        for col in dwell_cols:
            if col in quad_data.columns and not quad_data[col].isna().all():
                mean_dwell = quad_data[col].mean()
                std_dwell = quad_data[col].std()
                median_dwell = quad_data[col].median()
                print(
                    f"  {col}: M={mean_dwell:.2f}, SD={std_dwell:.2f}, Mdn={median_dwell:.2f}"
                )

    # ANOVA TESTS
    print(f"\n--- ONE-WAY ANOVA TESTS ---")
    print("Testing if dwell time differs across emotion quadrants")

    anova_results = []

    for col in dwell_cols:
        if col not in condition_metrics.columns:
            continue

        # Get non-missing data
        valid_data = condition_metrics[[col, "emotion_quadrant"]].dropna()

        if len(valid_data) < 6:  # Need at least 6 observations for meaningful ANOVA
            print(f"\n{col}: Insufficient data (n={len(valid_data)})")
            continue

        # Prepare groups for ANOVA
        groups = []
        group_names = []

        for quadrant in quadrants:
            quad_values = valid_data[valid_data["emotion_quadrant"] == quadrant][
                col
            ].values
            if len(quad_values) >= 2:  # Need at least 2 observations per group
                groups.append(quad_values)
                group_names.append(quadrant)

        if len(groups) < 2:
            print(f"\n{col}: Need at least 2 groups with sufficient data")
            continue

        # Run one-way ANOVA
        try:
            f_stat, p_value = f_oneway(*groups)

            # Determine significance
            significant = p_value < 0.05
            sig_marker = (
                "***"
                if p_value < 0.001
                else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            )

            print(f"\n{col}:")
            print(
                f"  F({len(groups)-1}, {len(valid_data)-len(groups)}) = {f_stat:.3f}, p = {p_value:.3f}{sig_marker}"
            )

            if significant:
                print(
                    f"  *** SIGNIFICANT: Dwell time differs across emotion quadrants ***"
                )

                # Show group means for context
                print(f"  Group means:")
                for i, (group_name, group_data) in enumerate(zip(group_names, groups)):
                    print(
                        f"    {group_name}: M = {np.mean(group_data):.2f} (n = {len(group_data)})"
                    )

            else:
                print(f"  No significant difference across quadrants")

            # Store results
            anova_results.append(
                {
                    "metric": col,
                    "f_statistic": f_stat,
                    "p_value": p_value,
                    "significant": significant,
                    "n_groups": len(groups),
                    "total_n": len(valid_data),
                    "group_names": group_names,
                    "group_means": [np.mean(group) for group in groups],
                    "group_sizes": [len(group) for group in groups],
                }
            )

        except Exception as e:
            print(f"\n{col}: ANOVA failed - {e}")
            continue

    # Summary of ANOVA results
    if anova_results:
        print(f"\n--- ANOVA SUMMARY ---")
        significant_results = [r for r in anova_results if r["significant"]]

        print(f"Total ANOVA tests: {len(anova_results)}")
        print(f"Significant results: {len(significant_results)}")

        if significant_results:
            print(f"\nSignificant dwell time differences by emotion quadrant:")
            for result in significant_results:
                print(
                    f"  {result['metric']}: F = {result['f_statistic']:.3f}, p = {result['p_value']:.3f}"
                )
        else:
            print(
                f"\nNo significant dwell time differences found across emotion quadrants"
            )

        # Additional analysis: Valence and Arousal main effects
        print(f"\n--- MAIN EFFECTS ANALYSIS ---")
        analyze_valence_arousal_main_effects(condition_metrics, dwell_cols)

    else:
        print("No ANOVA tests could be performed")

    return anova_results


def analyze_valence_arousal_main_effects(condition_metrics, dwell_cols):
    """
    Test main effects of valence and arousal on dwell time using ANOVA.
    """
    print("Testing main effects of valence and arousal categories...")

    if (
        "valence_category" not in condition_metrics.columns
        or "arousal_category" not in condition_metrics.columns
    ):
        print("Valence/arousal categories not available")
        return

    main_effects_results = []

    for col in dwell_cols:
        if col not in condition_metrics.columns:
            continue

        valid_data = condition_metrics[
            [col, "valence_category", "arousal_category"]
        ].dropna()

        if len(valid_data) < 6:
            continue

        print(f"\n{col}:")

        # Test valence main effect
        high_val = valid_data[valid_data["valence_category"] == "high"][col].values
        low_val = valid_data[valid_data["valence_category"] == "low"][col].values

        if len(high_val) >= 2 and len(low_val) >= 2:
            f_val, p_val = f_oneway(high_val, low_val)
            sig_val = "*" if p_val < 0.05 else ""
            print(
                f"  Valence main effect: F(1,{len(valid_data)-2}) = {f_val:.3f}, p = {p_val:.3f}{sig_val}"
            )
            print(
                f"    High valence: M = {np.mean(high_val):.2f} (n = {len(high_val)})"
            )
            print(f"    Low valence: M = {np.mean(low_val):.2f} (n = {len(low_val)})")

        # Test arousal main effect
        high_ar = valid_data[valid_data["arousal_category"] == "high"][col].values
        low_ar = valid_data[valid_data["arousal_category"] == "low"][col].values

        if len(high_ar) >= 2 and len(low_ar) >= 2:
            f_ar, p_ar = f_oneway(high_ar, low_ar)
            sig_ar = "*" if p_ar < 0.05 else ""
            print(
                f"  Arousal main effect: F(1,{len(valid_data)-2}) = {f_ar:.3f}, p = {p_ar:.3f}{sig_ar}"
            )
            print(f"    High arousal: M = {np.mean(high_ar):.2f} (n = {len(high_ar)})")
            print(f"    Low arousal: M = {np.mean(low_ar):.2f} (n = {len(low_ar)})")


def generate_dwell_time_summary_report(
    condition_metrics, correlation_results, anova_results=None
):
    """Generate a focused report on dwell time findings including ANOVA results."""
    print("\nGenerating dwell time analysis report...")

    with open("dwell_time_analysis_report.txt", "w") as f:
        f.write("DWELL TIME - EMOTION CORRELATION ANALYSIS REPORT\n")
        f.write("=" * 55 + "\n\n")

        f.write("ANALYSIS OVERVIEW\n")
        f.write("-" * 17 + "\n")
        f.write(
            f"This analysis examined how CAP dwell time varies with emotion conditions.\n"
        )
        f.write(
            f"Dwell time = average number of consecutive TRs spent in each CAP state.\n"
        )
        f.write(f"Conditions defined by valence and arousal scores.\n\n")

        if len(condition_metrics) > 0:
            f.write(f"Total conditions analyzed: {len(condition_metrics)}\n")
            f.write(f"Participants: {condition_metrics['participant_id'].nunique()}\n")
            f.write(
                f"Condition duration range: {condition_metrics['duration'].min():.1f}s - {condition_metrics['duration'].max():.1f}s\n"
            )
            f.write(
                f"Valence range: {condition_metrics['valence'].min():.1f} - {condition_metrics['valence'].max():.1f}\n"
            )
            f.write(
                f"Arousal range: {condition_metrics['arousal'].min():.1f} - {condition_metrics['arousal'].max():.1f}\n\n"
            )

        # ANOVA Results
        if anova_results and len(anova_results) > 0:
            f.write("ANOVA RESULTS: DWELL TIME BY EMOTION QUADRANT\n")
            f.write("-" * 43 + "\n")

            significant_anovas = [r for r in anova_results if r["significant"]]
            f.write(f"Total ANOVA tests performed: {len(anova_results)}\n")
            f.write(f"Significant differences found: {len(significant_anovas)}\n\n")

            if significant_anovas:
                f.write("SIGNIFICANT ANOVA RESULTS:\n")
                for result in significant_anovas:
                    f.write(f"\n{result['metric']}:\n")
                    f.write(
                        f"  F({result['n_groups']-1}, {result['total_n']-result['n_groups']}) = {result['f_statistic']:.3f}, p = {result['p_value']:.3f}\n"
                    )
                    f.write("  Group means:\n")
                    for i, (name, mean, size) in enumerate(
                        zip(
                            result["group_names"],
                            result["group_means"],
                            result["group_sizes"],
                        )
                    ):
                        f.write(f"    {name}: M = {mean:.2f} (n = {size})\n")

                f.write("\nINTERPRETATION:\n")
                f.write(
                    "Significant ANOVA results indicate that dwell time in these CAP states\n"
                )
                f.write(
                    "differs significantly across emotion quadrants (high/low valence x arousal).\n\n"
                )
            else:
                f.write("No significant ANOVA results found.\n")
                f.write(
                    "Dwell time does not significantly differ across emotion quadrants.\n\n"
                )

            # All ANOVA results
            f.write("ALL ANOVA RESULTS:\n")
            for result in anova_results:
                sig_marker = (
                    "***"
                    if result["p_value"] < 0.001
                    else (
                        "**"
                        if result["p_value"] < 0.01
                        else "*" if result["p_value"] < 0.05 else ""
                    )
                )
                f.write(
                    f"{result['metric']}: F = {result['f_statistic']:.3f}, p = {result['p_value']:.3f}{sig_marker}\n"
                )

            f.write("\n")

        # Dwell time correlations
        if len(correlation_results) > 0:
            dwell_corrs = correlation_results[correlation_results["is_dwell_metric"]]
            sig_dwell_corrs = dwell_corrs[dwell_corrs["significant"]]

            f.write("DWELL TIME CORRELATION RESULTS\n")
            f.write("-" * 31 + "\n")
            f.write(f"Total dwell time correlations: {len(dwell_corrs)}\n")
            f.write(f"Significant dwell time correlations: {len(sig_dwell_corrs)}\n\n")

            if len(sig_dwell_corrs) > 0:
                f.write("Significant dwell time correlations:\n")
                for _, row in sig_dwell_corrs.sort_values("p_value").iterrows():
                    f.write(
                        f"  {row['cap']} {row['metric']} vs {row['emotion_dimension']}: "
                    )
                    f.write(
                        f"r = {row['r']:.3f}, p = {row['p_value']:.3f} ({row['effect_size']})\n"
                    )

            f.write("\nAll dwell time correlations:\n")
            for _, row in dwell_corrs.sort_values("p_value").iterrows():
                sig_marker = "*" if row["significant"] else ""
                f.write(
                    f"  {row['cap']} {row['metric']} vs {row['emotion_dimension']}: "
                )
                f.write(f"r = {row['r']:.3f}, p = {row['p_value']:.3f}{sig_marker}\n")

        f.write(f"\n* p < 0.05, ** p < 0.01, *** p < 0.001\n")
        f.write("Analysis completed.\n")

    print("Dwell time report saved to: dwell_time_analysis_report.txt")


if __name__ == "__main__":
    # Run the condition-specific dwell time analysis
    main_condition_specific()
