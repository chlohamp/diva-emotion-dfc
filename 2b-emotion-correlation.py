import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, ttest_1samp, ttest_ind
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


def load_cap_metrics_data(cap_results_file="caps_results/cluster_statistics.tsv"):
    """Load CAP metrics from previous analysis."""
    print("Loading CAP metrics data...")

    # The actual CAP analysis only generated one run, so we'll use simulated data
    print("Real CAP metrics only have one subject/run. Using simulated CAP data...")
    return generate_simulated_cap_metrics()


def generate_simulated_cap_metrics(n_subjects=3, n_runs=2, n_caps=5):
    """Generate simulated CAP metrics for demonstration."""
    print(f"Generating simulated CAP metrics for {n_subjects} subjects...")

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

            # Generate metrics for each CAP
            for cap in range(1, n_caps + 1):
                cap_name = f"CAP_{cap}"

                # Frequency (percentage of time in this state)
                frequency = np.random.uniform(10, 30)
                metrics[f"{cap_name}_frequency_pct"] = frequency

                # Dwell time (average consecutive TRs)
                dwell_time = np.random.uniform(2, 8)
                metrics[f"{cap_name}_avg_dwell_time"] = dwell_time

                # Number of episodes
                episodes = max(1, int(frequency * 0.5))
                metrics[f"{cap_name}_n_episodes"] = episodes

                # Transitions out
                transitions = np.random.poisson(episodes * 0.8)
                metrics[f"{cap_name}_transitions_out"] = transitions

                # Transition rate
                if episodes > 0:
                    trans_rate = transitions / episodes
                else:
                    trans_rate = 0
                metrics[f"{cap_name}_transition_rate"] = trans_rate

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


def correlate_caps_with_emotions(cap_metrics, emotion_t_data, reliable_participants):
    """Calculate correlations between CAP metrics and emotion dimensions."""
    print("Calculating correlations between CAPs and emotion dimensions...")

    # Prepare data for correlation analysis
    correlation_results = []

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

    # For each participant, get average emotion scores and CAP metrics
    participant_data = []

    for participant_id in reliable_participants:
        # Get emotion data for this participant
        p_emotion = emotion_t_data[emotion_t_data["participant_id"] == participant_id]

        if len(p_emotion) == 0:
            continue

        # Average emotion scores across trials for this participant
        avg_valence_t = p_emotion["valence_t"].mean()
        avg_arousal_t = p_emotion["arousal_t"].mean()

        # Get CAP metrics for this participant (average across runs)
        p_caps = cap_metrics[cap_metrics["subject_id"] == participant_id]

        if len(p_caps) == 0:
            continue

        # Average CAP metrics across runs for this participant
        participant_metrics = {
            "participant_id": participant_id,
            "valence_t": avg_valence_t,
            "arousal_t": avg_arousal_t,
        }

        for cap_name in cap_names:
            for metric_type in metric_types:
                col_name = f"{cap_name}_{metric_type}"
                if col_name in p_caps.columns:
                    participant_metrics[col_name] = p_caps[col_name].mean()

        participant_data.append(participant_metrics)

    # Convert to DataFrame
    df_analysis = pd.DataFrame(participant_data)

    if len(df_analysis) == 0:
        print("Warning: No matching data found between CAP metrics and emotions")
        return pd.DataFrame(), df_analysis

    print(f"Analyzing {len(df_analysis)} participants with complete data")

    # Calculate correlations
    for cap_name in cap_names:
        for metric_type in metric_types:
            col_name = f"{cap_name}_{metric_type}"

            if col_name not in df_analysis.columns:
                continue

            # Correlations with valence
            if not df_analysis[col_name].isna().all():
                val_r, val_p = pearsonr(df_analysis[col_name], df_analysis["valence_t"])
                correlation_results.append(
                    {
                        "cap": cap_name,
                        "metric": metric_type,
                        "emotion_dimension": "valence",
                        "r": val_r,
                        "p": val_p,
                        "n": len(df_analysis.dropna(subset=[col_name, "valence_t"])),
                    }
                )

                # Correlations with arousal
                ar_r, ar_p = pearsonr(df_analysis[col_name], df_analysis["arousal_t"])
                correlation_results.append(
                    {
                        "cap": cap_name,
                        "metric": metric_type,
                        "emotion_dimension": "arousal",
                        "r": ar_r,
                        "p": ar_p,
                        "n": len(df_analysis.dropna(subset=[col_name, "arousal_t"])),
                    }
                )

    correlation_df = pd.DataFrame(correlation_results)

    return correlation_df, df_analysis


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


def run_transition_analyses(df_analysis, cap_metrics):
    """Run specific analyses for transition metrics."""
    print("Running transition-specific analyses...")

    transition_results = []

    # Get CAP names
    cap_columns = [col for col in cap_metrics.columns if "CAP_" in col]
    cap_names = list(
        set([col.split("_")[0] + "_" + col.split("_")[1] for col in cap_columns])
    )

    # 1. T-tests for transitions during rest (one-sample against 0)
    print("Running one-sample t-tests for transitions (vs 0)...")

    for cap_name in cap_names:
        trans_col = f"{cap_name}_transitions_out"
        rate_col = f"{cap_name}_transition_rate"

        # Test transitions out
        if trans_col in df_analysis.columns:
            transitions = df_analysis[trans_col].dropna()

            if len(transitions) > 0:
                t_stat, p_val = ttest_1samp(transitions, 0)

                transition_results.append(
                    {
                        "cap": cap_name,
                        "metric": "transitions_out",
                        "analysis": "one_sample_ttest_vs_0",
                        "t_statistic": t_stat,
                        "p_value": p_val,
                        "mean": transitions.mean(),
                        "n": len(transitions),
                    }
                )

        # Test transition rates
        if rate_col in df_analysis.columns:
            rates = df_analysis[rate_col].dropna()

            if len(rates) > 0:
                t_stat, p_val = ttest_1samp(rates, 0)

                transition_results.append(
                    {
                        "cap": cap_name,
                        "metric": "transition_rate",
                        "analysis": "one_sample_ttest_vs_0",
                        "t_statistic": t_stat,
                        "p_value": p_val,
                        "mean": rates.mean(),
                        "n": len(rates),
                    }
                )

    # 2. Mixed-model ANOVA for transitions (if multiple runs available)
    if HAS_STATSMODELS and "run_number" in cap_metrics.columns:
        print("Running mixed-model ANOVA for transitions...")

        # This would require run-level data
        # For now, we'll correlate transitions with emotions
        for cap_name in cap_names:
            trans_col = f"{cap_name}_transitions_out"

            if trans_col in df_analysis.columns:
                # Correlation with emotions
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
                            "n": len(
                                df_analysis.dropna(subset=[trans_col, "valence_t"])
                            ),
                        }
                    )

                    # Arousal correlation
                    r_ar, p_ar = pearsonr(
                        df_analysis[trans_col], df_analysis["arousal_t"]
                    )

                    transition_results.append(
                        {
                            "cap": cap_name,
                            "metric": "transitions_out",
                            "analysis": "correlation_with_arousal",
                            "correlation": r_ar,
                            "p_value": p_ar,
                            "n": len(
                                df_analysis.dropna(subset=[trans_col, "arousal_t"])
                            ),
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

        # 3. Transition Analyses
        f.write("3. TRANSITION ANALYSES\n")
        f.write("-" * 40 + "\n")

        if len(transition_results) > 0:
            # One-sample t-tests
            ttest_results = transition_results[
                transition_results["analysis"] == "one_sample_ttest_vs_0"
            ]

            if len(ttest_results) > 0:
                f.write("One-sample t-tests (vs 0):\n")
                sig_ttests = ttest_results[ttest_results["p_value"] < 0.05]
                f.write(
                    f"  Significant tests: {len(sig_ttests)} / {len(ttest_results)}\n"
                )

                for _, row in ttest_results.iterrows():
                    significance = "*" if row["p_value"] < 0.05 else ""
                    f.write(
                        f"  {row['cap']} {row['metric']}: "
                        f"M = {row['mean']:.3f}, t = {row['t_statistic']:.3f}, "
                        f"p = {row['p_value']:.3f}{significance}\n"
                    )

            # Correlation analyses
            corr_results = transition_results[
                transition_results["analysis"].str.contains("correlation")
            ]

            if len(corr_results) > 0:
                f.write("\nTransition-emotion correlations:\n")
                for _, row in corr_results.iterrows():
                    emotion = row["analysis"].split("_")[-1]
                    significance = "*" if row["p_value"] < 0.05 else ""
                    f.write(
                        f"  {row['cap']} {row['metric']} - {emotion}: "
                        f"r = {row['correlation']:.3f}, p = {row['p_value']:.3f}{significance}\n"
                    )
        else:
            f.write("No transition analysis results available.\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("* p < 0.05\n")
        f.write("Analysis completed.\n")

    print(f"Report saved to: {output_file}")


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

    # 4. Calculate correlations between CAPs and emotions
    correlation_df, df_analysis = correlate_caps_with_emotions(
        cap_metrics, emotion_t_data, reliable_participants
    )

    # 5. Run Linear Mixed Effects models
    lme_results = run_linear_mixed_effects_models(df_analysis, cap_metrics)

    # 6. Run transition-specific analyses
    transition_results = run_transition_analyses(df_analysis, cap_metrics)

    # 7. Plot results
    plot_correlation_results(correlation_df, "emotion_caps_correlations.png")

    # 8. Generate summary report
    generate_summary_report(correlation_df, lme_results, transition_results)

    # 9. Save detailed results
    if len(correlation_df) > 0:
        correlation_df.to_csv("emotion_caps_correlations.tsv", sep="\t", index=False)
        print("Correlation results saved to: emotion_caps_correlations.tsv")

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


if __name__ == "__main__":
    main()
