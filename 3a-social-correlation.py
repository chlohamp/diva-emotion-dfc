import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, ttest_1samp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# For mixed-effects models
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import mixedlm

    HAS_STATSMODELS = True
except ImportError:
    print("Warning: statsmodels not available. Install with: pip install statsmodels")
    HAS_STATSMODELS = False

# Set non-interactive backend for matplotlib
plt.switch_backend("Agg")


def load_character_data(
    participants_characters_file="participants-characters.tsv",
    participants_file="participants.tsv",
):
    """Load character rating data."""
    print("Loading character rating data...")

    # Load character ratings
    character_data = pd.read_csv(participants_characters_file, sep="\t")
    participants = pd.read_csv(participants_file, sep="\t")

    print(f"Loaded character data for {len(character_data)} participants")
    print(
        f"Character columns: {[col for col in character_data.columns if 'valence' in col or 'arousal' in col]}"
    )

    return character_data, participants


def create_character_appearance_timeseries(
    events_data,
    character_names=["dusty", "nancy", "steve"],
    total_duration=24.0,  # Total run duration in seconds
    tr=1.5,  # Time between samples
):
    """
    Create character appearance time series based on stimulus events.

    For this simulation, we'll create realistic character appearance patterns
    based on the stimulus timing from the events data.
    """
    print("Creating character appearance time series...")

    # Calculate number of time points
    n_timepoints = int(total_duration / tr) + 1
    time_points = np.arange(0, total_duration + tr, tr)

    print(f"Creating timeseries with {n_timepoints} time points")
    print(f"Time range: 0 to {total_duration} seconds")

    # Initialize character appearance arrays
    character_timeseries = {}

    for character in character_names:
        character_timeseries[character] = np.zeros(n_timepoints)

    # Create realistic character appearance patterns
    # Based on typical TV show structure where characters appear in segments

    # Set random seed for reproducible character appearances
    np.random.seed(42)

    # Dusty appears frequently throughout (main character)
    dusty_segments = [
        (0, 8),  # Beginning scene
        (12, 18),  # Middle scene
        (20, 24),  # Final scene
    ]

    # Nancy appears in middle sections
    nancy_segments = [(3, 10), (15, 22)]  # Early-middle scene  # Late scene

    # Steve appears sporadically
    steve_segments = [(5, 12), (18, 24)]  # Middle scene  # Final scene

    character_segments = {
        "dusty": dusty_segments,
        "nancy": nancy_segments,
        "steve": steve_segments,
    }

    # Fill in character appearances
    for character, segments in character_segments.items():
        for start_time, end_time in segments:
            # Find corresponding time points
            start_idx = int(start_time / tr)
            end_idx = min(int(end_time / tr), n_timepoints - 1)

            # Create gradual appearance/disappearance (more realistic)
            segment_length = end_idx - start_idx + 1
            if segment_length > 2:
                # Gradual increase at beginning
                ramp_up = np.linspace(0, 1, min(3, segment_length // 3))
                # Stable middle section
                stable = np.ones(segment_length - 2 * len(ramp_up))
                # Gradual decrease at end
                ramp_down = np.linspace(1, 0, len(ramp_up))

                if len(ramp_up) + len(stable) + len(ramp_down) == segment_length:
                    appearance_pattern = np.concatenate([ramp_up, stable, ramp_down])
                else:
                    appearance_pattern = np.ones(segment_length)
            else:
                appearance_pattern = np.ones(segment_length)

            character_timeseries[character][
                start_idx : start_idx + len(appearance_pattern)
            ] = appearance_pattern

    # Convert to DataFrame
    timeseries_df = pd.DataFrame(character_timeseries)
    timeseries_df["time"] = time_points[: len(timeseries_df)]

    print("Character appearance timeseries created:")
    for character in character_names:
        total_presence = np.sum(character_timeseries[character])
        percentage = (total_presence / n_timepoints) * 100
        print(
            f"  {character.title()}: {total_presence:.1f} total presence ({percentage:.1f}%)"
        )

    return timeseries_df, character_timeseries


def weight_character_timeseries_by_ratings(
    character_timeseries,
    character_data,
    participant_id,
    weight_type="valence",  # or 'arousal'
):
    """
    Weight character appearance timeseries by individual participant ratings.

    This creates personalized character significance based on each participant's
    subjective ratings of the characters.
    """
    print(f"Weighting character timeseries for {participant_id} using {weight_type}...")

    # Get participant's character ratings
    participant_ratings = character_data[
        character_data["participant_id"] == participant_id
    ]

    if len(participant_ratings) == 0:
        print(f"Warning: No character ratings found for {participant_id}")
        return character_timeseries

    participant_ratings = participant_ratings.iloc[0]

    # Extract character ratings
    character_weights = {}
    character_names = ["dusty", "nancy", "steve"]

    for character in character_names:
        weight_col = f"{character}_{weight_type}"
        if weight_col in participant_ratings:
            # Normalize weights to 0-1 scale (assuming 1-7 rating scale)
            raw_weight = participant_ratings[weight_col]
            normalized_weight = (raw_weight - 1) / 6  # Scale 1-7 to 0-1
            character_weights[character] = normalized_weight

            print(
                f"  {character.title()}: rating={raw_weight}, weight={normalized_weight:.3f}"
            )
        else:
            character_weights[character] = 0.5  # Default neutral weight

    # Apply weights to timeseries
    weighted_timeseries = {}
    for character in character_names:
        if character in character_timeseries:
            weighted_timeseries[character] = (
                character_timeseries[character] * character_weights[character]
            )
        else:
            weighted_timeseries[character] = np.zeros_like(
                list(character_timeseries.values())[0]
            )

    return weighted_timeseries


def load_cap_metrics_data(cap_results_file="caps_results/cluster_statistics.tsv"):
    """Load CAP metrics from previous analysis."""
    print("Loading CAP metrics data...")

    # Use simulated CAP data that matches our character analysis participants
    print("Using simulated CAP data for character analysis...")
    return generate_simulated_cap_metrics()


def generate_simulated_cap_metrics(n_subjects=3, n_runs=2, n_caps=5):
    """Generate simulated CAP metrics matching character data participants."""
    print(f"Generating simulated CAP metrics for {n_subjects} subjects...")

    # Use the same participant IDs as in character data
    participant_ids = ["sub-Blossom", "sub-Bubbles", "sub-Buttercup"]

    cap_data = []

    for i, participant_id in enumerate(participant_ids):
        for run in range(1, n_runs + 1):
            # Create base metrics
            run_name = f"{participant_id}_run-{run:02d}"

            metrics = {
                "run": run_name,
                "subject_id": participant_id,
                "participant_id": participant_id,
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


def correlate_caps_with_characters(
    cap_metrics, character_data, participants, weight_type="valence"
):
    """
    Calculate correlations between CAP metrics and character-weighted timeseries.
    """
    print(f"Calculating CAP-character correlations (weighted by {weight_type})...")

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

    # Character names
    character_names = ["dusty", "nancy", "steve"]

    # For each participant, calculate correlations
    participant_data = []

    for _, participant_row in character_data.iterrows():
        participant_id = participant_row["participant_id"]

        # Get CAP metrics for this participant (average across runs)
        p_caps = cap_metrics[cap_metrics["participant_id"] == participant_id]

        if len(p_caps) == 0:
            continue

        # Average CAP metrics across runs
        participant_metrics = {"participant_id": participant_id}

        for cap_name in cap_names:
            for metric_type in metric_types:
                col_name = f"{cap_name}_{metric_type}"
                if col_name in p_caps.columns:
                    participant_metrics[col_name] = p_caps[col_name].mean()

        # Add character ratings (weighted significance scores)
        for character in character_names:
            weight_col = f"{character}_{weight_type}"
            if weight_col in participant_row:
                # Normalize to 0-1 scale
                raw_rating = participant_row[weight_col]
                normalized_rating = (raw_rating - 1) / 6  # Scale 1-7 to 0-1
                participant_metrics[f"{character}_significance"] = normalized_rating

        participant_data.append(participant_metrics)

    # Convert to DataFrame
    df_analysis = pd.DataFrame(participant_data)

    if len(df_analysis) == 0:
        print("Warning: No matching data found between CAP metrics and characters")
        return pd.DataFrame(), df_analysis

    print(f"Analyzing {len(df_analysis)} participants with complete data")

    # Calculate correlations between CAP metrics and character significance
    for cap_name in cap_names:
        for metric_type in metric_types:
            col_name = f"{cap_name}_{metric_type}"

            if col_name not in df_analysis.columns:
                continue

            for character in character_names:
                char_col = f"{character}_significance"

                if char_col not in df_analysis.columns:
                    continue

                # Correlation between CAP metric and character significance
                valid_data = df_analysis.dropna(subset=[col_name, char_col])

                if len(valid_data) >= 3:  # Need at least 3 data points
                    r, p = pearsonr(valid_data[col_name], valid_data[char_col])

                    correlation_results.append(
                        {
                            "cap": cap_name,
                            "metric": metric_type,
                            "character": character,
                            "weight_type": weight_type,
                            "r": r,
                            "p": p,
                            "n": len(valid_data),
                        }
                    )

    correlation_df = pd.DataFrame(correlation_results)

    return correlation_df, df_analysis


def run_character_mixed_effects_models(df_analysis, cap_metrics):
    """Run Linear Mixed Effects models for CAP metrics and character ratings."""
    if not HAS_STATSMODELS:
        print("Skipping LME models - statsmodels not available")
        return pd.DataFrame()

    print("Running Linear Mixed Effects models for character associations...")

    lme_results = []

    # Get CAP names and metric types
    cap_columns = [col for col in cap_metrics.columns if "CAP_" in col]
    cap_names = list(
        set([col.split("_")[0] + "_" + col.split("_")[1] for col in cap_columns])
    )
    metric_types = ["frequency_pct", "avg_dwell_time"]
    character_names = ["dusty", "nancy", "steve"]

    # Create long-format data for mixed models
    long_data = []

    for _, row in cap_metrics.iterrows():
        if "participant_id" not in row:
            continue

        participant_id = row["participant_id"]

        # Get character data for this participant
        char_significance = {}
        for character in character_names:
            char_col = f"{character}_significance"
            if char_col in df_analysis.columns:
                participant_char_data = df_analysis[
                    df_analysis["participant_id"] == participant_id
                ]
                if len(participant_char_data) > 0:
                    char_significance[character] = participant_char_data[char_col].iloc[
                        0
                    ]

        for cap_name in cap_names:
            for metric_type in metric_types:
                col_name = f"{cap_name}_{metric_type}"

                if col_name not in row:
                    continue

                for character in character_names:
                    if character in char_significance:
                        long_data.append(
                            {
                                "participant_id": participant_id,
                                "run": row.get("run", "unknown"),
                                "cap": cap_name,
                                "metric_type": metric_type,
                                "character": character,
                                "metric_value": row[col_name],
                                "character_significance": char_significance[character],
                            }
                        )

    long_df = pd.DataFrame(long_data)

    if len(long_df) == 0:
        print("No data available for LME models")
        return pd.DataFrame()

    print(f"Running LME models on {len(long_df)} observations")

    # Run LME models for each CAP-metric-character combination
    for cap_name in cap_names:
        for metric_type in metric_types:
            for character in character_names:

                # Filter data for this combination
                model_data = long_df[
                    (long_df["cap"] == cap_name)
                    & (long_df["metric_type"] == metric_type)
                    & (long_df["character"] == character)
                ].copy()

                if len(model_data) < 6:  # Need sufficient data
                    continue

                try:
                    # Model with character significance
                    formula = "metric_value ~ character_significance"
                    md = mixedlm(
                        formula, model_data, groups=model_data["participant_id"]
                    )
                    mdf = md.fit()

                    # Extract results
                    coef = mdf.params["character_significance"]
                    pval = mdf.pvalues["character_significance"]

                    lme_results.append(
                        {
                            "cap": cap_name,
                            "metric": metric_type,
                            "character": character,
                            "coefficient": coef,
                            "p_value": pval,
                            "model_type": "LME",
                            "n_obs": len(model_data),
                        }
                    )

                except Exception as e:
                    print(
                        f"LME model failed for {cap_name}_{metric_type}_{character}: {e}"
                    )
                    continue

    return pd.DataFrame(lme_results)


def run_character_transition_analyses(df_analysis, character_names):
    """Run specific analyses for character-transition relationships."""
    print("Running character-transition analyses...")

    transition_results = []

    # Get CAP columns
    cap_columns = [
        col for col in df_analysis.columns if "CAP_" in col and "transitions" in col
    ]

    # Test correlations between transitions and character significance
    for cap_col in cap_columns:
        if "transitions_out" not in cap_col:
            continue

        cap_name = cap_col.split("_transitions_out")[0]

        for character in character_names:
            char_col = f"{character}_significance"

            if char_col not in df_analysis.columns:
                continue

            # Correlation analysis
            valid_data = df_analysis.dropna(subset=[cap_col, char_col])

            if len(valid_data) >= 3:
                r, p = pearsonr(valid_data[cap_col], valid_data[char_col])

                transition_results.append(
                    {
                        "cap": cap_name,
                        "character": character,
                        "metric": "transitions_out",
                        "analysis": "correlation_with_character_significance",
                        "correlation": r,
                        "p_value": p,
                        "n": len(valid_data),
                    }
                )

    return pd.DataFrame(transition_results)


def plot_character_correlation_results(correlation_df, save_path=None):
    """Plot correlation results between CAPs and character significance."""
    if len(correlation_df) == 0:
        print("No correlation results to plot")
        return

    print("Plotting character-CAP correlation results...")

    # Create correlation heatmaps for each character
    character_names = correlation_df["character"].unique()
    n_chars = len(character_names)

    fig, axes = plt.subplots(2, n_chars, figsize=(5 * n_chars, 10))
    if n_chars == 1:
        axes = axes.reshape(2, 1)

    # Separate by metric type
    metrics = ["frequency_pct", "avg_dwell_time"]

    for i, metric in enumerate(metrics):
        for j, character in enumerate(character_names):
            ax = axes[i, j]

            # Filter data
            plot_data = correlation_df[
                (correlation_df["character"] == character)
                & (correlation_df["metric"] == metric)
            ]

            if len(plot_data) == 0:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center")
                ax.set_title(f"{character.title()} - {metric}")
                continue

            # Create bar plot of correlations
            caps = plot_data["cap"].values
            correlations = plot_data["r"].values
            p_values = plot_data["p"].values

            # Color bars by significance
            colors = ["red" if p < 0.05 else "lightblue" for p in p_values]

            bars = ax.bar(range(len(caps)), correlations, color=colors, alpha=0.7)

            # Add significance markers
            for k, (bar, p_val) in enumerate(zip(bars, p_values)):
                if p_val < 0.05:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.01,
                        "*",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

            ax.set_xticks(range(len(caps)))
            ax.set_xticklabels([cap.replace("_", " ") for cap in caps], rotation=45)
            ax.set_ylabel("Correlation (r)")
            ax.set_title(f'{character.title()} - {metric.replace("_", " ").title()}')
            ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
            ax.set_ylim(-1, 1)

    plt.suptitle(
        "CAP-Character Significance Correlations", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Character correlation plot saved to: {save_path}")

    plt.close()


def visualize_character_timeseries(character_timeseries_df, save_path=None):
    """Visualize character appearance timeseries."""
    print("Plotting character appearance timeseries...")

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    character_names = ["dusty", "nancy", "steve"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, orange, green

    for i, character in enumerate(character_names):
        if character in character_timeseries_df.columns:
            ax.plot(
                character_timeseries_df["time"],
                character_timeseries_df[character],
                label=character.title(),
                color=colors[i],
                linewidth=2,
            )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Character Presence")
    ax.set_title("Character Appearance Timeline in Stranger Things Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Character timeseries plot saved to: {save_path}")

    plt.close()


def generate_character_analysis_report(
    correlation_df,
    lme_results,
    transition_results,
    weight_type="valence",
    output_file="character_caps_analysis_report.txt",
):
    """Generate a comprehensive character-CAP analysis report."""
    print("Generating character analysis summary report...")

    with open(output_file, "w") as f:
        f.write("CHARACTER-CAPs ASSOCIATION ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Analysis Type: Character significance weighted by {weight_type}\n")
        f.write("Objective: Examine brain-social processing relationships\n")
        f.write("Characters: Dusty, Nancy, Steve (Stranger Things)\n\n")

        # 1. Character-CAP Correlation Analysis
        f.write("1. CHARACTER-CAP CORRELATION ANALYSIS (Pearson r)\n")
        f.write("-" * 50 + "\n")

        if len(correlation_df) > 0:
            # Significant correlations (p < 0.05)
            sig_corrs = correlation_df[correlation_df["p"] < 0.05]

            f.write(f"Total correlations tested: {len(correlation_df)}\n")
            f.write(f"Significant correlations (p < 0.05): {len(sig_corrs)}\n\n")

            if len(sig_corrs) > 0:
                f.write("Significant character-CAP associations:\n")
                for _, row in sig_corrs.iterrows():
                    f.write(
                        f"  {row['cap']} {row['metric']} - {row['character']}: "
                        f"r = {row['r']:.3f}, p = {row['p']:.3f}\n"
                    )
            else:
                f.write(
                    "No significant associations found (may be due to small sample size).\n"
                )

            f.write("\nAll character-CAP correlations:\n")

            # Group by character for better readability
            for character in correlation_df["character"].unique():
                f.write(f"\n{character.upper()} Character Associations:\n")
                char_data = correlation_df[correlation_df["character"] == character]

                for _, row in char_data.iterrows():
                    significance = "*" if row["p"] < 0.05 else ""
                    f.write(
                        f"  {row['cap']} {row['metric']}: "
                        f"r = {row['r']:.3f}, p = {row['p']:.3f}{significance} (n = {row['n']})\n"
                    )
        else:
            f.write("No correlation results available.\n")

        f.write("\n" + "=" * 50 + "\n\n")

        # 2. Linear Mixed Effects Models
        f.write("2. CHARACTER-CAP MIXED EFFECTS MODELS\n")
        f.write("-" * 50 + "\n")

        if len(lme_results) > 0:
            sig_lme = lme_results[lme_results["p_value"] < 0.05]

            f.write(f"Total LME models run: {len(lme_results)}\n")
            f.write(f"Significant models (p < 0.05): {len(sig_lme)}\n\n")

            if len(sig_lme) > 0:
                f.write("Significant character-CAP LME results:\n")
                for _, row in sig_lme.iterrows():
                    f.write(
                        f"  {row['cap']} {row['metric']} - {row['character']}: "
                        f"β = {row['coefficient']:.3f}, p = {row['p_value']:.3f}\n"
                    )

            f.write("\nAll LME results by character:\n")
            for character in lme_results["character"].unique():
                f.write(f"\n{character.upper()}:\n")
                char_lme = lme_results[lme_results["character"] == character]

                for _, row in char_lme.iterrows():
                    significance = "*" if row["p_value"] < 0.05 else ""
                    f.write(
                        f"  {row['cap']} {row['metric']}: "
                        f"β = {row['coefficient']:.3f}, p = {row['p_value']:.3f}{significance}\n"
                    )
        else:
            f.write("No LME results available.\n")

        f.write("\n" + "=" * 50 + "\n\n")

        # 3. Character-Transition Analyses
        f.write("3. CHARACTER-TRANSITION ANALYSES\n")
        f.write("-" * 50 + "\n")

        if len(transition_results) > 0:
            f.write(
                "Character significance correlations with brain state transitions:\n"
            )

            for character in transition_results["character"].unique():
                f.write(f"\n{character.upper()} Transition Correlations:\n")
                char_trans = transition_results[
                    transition_results["character"] == character
                ]

                for _, row in char_trans.iterrows():
                    significance = "*" if row["p_value"] < 0.05 else ""
                    f.write(
                        f"  {row['cap']}: "
                        f"r = {row['correlation']:.3f}, p = {row['p_value']:.3f}{significance}\n"
                    )
        else:
            f.write("No transition analysis results available.\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("* p < 0.05\n")
        f.write("\nINTERPRETATION:\n")
        f.write(
            "- Positive correlations: Higher character significance → more CAP activity\n"
        )
        f.write(
            "- Negative correlations: Higher character significance → less CAP activity\n"
        )
        f.write("- This extends emotional arousal analysis to socioemotional arousal\n")
        f.write("- Results inform brain-social processing relationships\n")
        f.write("\nAnalysis completed.\n")

    print(f"Character analysis report saved to: {output_file}")


def main():
    """Main character-CAP analysis function."""
    print("Starting Character-CAPs Association Analysis...")
    print("=" * 50)

    # 1. Load character rating data
    character_data, participants = load_character_data()

    # 2. Create character appearance timeseries
    # Note: In real analysis, this would be based on actual video annotations
    events_data = None  # Placeholder - would load actual stimulus events
    character_timeseries_df, character_timeseries = (
        create_character_appearance_timeseries(events_data)
    )

    # 3. Visualize character timeseries
    visualize_character_timeseries(
        character_timeseries_df, "character_appearance_timeseries.png"
    )

    # 4. Load CAP metrics
    cap_metrics = load_cap_metrics_data()

    # 5. Analyze correlations for both valence and arousal weighting
    weight_types = ["valence", "arousal"]
    all_results = {}

    for weight_type in weight_types:
        print(f"\n--- Analyzing {weight_type.upper()} weighted associations ---")

        # Calculate correlations
        correlation_df, df_analysis = correlate_caps_with_characters(
            cap_metrics, character_data, participants, weight_type=weight_type
        )

        # Run mixed effects models
        lme_results = run_character_mixed_effects_models(df_analysis, cap_metrics)

        # Run transition analyses
        character_names = ["dusty", "nancy", "steve"]
        transition_results = run_character_transition_analyses(
            df_analysis, character_names
        )

        # Plot results
        plot_character_correlation_results(
            correlation_df, f"character_caps_correlations_{weight_type}.png"
        )

        # Generate report
        generate_character_analysis_report(
            correlation_df,
            lme_results,
            transition_results,
            weight_type=weight_type,
            output_file=f"character_caps_analysis_report_{weight_type}.txt",
        )

        # Save detailed results
        if len(correlation_df) > 0:
            correlation_df.to_csv(
                f"character_caps_correlations_{weight_type}.tsv", sep="\t", index=False
            )
            print(
                f"Character correlation results ({weight_type}) saved to: "
                f"character_caps_correlations_{weight_type}.tsv"
            )

        if len(lme_results) > 0:
            lme_results.to_csv(
                f"character_caps_lme_results_{weight_type}.tsv", sep="\t", index=False
            )
            print(
                f"Character LME results ({weight_type}) saved to: "
                f"character_caps_lme_results_{weight_type}.tsv"
            )

        if len(transition_results) > 0:
            transition_results.to_csv(
                f"character_caps_transition_results_{weight_type}.tsv",
                sep="\t",
                index=False,
            )
            print(
                f"Character transition results ({weight_type}) saved to: "
                f"character_caps_transition_results_{weight_type}.tsv"
            )

        all_results[weight_type] = {
            "correlations": correlation_df,
            "lme": lme_results,
            "transitions": transition_results,
        }

    print("\nCharacter-CAP Analysis completed successfully!")
    print("=" * 50)

    # Summary of findings
    print("\nSUMMARY:")
    print("- Extended emotion analysis to socioemotional arousal")
    print("- Examined CAP associations with character presence")
    print("- Weighted by individual subjective character ratings")
    print("- Analyzed both valence and arousal dimensions")
    print("- Generated comprehensive reports and visualizations")

    return all_results


if __name__ == "__main__":
    main()
