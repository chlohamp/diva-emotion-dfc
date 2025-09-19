import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from scipy.stats import pearsonr, ttest_1samp, f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# For mixed-effects models
try:

    from statsmodels.formula.api import mixedlm

    HAS_STATSMODELS = True
except ImportError:
    print("Warning: statsmodels not available. Install with: pip install statsmodels")
    HAS_STATSMODELS = False

# Set non-interactive backend for matplotlib
plt.switch_backend("Agg")


def load_character_data(
    participants_characters_file="derivatives/simulated/participants-characters.tsv",
    participants_file="derivatives/simulated/participants.tsv",
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


def load_character_events_data(
    events_file_a1="derivatives/simulated/ses-01_task-strangerthings_acq-A1_run-1_events.tsv",
    events_file_a2="derivatives/simulated/ses-01_task-strangerthings_acq-A2_run-1_events.tsv",
):
    """Load events data with character appearances."""
    print("Loading character events data...")

    # Load the events data
    events_a1 = pd.read_csv(events_file_a1, sep="\t")
    events_a2 = pd.read_csv(events_file_a2, sep="\t")

    print(f"Loaded {len(events_a1)} events from A1, {len(events_a2)} events from A2")

    if "characters" in events_a1.columns:
        print("Found 'characters' column in events data")
        print(f"Characters found: {events_a1['characters'].unique()}")
    else:
        print("WARNING: 'characters' column not found in events data")
        return pd.DataFrame()

    # For this analysis, we'll use the A1 events (or average between raters)
    # In practice, you might want to average emotion ratings between raters
    return events_a1


def calculate_character_specific_cap_metrics(
    cluster_labels, events_data, tr_duration=1.5, participant_id=None
):
    """
    Calculate CAP metrics for specific character appearance conditions.
    Focus on dwell time calculations when characters are on screen.

    Parameters:
    -----------
    cluster_labels : array
        CAP assignments for each TR
    events_data : DataFrame
        Contains onset, duration, characters for each condition
    tr_duration : float
        Duration of each TR in seconds (default: 1.5s)
    participant_id : str
        Participant identifier

    Returns:
    --------
    character_condition_metrics : DataFrame
        CAP metrics calculated for each character appearance condition
    """
    print(
        f"Calculating character-specific CAP dwell time metrics for {participant_id}..."
    )

    if len(events_data) == 0:
        return pd.DataFrame()

    condition_results = []
    unique_clusters = np.unique(cluster_labels)

    # Get unique characters
    all_characters = set()
    for characters_str in events_data["characters"].dropna():
        # Handle cases where multiple characters are listed (e.g., "Mike Wheeler; Eleven")
        chars = [char.strip() for char in str(characters_str).split(";")]
        all_characters.update(chars)

    print(f"Found characters: {list(all_characters)}")

    # Process each event/condition
    for idx, condition in events_data.iterrows():
        onset = condition["onset"]
        duration = condition["duration"]
        characters = condition["characters"]

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

        # Parse characters on screen
        if pd.isna(characters):
            characters_list = []
        else:
            characters_list = [char.strip() for char in str(characters).split(";")]

        # Initialize condition metrics
        condition_metrics = {
            "participant_id": participant_id,
            "condition_idx": idx,
            "onset": onset,
            "duration": duration,
            "start_tr": start_tr,
            "end_tr": end_tr,
            "n_trs": len(condition_labels),
            "characters": characters,
            "n_characters": len(characters_list),
            # Add individual character presence indicators
            "mike_wheeler_present": int("Mike Wheeler" in characters_list),
            "eleven_present": int("Eleven" in characters_list),
            "nancy_present": int("Nancy" in characters_list),
        }

        # Calculate comprehensive dwell time metrics for each CAP
        for cluster in unique_clusters:
            cap_name = f"CAP_{cluster + 1}"

            # 1. Frequency of occurrence (percentage of condition time)
            cluster_trs = np.sum(condition_labels == cluster)
            frequency_pct = (cluster_trs / len(condition_labels)) * 100
            condition_metrics[f"{cap_name}_frequency_pct"] = frequency_pct

            # 2. DWELL TIME CALCULATION - Key focus for character analysis
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

            # Calculate comprehensive dwell time statistics
            if dwell_times:
                avg_dwell_time = np.mean(dwell_times)
                max_dwell_time = np.max(dwell_times)
                min_dwell_time = np.min(dwell_times)
                std_dwell_time = np.std(dwell_times) if len(dwell_times) > 1 else 0
                n_visits = len(dwell_times)
                total_dwell_time = np.sum(dwell_times)
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

        condition_results.append(condition_metrics)

    condition_df = pd.DataFrame(condition_results)

    if len(condition_df) > 0:
        print(f"  Calculated metrics for {len(condition_df)} character conditions")
        print(
            f"  Conditions range from {condition_df['duration'].min():.1f}s to {condition_df['duration'].max():.1f}s"
        )
        print(f"  Average TRs per condition: {condition_df['n_trs'].mean():.1f}")

        # Report character presence statistics
        character_counts = {
            "Mike Wheeler": condition_df["mike_wheeler_present"].sum(),
            "Eleven": condition_df["eleven_present"].sum(),
            "Nancy": condition_df["nancy_present"].sum(),
        }

        for char, count in character_counts.items():
            percentage = (count / len(condition_df)) * 100
            print(
                f"  {char}: present in {count}/{len(condition_df)} conditions ({percentage:.1f}%)"
            )

    return condition_df


def load_caps_and_calculate_character_metrics(
    events_data,
    reliable_participants,
    caps_results_dir="derivatives/caps/cap-analysis/sub_Bubbles_ses_01",
    tr_duration=1.5,
):
    """
    Load CAP clustering results and calculate character-specific metrics.
    """
    print("Loading CAPs results and calculating character-specific metrics...")

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

        # Try to detect number of clusters from CAP metrics
        try:
            # Try to load cap metrics to get number of clusters
            cap_metrics_file = f"{caps_results_dir}/cap_metrics.tsv"
            cap_metrics = pd.read_csv(cap_metrics_file, sep="\t")
            n_clusters = (
                cap_metrics["n_clusters"].iloc[0]
                if "n_clusters" in cap_metrics.columns
                else 4
            )
            print(f"Detected {n_clusters} clusters from CAP metrics")
        except FileNotFoundError:
            print("Could not find CAP metrics file, using default 4 clusters")
            n_clusters = 4

        # Generate simulated cluster labels with detected number of clusters
        n_timepoints = 300  # Approximate length for demo
        cluster_labels = np.random.randint(0, n_clusters, n_timepoints)

    # Calculate character-specific metrics for each participant
    all_character_metrics = []

    for participant_id in reliable_participants:
        print(f"\nProcessing {participant_id}...")

        # Get events data for this participant (expand events to all participants)
        participant_events = events_data.copy()

        if len(participant_events) == 0:
            print(f"  No events data for {participant_id}")
            continue

        # Calculate character-specific CAP metrics
        character_metrics = calculate_character_specific_cap_metrics(
            cluster_labels=cluster_labels,
            events_data=participant_events,
            tr_duration=tr_duration,
            participant_id=participant_id,
        )

        if len(character_metrics) > 0:
            all_character_metrics.append(character_metrics)

    # Combine all participants' character metrics
    if all_character_metrics:
        combined_metrics = pd.concat(all_character_metrics, ignore_index=True)
        print(
            f"\nCombined metrics: {len(combined_metrics)} conditions across {len(reliable_participants)} participants"
        )

        return combined_metrics
    else:
        print("No character metrics calculated")
        return pd.DataFrame()


def analyze_dwell_time_by_character_presence(character_metrics_df):
    """
    Analyze dwell time patterns by character presence with ANOVA tests.

    This function:
    1. Compares dwell time when specific characters are present vs absent
    2. Runs ANOVA tests for each CAP's dwell time metrics
    3. Tests main effects of individual characters
    """
    print("\n=== DWELL TIME BY CHARACTER PRESENCE WITH ANOVA ===")

    if len(character_metrics_df) == 0:
        print("No character metrics data available")
        return []

    # Get dwell time columns
    dwell_cols = [
        col
        for col in character_metrics_df.columns
        if "avg_dwell_time" in col and "sec" not in col
    ]

    if not dwell_cols:
        print("No dwell time metrics found")
        return []

    print(f"Analyzing {len(dwell_cols)} dwell time metrics by character presence...")

    # Character presence indicators
    character_indicators = ["mike_wheeler_present", "eleven_present", "nancy_present"]
    character_names = ["Mike Wheeler", "Eleven", "Nancy"]

    anova_results = []

    # Test each dwell time metric
    for col in dwell_cols:
        print(f"\n--- {col} ---")

        if col not in character_metrics_df.columns:
            continue

        valid_data = character_metrics_df[[col] + character_indicators].dropna()

        if len(valid_data) < 6:  # Need at least 6 observations
            print(f"  Insufficient data (n={len(valid_data)})")
            continue

        # Test each character individually
        for char_indicator, char_name in zip(character_indicators, character_names):

            # Group data by character presence
            present_data = valid_data[valid_data[char_indicator] == 1][col].values
            absent_data = valid_data[valid_data[char_indicator] == 0][col].values

            if len(present_data) >= 2 and len(absent_data) >= 2:
                try:
                    # Run ANOVA (t-test for 2 groups)
                    f_stat, p_value = f_oneway(present_data, absent_data)

                    # Determine significance
                    significant = p_value < 0.05
                    sig_marker = (
                        "***"
                        if p_value < 0.001
                        else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    )

                    print(f"  {char_name}:")
                    print(
                        f"    F(1, {len(valid_data)-2}) = {f_stat:.3f}, p = {p_value:.3f}{sig_marker}"
                    )

                    if significant:
                        print(
                            f"    *** SIGNIFICANT: Dwell time differs when {char_name} is on screen ***"
                        )

                    # Show group means
                    present_mean = np.mean(present_data)
                    absent_mean = np.mean(absent_data)
                    print(
                        f"    {char_name} present: M = {present_mean:.2f} (n = {len(present_data)})"
                    )
                    print(
                        f"    {char_name} absent: M = {absent_mean:.2f} (n = {len(absent_data)})"
                    )

                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(
                        (
                            (len(present_data) - 1) * np.var(present_data, ddof=1)
                            + (len(absent_data) - 1) * np.var(absent_data, ddof=1)
                        )
                        / (len(present_data) + len(absent_data) - 2)
                    )
                    cohens_d = (
                        (present_mean - absent_mean) / pooled_std
                        if pooled_std > 0
                        else 0
                    )

                    effect_size_label = (
                        "large"
                        if abs(cohens_d) > 0.8
                        else "medium" if abs(cohens_d) > 0.5 else "small"
                    )
                    print(f"    Effect size: d = {cohens_d:.3f} ({effect_size_label})")

                    # Store results
                    anova_results.append(
                        {
                            "cap_metric": col,
                            "character": char_name,
                            "f_statistic": f_stat,
                            "p_value": p_value,
                            "significant": significant,
                            "present_mean": present_mean,
                            "absent_mean": absent_mean,
                            "present_n": len(present_data),
                            "absent_n": len(absent_data),
                            "cohens_d": cohens_d,
                            "effect_size": effect_size_label,
                            "total_n": len(valid_data),
                        }
                    )

                except Exception as e:
                    print(f"    ANOVA failed for {char_name}: {e}")
                    continue
            else:
                print(f"  {char_name}: Insufficient data in one or both groups")
                print(
                    f"    Present: n={len(present_data)}, Absent: n={len(absent_data)}"
                )

    # Summary of ANOVA results
    if anova_results:
        print(f"\n--- CHARACTER PRESENCE ANOVA SUMMARY ---")
        significant_results = [r for r in anova_results if r["significant"]]

        print(f"Total ANOVA tests: {len(anova_results)}")
        print(f"Significant results: {len(significant_results)}")

        if significant_results:
            print(f"\nSignificant dwell time differences by character presence:")
            for result in significant_results:
                direction = (
                    "higher"
                    if result["present_mean"] > result["absent_mean"]
                    else "lower"
                )
                print(
                    f"  {result['cap_metric']} - {result['character']}: {direction} when present"
                )
                print(
                    f"    F = {result['f_statistic']:.3f}, p = {result['p_value']:.3f}, d = {result['cohens_d']:.3f}"
                )
        else:
            print(
                f"\nNo significant dwell time differences found by character presence"
            )

        # Character-specific summary
        print(f"\n--- CHARACTER-SPECIFIC EFFECTS ---")
        for char_name in character_names:
            char_results = [r for r in anova_results if r["character"] == char_name]
            char_significant = [r for r in char_results if r["significant"]]

            if char_results:
                print(
                    f"{char_name}: {len(char_significant)}/{len(char_results)} significant effects"
                )
                if char_significant:
                    for result in char_significant:
                        cap_name = result["cap_metric"].replace("_avg_dwell_time", "")
                        direction = (
                            "↑"
                            if result["present_mean"] > result["absent_mean"]
                            else "↓"
                        )
                        print(
                            f"  {cap_name}: {direction} dwell time when present (p = {result['p_value']:.3f})"
                        )

    else:
        print("No ANOVA tests could be performed")

    return anova_results


def correlate_character_metrics_with_cap_dwell_times(character_metrics_df):
    """
    Calculate correlations between character presence and CAP dwell time metrics.
    """
    print("Calculating correlations between character presence and CAP dwell times...")

    if len(character_metrics_df) == 0:
        return pd.DataFrame()

    correlation_results = []

    # Get dwell time columns
    dwell_cols = [
        col
        for col in character_metrics_df.columns
        if "avg_dwell_time" in col and "sec" not in col
    ]

    # Character presence indicators
    character_indicators = ["mike_wheeler_present", "eleven_present", "nancy_present"]
    character_names = ["Mike Wheeler", "Eleven", "Nancy"]

    # Calculate correlations
    for dwell_col in dwell_cols:
        cap_name = dwell_col.replace("_avg_dwell_time", "")

        for char_indicator, char_name in zip(character_indicators, character_names):

            # Get valid data
            valid_data = character_metrics_df[[dwell_col, char_indicator]].dropna()

            if len(valid_data) >= 3:  # Need at least 3 data points
                try:
                    r, p_value = pearsonr(
                        valid_data[dwell_col], valid_data[char_indicator]
                    )

                    # Determine effect size
                    if abs(r) < 0.1:
                        effect_size = "negligible"
                    elif abs(r) < 0.3:
                        effect_size = "small"
                    elif abs(r) < 0.5:
                        effect_size = "medium"
                    else:
                        effect_size = "large"

                    correlation_results.append(
                        {
                            "cap": cap_name,
                            "character": char_name,
                            "metric": "avg_dwell_time",
                            "r": r,
                            "p_value": p_value,
                            "significant": p_value < 0.05,
                            "effect_size": effect_size,
                            "n": len(valid_data),
                        }
                    )

                except Exception as e:
                    print(f"Correlation failed for {cap_name} - {char_name}: {e}")

    results_df = pd.DataFrame(correlation_results)

    if len(results_df) > 0:
        n_significant = (results_df["p_value"] < 0.05).sum()
        print(f"Calculated {len(results_df)} correlations, {n_significant} significant")

        # Show significant correlations
        if n_significant > 0:
            print("Significant character-dwell time correlations:")
            sig_results = results_df[results_df["significant"]].sort_values("p_value")
            for _, row in sig_results.iterrows():
                direction = "positive" if row["r"] > 0 else "negative"
                print(
                    f"  {row['cap']} - {row['character']}: r = {row['r']:.3f}, p = {row['p_value']:.3f} ({direction})"
                )

    return results_df


def generate_character_dwell_time_report(
    character_metrics_df,
    correlation_results,
    anova_results,
    output_file="character_dwell_time_analysis_report.txt",
):
    """Generate a comprehensive report on character dwell time findings."""
    print("\nGenerating character dwell time analysis report...")

    with open(output_file, "w") as f:
        f.write("CHARACTER DWELL TIME ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write("ANALYSIS OVERVIEW\n")
        f.write("-" * 17 + "\n")
        f.write(
            "This analysis examined how CAP dwell time varies with character presence.\n"
        )
        f.write(
            "Dwell time = average number of consecutive TRs spent in each CAP state.\n"
        )
        f.write("Conditions defined by character appearances on screen.\n")
        f.write("Characters analyzed: Mike Wheeler, Eleven, Nancy\n\n")

        if len(character_metrics_df) > 0:
            f.write(f"Total conditions analyzed: {len(character_metrics_df)}\n")
            f.write(
                f"Participants: {character_metrics_df['participant_id'].nunique()}\n"
            )
            f.write(
                f"Condition duration range: {character_metrics_df['duration'].min():.1f}s - {character_metrics_df['duration'].max():.1f}s\n\n"
            )

            # Character presence statistics
            f.write("CHARACTER PRESENCE STATISTICS:\n")
            character_stats = {
                "Mike Wheeler": character_metrics_df["mike_wheeler_present"].sum(),
                "Eleven": character_metrics_df["eleven_present"].sum(),
                "Nancy": character_metrics_df["nancy_present"].sum(),
            }

            for char, count in character_stats.items():
                percentage = (count / len(character_metrics_df)) * 100
                f.write(
                    f"  {char}: {count}/{len(character_metrics_df)} conditions ({percentage:.1f}%)\n"
                )
            f.write("\n")

        # ANOVA Results
        if anova_results and len(anova_results) > 0:
            f.write("ANOVA RESULTS: DWELL TIME BY CHARACTER PRESENCE\n")
            f.write("-" * 47 + "\n")

            significant_anovas = [r for r in anova_results if r["significant"]]
            f.write(f"Total ANOVA tests performed: {len(anova_results)}\n")
            f.write(f"Significant differences found: {len(significant_anovas)}\n\n")

            if significant_anovas:
                f.write("SIGNIFICANT ANOVA RESULTS:\n")
                for result in significant_anovas:
                    direction = (
                        "higher"
                        if result["present_mean"] > result["absent_mean"]
                        else "lower"
                    )
                    f.write(f"\n{result['cap_metric']} - {result['character']}:\n")
                    f.write(
                        f"  F(1, {result['total_n']-2}) = {result['f_statistic']:.3f}, p = {result['p_value']:.3f}\n"
                    )
                    f.write(
                        f"  Effect: Dwell time is {direction} when {result['character']} is present\n"
                    )
                    f.write(
                        f"  Present: M = {result['present_mean']:.2f} (n = {result['present_n']})\n"
                    )
                    f.write(
                        f"  Absent: M = {result['absent_mean']:.2f} (n = {result['absent_n']})\n"
                    )
                    f.write(
                        f"  Effect size: d = {result['cohens_d']:.3f} ({result['effect_size']})\n"
                    )

                f.write("\nINTERPRETATION:\n")
                f.write(
                    "Significant ANOVA results indicate that dwell time in these CAP states\n"
                )
                f.write(
                    "differs significantly when specific characters are on screen vs. absent.\n\n"
                )
            else:
                f.write("No significant ANOVA results found.\n")
                f.write(
                    "Dwell time does not significantly differ by character presence.\n\n"
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
                    f"{result['cap_metric']} - {result['character']}: F = {result['f_statistic']:.3f}, p = {result['p_value']:.3f}{sig_marker}\n"
                )

            f.write("\n")

        # Correlation Results
        if len(correlation_results) > 0:
            f.write("CHARACTER PRESENCE - DWELL TIME CORRELATIONS\n")
            f.write("-" * 42 + "\n")

            sig_corrs = correlation_results[correlation_results["significant"]]
            f.write(f"Total correlations calculated: {len(correlation_results)}\n")
            f.write(f"Significant correlations: {len(sig_corrs)}\n\n")

            if len(sig_corrs) > 0:
                f.write("Significant correlations:\n")
                for _, row in sig_corrs.sort_values("p_value").iterrows():
                    direction = "positive" if row["r"] > 0 else "negative"
                    f.write(
                        f"  {row['cap']} - {row['character']}: r = {row['r']:.3f}, p = {row['p_value']:.3f} ({direction}, {row['effect_size']})\n"
                    )

            f.write("\nAll correlations by character:\n")
            for character in correlation_results["character"].unique():
                f.write(f"\n{character.upper()}:\n")
                char_corrs = correlation_results[
                    correlation_results["character"] == character
                ]
                for _, row in char_corrs.iterrows():
                    sig_marker = "*" if row["significant"] else ""
                    f.write(
                        f"  {row['cap']}: r = {row['r']:.3f}, p = {row['p_value']:.3f}{sig_marker}\n"
                    )

        f.write(f"\n* p < 0.05, ** p < 0.01, *** p < 0.001\n")
        f.write("Analysis completed.\n")

    print(f"Character dwell time report saved to: {output_file}")


def main_character_dwell_time_analysis():
    """
    Main analysis function for character-specific dwell time analysis.
    """
    print("=== CHARACTER-SPECIFIC DWELL TIME ANALYSIS ===")
    print("Analyzing how dwell time varies with character presence on screen")
    print("=" * 60)

    # Create output directories
    from pathlib import Path

    output_dir = Path("derivatives/caps/cap-analysis/sub_Bubbles_ses_01")
    figures_dir = Path("derivatives/caps/cap-analysis/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load character data
    character_data, participants = load_character_data()

    # Get reliable participants (for consistency with other analyses)
    reliable_participants = character_data["participant_id"].tolist()

    # 2. Load character events data
    events_data = load_character_events_data()

    if len(events_data) == 0:
        print("No events data available. Exiting.")
        return

    # 3. Load CAPs and calculate character-specific metrics
    character_metrics = load_caps_and_calculate_character_metrics(
        events_data, reliable_participants
    )

    if len(character_metrics) == 0:
        print("No character metrics calculated. Exiting.")
        return

    # 4. Analyze dwell time by character presence with ANOVA
    anova_results = analyze_dwell_time_by_character_presence(character_metrics)

    # 5. Calculate correlations between character presence and dwell times
    correlation_results = correlate_character_metrics_with_cap_dwell_times(
        character_metrics
    )

    # 6. Save results
    if len(character_metrics) > 0:
        character_metrics.to_csv(
            output_dir / "character_specific_cap_metrics.tsv", sep="\t", index=False
        )
        print(
            f"Character-specific metrics saved to: {output_dir / 'character_specific_cap_metrics.tsv'}"
        )

    if len(correlation_results) > 0:
        correlation_results.to_csv(
            output_dir / "character_dwell_time_correlations.tsv", sep="\t", index=False
        )
        print(
            f"Character correlations saved to: {output_dir / 'character_dwell_time_correlations.tsv'}"
        )

    # 7. Generate comprehensive report
    generate_character_dwell_time_report(
        character_metrics,
        correlation_results,
        anova_results,
        output_file=output_dir / "character_dwell_time_analysis_report.txt",
    )

    print("\n=== CHARACTER-SPECIFIC DWELL TIME ANALYSIS COMPLETED ===")
    return character_metrics, correlation_results, anova_results


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


def load_cap_metrics_data(
    cap_results_file="derivatives/caps/cap-analysis/sub_Bubbles_ses_01/cap_metrics.tsv",
):
    """Load CAP metrics from previous analysis."""
    print("Loading CAP metrics data...")

    # Try to load real CAP metrics first to detect number of CAPs
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
            "derivatives/caps/cap-analysis/sub_Bubbles_ses_01/cap_metrics.tsv",
            "derivatives/caps_results_sub_Bubbles_ses_01/cap_metrics.tsv",
            "caps_results_sub_Bubbles_ses_01/cap_metrics.tsv",
            "derivatives/caps_results/cap_metrics.tsv",
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
    """Generate simulated CAP metrics matching character data participants."""
    if n_caps is None:
        # Auto-detect from real CAP analysis if not specified
        n_caps = 4  # fallback default

    print(
        f"Generating simulated CAP metrics for {n_subjects} subjects with {n_caps} CAPs..."
    )

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

    # Create output directories
    from pathlib import Path

    output_dir = Path("derivatives/caps/cap-analysis/sub_Bubbles_ses_01")
    figures_dir = Path("derivatives/caps/cap-analysis/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

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
        character_timeseries_df, figures_dir / "character_appearance_timeseries.png"
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
            correlation_df,
            figures_dir / f"character_caps_correlations_{weight_type}.png",
        )

        # Generate report
        generate_character_analysis_report(
            correlation_df,
            lme_results,
            transition_results,
            weight_type=weight_type,
            output_file=output_dir
            / f"character_caps_analysis_report_{weight_type}.txt",
        )

        # Save detailed results
        if len(correlation_df) > 0:
            correlation_df.to_csv(
                output_dir / f"character_caps_correlations_{weight_type}.tsv",
                sep="\t",
                index=False,
            )
            print(
                f"Character correlation results ({weight_type}) saved to: "
                f"{output_dir / f'character_caps_correlations_{weight_type}.tsv'}"
            )

        if len(lme_results) > 0:
            lme_results.to_csv(
                output_dir / f"character_caps_lme_results_{weight_type}.tsv",
                sep="\t",
                index=False,
            )
            print(
                f"Character LME results ({weight_type}) saved to: "
                f"{output_dir / f'character_caps_lme_results_{weight_type}.tsv'}"
            )

        if len(transition_results) > 0:
            transition_results.to_csv(
                output_dir / f"character_caps_transition_results_{weight_type}.tsv",
                sep="\t",
                index=False,
            )
            print(
                f"Character transition results ({weight_type}) saved to: "
                f"{output_dir / f'character_caps_transition_results_{weight_type}.tsv'}"
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
    # Choose which analysis to run
    analysis_type = input(
        "Choose analysis type:\n1. Original character-CAP analysis\n2. Character dwell time analysis\nEnter 1 or 2: "
    ).strip()

    if analysis_type == "2":
        main_character_dwell_time_analysis()
    else:
        main()
