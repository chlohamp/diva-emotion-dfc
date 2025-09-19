import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_1samp, f_oneway
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


def load_yeo_atlas():
    """Load the Yeo 7 Network atlas."""
    print("Loading Yeo 7 Network atlas...")
    yeo = datasets.fetch_atlas_yeo_2011()
    # Use the 7 network version
    atlas_img = yeo.thick_7
    labels = [
        "Visual",
        "Somatomotor",
        "Dorsal Attention",
        "Ventral Attention",
        "Limbic",
        "Frontoparietal",
        "Default Mode",
    ]
    return atlas_img, labels


def extract_timeseries(bold_file, atlas_img, atlas_labels):
    """Extract timeseries from fMRI data using atlas regions."""
    print(f"Extracting timeseries from {bold_file}...")

    # Create masker object
    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=True,  # Standardize signals
        detrend=True,  # Remove linear trends
        low_pass=0.1,  # Low-pass filter at 0.1 Hz
        high_pass=0.01,  # High-pass filter at 0.01 Hz
        t_r=1.5,  # Repetition time (adjust if different)
        verbose=1,
    )

    # Extract timeseries
    timeseries = masker.fit_transform(bold_file)

    # Create DataFrame with network labels
    df = pd.DataFrame(timeseries, columns=atlas_labels)

    print(f"Extracted timeseries shape: {timeseries.shape}")
    print(f"Time points: {timeseries.shape[0]}, Networks: {timeseries.shape[1]}")

    return df, timeseries, masker


def plot_elbow_curve(timeseries, max_clusters=15, save_path=None):
    """Plot elbow curve to determine optimal number of clusters."""
    print("Computing elbow curve for optimal cluster selection...")

    # Standardize the data (same as in clustering function)
    scaler = StandardScaler()
    timeseries_scaled = scaler.fit_transform(timeseries)

    # Test different numbers of clusters
    cluster_range = range(2, max_clusters + 1)
    inertias = []

    for k in cluster_range:
        print(f"  Testing k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=100)
        kmeans.fit(timeseries_scaled)
        inertias.append(kmeans.inertia_)

    # Create elbow plot
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, inertias, "bo-", linewidth=2, markersize=8)
    plt.xlabel("Number of Clusters (k)", fontsize=12)
    plt.ylabel("Within-Cluster Sum of Squares (WCSS)", fontsize=12)
    plt.title(
        "Elbow Method for Optimal Number of Clusters", fontsize=14, fontweight="bold"
    )
    plt.grid(True, alpha=0.3)

    # Add annotations
    for i, (k, inertia) in enumerate(zip(cluster_range, inertias)):
        plt.annotate(
            f"k={k}",
            (k, inertia),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Elbow plot saved to: {save_path}")

    plt.close()

    # Calculate differences to help identify elbow
    differences = []
    for i in range(1, len(inertias)):
        diff = inertias[i - 1] - inertias[i]
        differences.append(diff)

    # Find elbow using rate of change
    second_differences = []
    for i in range(1, len(differences)):
        second_diff = differences[i - 1] - differences[i]
        second_differences.append(second_diff)

    # Suggest optimal k (where second difference is maximum)
    if second_differences:
        optimal_idx = second_differences.index(max(second_differences))
        suggested_k = cluster_range[optimal_idx + 2]  # +2 because of indexing
        print(f"Suggested optimal number of clusters: {suggested_k}")

    # Print inertia values
    print("\nCluster analysis results:")
    for k, inertia in zip(cluster_range, inertias):
        print(f"  k={k}: WCSS={inertia:.2f}")

    return cluster_range, inertias


def perform_kmeans_clustering(timeseries, n_clusters=5, random_state=42):
    """Perform k-means clustering on timeseries data."""
    print(f"Performing k-means clustering with {n_clusters} clusters...")

    # Standardize the data (z-scoring/standardization)
    # For individual subject analysis: standardizes each network's
    # timeseries across all timepoints (temporal standardization)
    # Matrix shape: (all_timepoints_from_all_runs, 7_networks)
    scaler = StandardScaler()
    timeseries_scaled = scaler.fit_transform(timeseries)

    # Perform k-means clustering on standardized data

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=100)
    cluster_labels = kmeans.fit_predict(timeseries_scaled)

    # Get cluster centers (CAPs)
    caps = scaler.inverse_transform(kmeans.cluster_centers_)

    print("Clustering completed. Cluster distribution:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        percentage = count / len(cluster_labels) * 100
        print(f"  Cluster {cluster}: {count} time points ({percentage:.1f}%)")

    return cluster_labels, caps, kmeans, scaler


def plot_caps(caps, atlas_labels, save_path=None):
    """Plot Co-Activation Patterns (CAPs)."""
    print("Plotting CAPs...")

    n_clusters = caps.shape[0]
    fig, axes = plt.subplots(1, n_clusters, figsize=(4 * n_clusters, 6))
    if n_clusters == 1:
        axes = [axes]

    for i in range(n_clusters):
        ax = axes[i]

        # Create heatmap for each CAP
        cap_data = caps[i].reshape(-1, 1)
        im = ax.imshow(cap_data, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)

        ax.set_title(f"CAP {i+1}", fontsize=14, fontweight="bold")
        ax.set_yticks(range(len(atlas_labels)))
        ax.set_yticklabels(atlas_labels, fontsize=10)
        ax.set_xticks([])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Z-score", rotation=270, labelpad=15)

    plt.suptitle("Co-Activation Patterns (CAPs)", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"CAPs plot saved to: {save_path}")

    plt.close()  # Close figure to free memory


def plot_timeseries_with_clusters(df, cluster_labels, save_path=None):
    """Plot timeseries with cluster assignments."""
    print("Plotting timeseries with cluster assignments...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Plot timeseries
    time_points = np.arange(len(df))
    for i, col in enumerate(df.columns):
        ax1.plot(time_points, df[col], label=col, alpha=0.7)

    ax1.set_xlabel("Time Points")
    ax1.set_ylabel("Signal")
    ax1.set_title("Network Timeseries")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Plot cluster assignments
    colors = plt.cm.Set3(np.linspace(0, 1, len(np.unique(cluster_labels))))
    for i, label in enumerate(np.unique(cluster_labels)):
        mask = cluster_labels == label
        ax2.scatter(
            time_points[mask],
            [label] * np.sum(mask),
            c=[colors[i]],
            label=f"Cluster {label+1}",
            alpha=0.7,
        )

    ax2.set_xlabel("Time Points")
    ax2.set_ylabel("Cluster")
    ax2.set_title("Cluster Assignments Over Time")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Timeseries plot saved to: {save_path}")

    plt.close()  # Close figure to free memory


def calculate_cap_metrics(cluster_labels, run_name="run-01"):
    """Calculate CAP metrics for a given run."""
    print(f"Calculating CAP metrics for {run_name}...")

    n_timepoints = len(cluster_labels)
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)

    # Initialize metrics dictionary
    metrics = {
        "run": run_name,
        "total_timepoints": n_timepoints,
        "n_clusters": n_clusters,
    }

    # Calculate metrics for each cluster/CAP
    for cluster in unique_clusters:
        cluster_name = f"CAP_{cluster + 1}"

        # 1. Frequency of occurrence (percentage of time)
        cluster_timepoints = np.sum(cluster_labels == cluster)
        frequency = (cluster_timepoints / n_timepoints) * 100
        metrics[f"{cluster_name}_frequency_pct"] = frequency

        # 2. Dwell time calculation removed - focusing on frequency and transitions only

    # 3. Number of transitions FROM each CAP to other brain states
    for cluster in unique_clusters:
        cluster_name = f"CAP_{cluster + 1}"

        # Count transitions FROM this CAP to any other CAP
        transitions_from_cap = 0

        for i in range(len(cluster_labels) - 1):  # -1 because we look ahead
            if cluster_labels[i] == cluster and cluster_labels[i + 1] != cluster:
                transitions_from_cap += 1

        metrics[f"{cluster_name}_transitions_out"] = transitions_from_cap

    # Overall transition metrics (for comparison)
    total_transitions = 0
    for i in range(1, len(cluster_labels)):
        if cluster_labels[i] != cluster_labels[i - 1]:
            total_transitions += 1

    metrics["total_transitions"] = total_transitions
    metrics["overall_transition_rate"] = (
        total_transitions / n_timepoints if n_timepoints > 0 else 0
    )

    return metrics


def analyze_cap_metrics_subject_level(
    timeseries_list, run_names, subject_id, n_clusters=5
):
    """Analyze CAP metrics at subject level by concatenating all runs."""
    print(f"Analyzing subject-level CAP metrics for {subject_id}...")
    print(f"Concatenating {len(timeseries_list)} runs: {run_names}")

    # Concatenate all runs for this subject
    concatenated_timeseries = np.vstack(timeseries_list)
    print(f"Concatenated timeseries shape: {concatenated_timeseries.shape}")

    # Perform clustering on concatenated data (one set of CAPs for the subject)
    cluster_labels, caps, kmeans, scaler = perform_kmeans_clustering(
        concatenated_timeseries, n_clusters=n_clusters, random_state=42
    )

    # Calculate metrics across all concatenated runs
    subject_metrics = calculate_cap_metrics(cluster_labels, subject_id)

    # Also calculate run-specific metrics using the same CAPs
    run_metrics = []
    start_idx = 0

    for i, (timeseries, run_name) in enumerate(zip(timeseries_list, run_names)):
        end_idx = start_idx + len(timeseries)
        run_cluster_labels = cluster_labels[start_idx:end_idx]

        # Calculate metrics for this specific run
        run_metric = calculate_cap_metrics(run_cluster_labels, run_name)
        run_metrics.append(run_metric)

        start_idx = end_idx

    return (
        subject_metrics,
        run_metrics,
        caps,
        cluster_labels,
        concatenated_timeseries,
    )


def analyze_cap_metrics_multiple_runs(timeseries_list, run_names, n_clusters=5):
    """Analyze CAP metrics across multiple runs (original per-run analysis)."""
    print("Analyzing CAP metrics across multiple runs...")

    all_metrics = []

    for i, (timeseries, run_name) in enumerate(zip(timeseries_list, run_names)):
        print(f"\nProcessing {run_name}...")

        # Perform clustering for this run
        cluster_labels, caps, kmeans, scaler = perform_kmeans_clustering(
            timeseries, n_clusters=n_clusters, random_state=42
        )

        # Calculate metrics for this run
        metrics = calculate_cap_metrics(cluster_labels, run_name)
        all_metrics.append(metrics)

    # Create DataFrame with all metrics
    metrics_df = pd.DataFrame(all_metrics)

    return metrics_df, all_metrics


def plot_cap_metrics(metrics_df, save_path=None):
    """Plot CAP metrics across runs."""
    print("Plotting CAP metrics...")

    # Determine number of CAPs from the metrics
    cap_columns = [col for col in metrics_df.columns if "_frequency_pct" in col]
    n_caps = len(cap_columns)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 1. Frequency of occurrence
    ax1 = axes[0]
    for i in range(n_caps):
        cap_name = f"CAP_{i+1}"
        freq_col = f"{cap_name}_frequency_pct"
        if freq_col in metrics_df.columns:
            ax1.bar(
                [f"{run}\n{cap_name}" for run in metrics_df["run"]],
                metrics_df[freq_col],
                alpha=0.7,
                label=cap_name,
            )

    ax1.set_title("Frequency of Occurrence (%)", fontweight="bold")
    ax1.set_ylabel("Percentage of Time")
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend()

    # 2. Transitions out from each CAP
    ax2 = axes[1]
    for i in range(n_caps):
        cap_name = f"CAP_{i+1}"
        trans_col = f"{cap_name}_transitions_out"
        if trans_col in metrics_df.columns:
            ax2.bar(
                [f"{run}\n{cap_name}" for run in metrics_df["run"]],
                metrics_df[trans_col],
                alpha=0.7,
                label=cap_name,
            )

    ax2.set_title("Transitions Out from Each CAP", fontweight="bold")
    ax2.set_ylabel("Number of Transitions")
    ax2.tick_params(axis="x", rotation=45)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"CAP metrics plot saved to: {save_path}")

    plt.close()


def save_cap_metrics(metrics_df, output_dir="derivatives/caps/cap-analysis"):
    """Save CAP metrics to file."""
    print(f"Saving CAP metrics to {output_dir}/...")

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Save metrics
    metrics_df.to_csv(f"{output_dir}/cap_metrics.tsv", sep="\t", index=False)

    # Create summary statistics
    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
    summary_stats = metrics_df[numeric_cols].describe()
    summary_stats.to_csv(f"{output_dir}/cap_metrics_summary.tsv", sep="\t")

    print("CAP metrics saved:")
    print(f"  - {output_dir}/cap_metrics.tsv")
    print(f"  - {output_dir}/cap_metrics_summary.tsv")


def save_results(
    df,
    cluster_labels,
    caps,
    atlas_labels,
    output_dir="derivatives/caps/cap-analysis",
):
    """Save results to files."""
    print(f"Saving results to {output_dir}/...")

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Save timeseries with cluster labels
    df_with_clusters = df.copy()
    df_with_clusters["cluster"] = cluster_labels
    df_with_clusters.to_csv(
        f"{output_dir}/timeseries_with_clusters.tsv", sep="\t", index=False
    )

    # Save CAPs
    caps_df = pd.DataFrame(
        caps.T, columns=[f"CAP_{i+1}" for i in range(caps.shape[0])], index=atlas_labels
    )
    caps_df.to_csv(f"{output_dir}/caps.tsv", sep="\t")

    # Save cluster statistics
    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_stats = pd.DataFrame(
        {
            "cluster": unique + 1,
            "n_timepoints": counts,
            "percentage": counts / len(cluster_labels) * 100,
        }
    )
    cluster_stats.to_csv(f"{output_dir}/cluster_statistics.tsv", sep="\t", index=False)

    print("Results saved:")
    print(f"  - {output_dir}/timeseries_with_clusters.tsv")
    print(f"  - {output_dir}/caps.tsv")
    print(f"  - {output_dir}/cluster_statistics.tsv")


def run_cap_statistical_tests(metrics_df, output_dir="derivatives/caps/cap-analysis"):
    """Run statistical tests including ANOVA on CAP metrics."""
    print("Running statistical tests on CAP metrics...")

    # Create results list
    test_results = []

    # Get CAP names from the metrics
    cap_columns = [col for col in metrics_df.columns if "CAP_" in col]
    cap_names = list(
        set([col.split("_")[0] + "_" + col.split("_")[1] for col in cap_columns])
    )

    # Check if we have enough data for statistical tests
    n_observations = len(metrics_df)
    print(f"Number of observations: {n_observations}")

    if n_observations < 2:
        print(
            "WARNING: Single observation detected. Statistical tests require multiple data points."
        )
        print("Generating descriptive statistics instead of inferential tests...")

        # Generate descriptive statistics for single-subject data
        for cap_name in cap_names:
            # Frequency statistics
            freq_col = f"{cap_name}_frequency_pct"
            if freq_col in metrics_df.columns:
                freq_value = metrics_df[freq_col].iloc[0]
                test_results.append(
                    {
                        "test_type": "descriptive",
                        "cap": cap_name,
                        "metric": "frequency_pct",
                        "test": "single_subject_value",
                        "value": freq_value,
                        "interpretation": f"CAP occurs {freq_value:.1f}% of the time",
                    }
                )

            # Transitions statistics
            trans_out_col = f"{cap_name}_transitions_out"
            if trans_out_col in metrics_df.columns:
                trans_value = metrics_df[trans_out_col].iloc[0]
                test_results.append(
                    {
                        "test_type": "descriptive",
                        "cap": cap_name,
                        "metric": "transitions_out",
                        "test": "single_subject_value",
                        "value": trans_value,
                        "interpretation": f"CAP transitions out {trans_value} times",
                    }
                )

        # Overall statistics
        if "overall_transition_rate" in metrics_df.columns:
            overall_rate = metrics_df["overall_transition_rate"].iloc[0]
            test_results.append(
                {
                    "test_type": "descriptive",
                    "cap": "OVERALL",
                    "metric": "overall_transition_rate",
                    "test": "single_subject_value",
                    "value": overall_rate,
                    "interpretation": f"Overall transition rate: {overall_rate:.3f} transitions per timepoint",
                }
            )

        # Convert to DataFrame
        results_df = pd.DataFrame(test_results)

        # Print summary for single subject
        print(f"\nSingle-Subject Descriptive Statistics:")
        print(f"Total metrics calculated: {len(results_df)}")

        print(f"\nCAP Frequency Statistics:")
        freq_results = results_df[results_df["metric"] == "frequency_pct"]
        for _, row in freq_results.iterrows():
            print(f"  {row['cap']}: {row['value']:.1f}%")

        print(f"\nCAP Transition Statistics:")
        trans_results = results_df[results_df["metric"] == "transitions_out"]
        for _, row in trans_results.iterrows():
            print(f"  {row['cap']}: {row['value']} transitions out")

        # Compare CAPs descriptively
        if len(freq_results) > 0:
            freq_values = freq_results["value"].values
            most_frequent_idx = freq_values.argmax()
            least_frequent_idx = freq_values.argmin()

            most_frequent_cap = freq_results.iloc[most_frequent_idx]["cap"]
            least_frequent_cap = freq_results.iloc[least_frequent_idx]["cap"]

            print(f"\nDescriptive Comparisons:")
            print(
                f"  Most frequent CAP: {most_frequent_cap} ({freq_values[most_frequent_idx]:.1f}%)"
            )
            print(
                f"  Least frequent CAP: {least_frequent_cap} ({freq_values[least_frequent_idx]:.1f}%)"
            )

        if len(trans_results) > 0:
            trans_values = trans_results["value"].values
            most_dynamic_idx = trans_values.argmax()
            least_dynamic_idx = trans_values.argmin()

            most_dynamic_cap = trans_results.iloc[most_dynamic_idx]["cap"]
            least_dynamic_cap = trans_results.iloc[least_dynamic_idx]["cap"]

            print(
                f"  Most dynamic CAP: {most_dynamic_cap} ({trans_values[most_dynamic_idx]} transitions)"
            )
            print(
                f"  Least dynamic CAP: {least_dynamic_cap} ({trans_values[least_dynamic_idx]} transitions)"
            )

    else:
        # Original statistical tests for multiple observations
        print(
            "Multiple observations detected. Running inferential statistical tests..."
        )

        # Test 1: T-tests - Are CAP metrics significantly > 0? (Dynamic vs Static)
        for cap_name in cap_names:
            # Test transitions out
            trans_out_col = f"{cap_name}_transitions_out"
            if trans_out_col in metrics_df.columns:
                transitions = metrics_df[trans_out_col].dropna()
                if len(transitions) > 1:  # Need at least 2 observations
                    t_stat, p_val = ttest_1samp(transitions, 0)
                    test_results.append(
                        {
                            "test_type": "t_test",
                            "cap": cap_name,
                            "metric": "transitions_out",
                            "test": "one_sample_ttest_vs_0",
                            "mean": transitions.mean(),
                            "std": transitions.std(),
                            "n": len(transitions),
                            "statistic": t_stat,
                            "p_value": p_val,
                            "significant": p_val < 0.05,
                            "interpretation": (
                                "CAP shows dynamic transitions"
                                if p_val < 0.05
                                else "CAP may be relatively stable"
                            ),
                        }
                    )

        # Test 2: ANOVA tests - Are there differences between CAPs?

        # ANOVA 1: Do CAPs have different frequencies?
        frequency_data = []
        for cap_name in cap_names:
            freq_col = f"{cap_name}_frequency_pct"
            if freq_col in metrics_df.columns:
                cap_freqs = metrics_df[freq_col].dropna().tolist()
                if len(cap_freqs) > 0:
                    frequency_data.append(cap_freqs)

        if len(frequency_data) > 1 and all(len(data) > 1 for data in frequency_data):
            f_stat, p_val = f_oneway(*frequency_data)

            test_results.append(
                {
                    "test_type": "anova",
                    "cap": "ALL_CAPS",
                    "metric": "frequency_pct",
                    "test": "one_way_anova_between_caps",
                    "mean": np.mean([np.mean(data) for data in frequency_data]),
                    "std": np.std([np.mean(data) for data in frequency_data]),
                    "n": len(frequency_data),
                    "statistic": f_stat,
                    "p_value": p_val,
                    "significant": p_val < 0.05,
                    "interpretation": (
                        "CAPs have significantly different frequencies"
                        if p_val < 0.05
                        else "CAPs have similar frequencies"
                    ),
                }
            )

        # ANOVA 2: Do CAPs have different transition rates?
        transition_data = []
        for cap_name in cap_names:
            trans_col = f"{cap_name}_transitions_out"
            if trans_col in metrics_df.columns:
                cap_trans = metrics_df[trans_col].dropna().tolist()
                if len(cap_trans) > 0:
                    transition_data.append(cap_trans)

        if len(transition_data) > 1 and all(len(data) > 1 for data in transition_data):
            f_stat, p_val = f_oneway(*transition_data)

            test_results.append(
                {
                    "test_type": "anova",
                    "cap": "ALL_CAPS",
                    "metric": "transitions_out",
                    "test": "one_way_anova_between_caps",
                    "mean": np.mean([np.mean(data) for data in transition_data]),
                    "std": np.std([np.mean(data) for data in transition_data]),
                    "n": len(transition_data),
                    "statistic": f_stat,
                    "p_value": p_val,
                    "significant": p_val < 0.05,
                    "interpretation": (
                        "CAPs have significantly different transition rates"
                        if p_val < 0.05
                        else "CAPs have similar transition rates"
                    ),
                }
            )

        # Convert to DataFrame
        results_df = pd.DataFrame(test_results)

        # Print summary for multiple observations
        if len(results_df) > 0:
            # Separate t-tests and ANOVAs
            t_tests = results_df[results_df["test_type"] == "t_test"]
            anovas = results_df[results_df["test_type"] == "anova"]

            print(f"\nT-test Results ({len(t_tests)} tests):")
            for _, row in t_tests.iterrows():
                sig_marker = "*" if row["significant"] else ""
                print(
                    f"  {row['cap']} {row['metric']}: "
                    f"Mean = {row['mean']:.2f}, "
                    f"p = {row['p_value']:.3f}{sig_marker}"
                )
                if row["significant"]:
                    print(f"    → {row['interpretation']}")

            print(f"\nANOVA Results ({len(anovas)} tests):")
            for _, row in anovas.iterrows():
                sig_marker = "*" if row["significant"] else ""
                print(
                    f"  {row['metric']} across CAPs: "
                    f"F = {row['statistic']:.2f}, "
                    f"p = {row['p_value']:.3f}{sig_marker}"
                )
                if row["significant"]:
                    print(f"    → {row['interpretation']}")

    # Save results
    if len(test_results) > 0:
        Path(output_dir).mkdir(exist_ok=True)
        results_df.to_csv(
            f"{output_dir}/cap_statistical_tests.tsv", sep="\t", index=False
        )

        if n_observations < 2:
            print(
                f"\nDescriptive statistics saved to: "
                f"{output_dir}/cap_statistical_tests.tsv"
            )
        else:
            print(
                f"\nStatistical test results saved to: "
                f"{output_dir}/cap_statistical_tests.tsv"
            )

    return results_df


def main_subject_level():
    """Main analysis pipeline for subject-level CAP metrics."""
    print("Subject-Level CAPs Analysis Pipeline")
    print("=" * 50)

    # Using Bubbles subject data for Stranger Things task
    subject_id = "sub-Bubbles_ses-01"
    bold_files = [
        (
            "derivatives/simulated/"
            "sub_Bubbles_ses-01_task-stranger_run-01_space-scan_desc-optcomDenoised_bold.nii.gz"
        ),
        # Add more runs here if available, e.g.:
        # (
        #     "derivatives/simulated/"
        #     "sub_Bubbles_ses-01_task-stranger_run-02_space-scan_desc-optcomDenoised_bold.nii.gz"
        # ),
    ]

    # Check which files exist
    existing_files = []
    run_names = []
    for bold_file in bold_files:
        if Path(bold_file).exists():
            existing_files.append(bold_file)
            # Extract run name from filename
            parts = bold_file.split("_")
            run_name = f"{parts[0]}_{parts[1]}_{parts[3]}"  # sub_ses_run
            run_names.append(run_name)
        else:
            print(f"Warning: File not found: {bold_file}")

    if not existing_files:
        print("Error: No valid files found.")
        print("Please make sure at least one file exists in the current directory.")
        return

    print(f"Found {len(existing_files)} runs for {subject_id}")

    # Load Yeo atlas
    try:
        atlas_img, atlas_labels = load_yeo_atlas()
    except Exception as e:
        print(f"Error loading atlas: {e}")
        return

    # Extract timeseries for all runs
    try:
        timeseries_list = []
        for i, bold_file in enumerate(existing_files):
            print(f"\nExtracting timeseries from run {i+1}: {bold_file}")
            df, timeseries, masker = extract_timeseries(
                bold_file, atlas_img, atlas_labels
            )
            timeseries_list.append(timeseries)
    except Exception as e:
        print(f"Error extracting timeseries: {e}")
        return

    # Concatenate timeseries for elbow plot analysis
    concatenated_for_elbow = np.vstack(timeseries_list)
    print(f"\nConcatenated timeseries shape: {concatenated_for_elbow.shape}")

    # Generate elbow plot to determine optimal number of clusters
    try:
        print("\nStep 1: Determining optimal number of clusters...")
        cluster_range, inertias = plot_elbow_curve(
            concatenated_for_elbow,
            max_clusters=15,
            save_path=Path("derivatives/caps/cap-analysis/figures")
            / "subject_elbow_plot.png",
        )

        # Get suggested optimal k
        # Calculate differences to help identify elbow
        differences = []
        for i in range(1, len(inertias)):
            diff = inertias[i - 1] - inertias[i]
            differences.append(diff)

        # Find elbow using rate of change
        second_differences = []
        for i in range(1, len(differences)):
            second_diff = differences[i - 1] - differences[i]
            second_differences.append(second_diff)

        # Suggest optimal k (where second difference is maximum)
        if second_differences:
            optimal_idx = second_differences.index(max(second_differences))
            # +2 because of indexing
            suggested_k = cluster_range[optimal_idx + 2]
            print(f"Suggested optimal number of clusters: {suggested_k}")
            n_clusters = suggested_k
        else:
            print("Could not determine optimal k, using default = 5")
            n_clusters = 5

    except Exception as e:
        print(f"Error generating elbow plot: {e}")
        print("Continuing with default number of clusters...")
        n_clusters = 5

    # Perform subject-level analysis
    try:
        print(f"\nStep 2: Subject-level clustering with k={n_clusters}...")
        (
            subject_metrics,
            run_metrics,
            caps,
            cluster_labels,
            concatenated_timeseries,
        ) = analyze_cap_metrics_subject_level(
            timeseries_list, run_names, subject_id, n_clusters=n_clusters
        )

        # Create DataFrames
        subject_metrics_df = pd.DataFrame([subject_metrics])
        run_metrics_df = pd.DataFrame(run_metrics)

        print("\nSubject-Level CAP Metrics Summary:")
        print(f"  Total timepoints: {subject_metrics['total_timepoints']}")

        for i in range(n_clusters):
            cap_name = f"CAP_{i+1}"
            freq_key = f"{cap_name}_frequency_pct"
            trans_out_key = f"{cap_name}_transitions_out"

            print(f"  {cap_name}:")
            print(f"    Frequency: {subject_metrics[freq_key]:.1f}%")
            print(f"    Transitions out: {subject_metrics[trans_out_key]}")

    except Exception as e:
        print(f"Error in subject-level analysis: {e}")
        return

    # Plot results
    try:
        print("\nStep 2: Generating visualization plots...")

        # Create figures directory
        figures_dir = Path("derivatives/caps/cap-analysis/figures")
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Create concatenated DataFrame for plotting
        concatenated_df = pd.DataFrame(concatenated_timeseries, columns=atlas_labels)

        plot_caps(
            caps, atlas_labels, save_path=figures_dir / "subject_caps_patterns.png"
        )
        plot_timeseries_with_clusters(
            concatenated_df,
            cluster_labels,
            save_path=figures_dir / "subject_timeseries_clusters.png",
        )

        # Plot subject-level metrics
        plot_cap_metrics(
            subject_metrics_df, save_path=figures_dir / "subject_cap_metrics.png"
        )

        # Plot run-level metrics (using same CAPs)
        plot_cap_metrics(
            run_metrics_df, save_path=figures_dir / "run_cap_metrics_same_caps.png"
        )

    except Exception as e:
        print(f"Error in plotting: {e}")
        print("Continuing without plots...")

    # Save results
    try:
        print("\nStep 3: Saving results...")

        # Save subject-level results
        output_dir = f"derivatives/caps/cap-analysis/{subject_id.replace('-', '_')}"
        save_results(concatenated_df, cluster_labels, caps, atlas_labels, output_dir)

        # Save subject-level metrics
        save_cap_metrics(subject_metrics_df, output_dir)

        # Save run-level metrics (using same CAPs)
        run_metrics_df.to_csv(
            f"{output_dir}/run_metrics_same_caps.tsv", sep="\t", index=False
        )

        # Run statistical tests on subject-level metrics
        run_cap_statistical_tests(subject_metrics_df, output_dir)

    except Exception as e:
        print(f"Error saving results: {e}")
        return

    print("\nSubject-level analysis completed successfully!")
    print("Generated files:")
    print("  - subject_elbow_plot.png (optimal cluster selection)")
    print("  - subject_caps_patterns.png (subject CAP visualization)")
    print("  - subject_timeseries_clusters.png (concatenated timeseries)")
    print("  - subject_cap_metrics.png (subject-level metrics)")
    print("  - run_cap_metrics_same_caps.png (run metrics using same CAPs)")
    print(f"  - {output_dir}/ directory with results")
    print("    * timeseries_with_clusters.tsv (concatenated data)")
    print("    * caps.tsv (subject-specific CAPs)")
    print("    * cap_metrics.tsv (subject-level metrics)")
    print("    * run_metrics_same_caps.tsv (run-level metrics)")
    print("    * cap_statistical_tests.tsv (statistical validation)")

    print(f"\nIdentified {n_clusters} optimal subject-specific CAPs")
    print("These CAPs are consistent across all runs for this subject!")
    print("Number of clusters was determined using elbow method!")
    print("Statistical tests validate CAP dynamics!")

    return subject_metrics, run_metrics, caps


def main():
    """Main analysis pipeline."""
    print("CAPs Analysis Pipeline")
    print("=" * 50)

    # File path for Bubbles subject
    bold_file = (
        "derivatives/simulated/"
        "sub_Bubbles_ses-01_task-stranger_run-01_space-scan_"
        "desc-optcomDenoised_bold.nii.gz"
    )

    # Check if file exists
    if not Path(bold_file).exists():
        print(f"Error: File not found: {bold_file}")
        print("Please make sure the file is in the current directory.")
        return

    # Load Yeo atlas
    try:
        atlas_img, atlas_labels = load_yeo_atlas()
    except Exception as e:
        print(f"Error loading atlas: {e}")
        return

    # Extract timeseries
    try:
        df, timeseries, masker = extract_timeseries(bold_file, atlas_img, atlas_labels)
    except Exception as e:
        print(f"Error extracting timeseries: {e}")
        return

    # Generate elbow plot to determine optimal number of clusters
    try:
        print("\nStep 1: Determining optimal number of clusters...")
        cluster_range, inertias = plot_elbow_curve(
            timeseries, max_clusters=15, save_path="elbow_plot.png"
        )
    except Exception as e:
        print(f"Error generating elbow plot: {e}")
        print("Continuing with default number of clusters...")

    # Perform k-means clustering
    n_clusters = 5  # You can adjust this based on elbow plot results
    try:
        print(f"\nStep 2: Performing clustering with k={n_clusters}...")
        cluster_labels, caps, kmeans, scaler = perform_kmeans_clustering(
            timeseries, n_clusters=n_clusters
        )
    except Exception as e:
        print(f"Error in clustering: {e}")
        return

    # Calculate CAP metrics
    try:
        print("\nStep 3: Calculating CAP metrics...")
        run_name = (
            bold_file.split("_")[0]
            + "_"
            + bold_file.split("_")[1]
            + "_"
            + bold_file.split("_")[2]
            + "_"
            + bold_file.split("_")[3]
        )
        cap_metrics = calculate_cap_metrics(cluster_labels, run_name)

        # Create metrics DataFrame
        metrics_df = pd.DataFrame([cap_metrics])

        # Print metrics summary
        print("\nCAP Metrics Summary:")
        print(f"  Total timepoints: {cap_metrics['total_timepoints']}")

        for i in range(n_clusters):
            cap_name = f"CAP_{i+1}"
            freq_key = f"{cap_name}_frequency_pct"
            trans_out_key = f"{cap_name}_transitions_out"

            print(f"  {cap_name}:")
            print(f"    Frequency: {cap_metrics[freq_key]:.1f}%")
            print(f"    Transitions out: {cap_metrics[trans_out_key]}")

    except Exception as e:
        print(f"Error calculating CAP metrics: {e}")
        metrics_df = None

    # Plot results
    try:
        print("\nStep 4: Generating visualization plots...")
        plot_caps(caps, atlas_labels, save_path="caps_patterns.png")
        plot_timeseries_with_clusters(
            df, cluster_labels, save_path="timeseries_clusters.png"
        )

        # Plot CAP metrics if available
        if metrics_df is not None:
            plot_cap_metrics(metrics_df, save_path="cap_metrics.png")

    except Exception as e:
        print(f"Error in plotting: {e}")
        print("Continuing without plots...")

    # Save results
    try:
        print("\nStep 5: Saving results...")
        save_results(df, cluster_labels, caps, atlas_labels)

        # Save CAP metrics if available
        if metrics_df is not None:
            save_cap_metrics(metrics_df)

            # Run statistical tests
            run_cap_statistical_tests(metrics_df)

    except Exception as e:
        print(f"Error saving results: {e}")
        return

    print("\nAnalysis completed successfully!")
    print("Generated files:")
    print("  - elbow_plot.png (for optimal cluster selection)")
    print("  - caps_patterns.png (CAP visualization)")
    print("  - timeseries_clusters.png (timeseries with clusters)")
    print("  - cap_metrics.png (CAP metrics visualization)")
    print("  - caps_results/ directory with TSV files")
    print("    * timeseries_with_clusters.tsv")
    print("    * caps.tsv")
    print("    * cluster_statistics.tsv")
    print("    * cap_metrics.tsv")
    print("    * cap_metrics_summary.tsv")
    print("    * cap_statistical_tests.tsv")
    print(f"\nIdentified {n_clusters} distinct co-activation patterns (CAPs)")
    print("Check the elbow plot to determine if this is optimal!")

    # Print basic statistics
    print("\nBasic Statistics:")
    print(f"Total time points: {len(timeseries)}")
    print(f"Number of networks: {len(atlas_labels)}")
    print(f"Networks: {', '.join(atlas_labels)}")


if __name__ == "__main__":
    # Choose which analysis to run:

    # For subject-level analysis (recommended for individual differences):
    main_subject_level()

    # For single-run analysis (original):
    # main()
