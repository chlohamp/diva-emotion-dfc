import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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
    # This is NOT Fisher r-to-z transformation
    # For each network: z = (x - mean) / std across time
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

        # 2. Dwell time (average consecutive TRs in this state)
        dwell_times = []
        current_dwell = 0
        in_cluster = False

        for i, label in enumerate(cluster_labels):
            if label == cluster:
                if not in_cluster:
                    # Starting a new dwell period
                    in_cluster = True
                    current_dwell = 1
                else:
                    # Continuing current dwell period
                    current_dwell += 1
            else:
                if in_cluster:
                    # Ending current dwell period
                    dwell_times.append(current_dwell)
                    in_cluster = False
                    current_dwell = 0

        # Don't forget the last dwell period if it ends at the last timepoint
        if in_cluster:
            dwell_times.append(current_dwell)

        # Calculate average dwell time
        avg_dwell_time = np.mean(dwell_times) if dwell_times else 0
        metrics[f"{cluster_name}_avg_dwell_time"] = avg_dwell_time
        metrics[f"{cluster_name}_n_episodes"] = len(dwell_times)

    # 3. Number of transitions FROM each CAP to other brain states
    for cluster in unique_clusters:
        cluster_name = f"CAP_{cluster + 1}"

        # Count transitions FROM this CAP to any other CAP
        transitions_from_cap = 0

        for i in range(len(cluster_labels) - 1):  # -1 because we look ahead
            if cluster_labels[i] == cluster and cluster_labels[i + 1] != cluster:
                transitions_from_cap += 1

        metrics[f"{cluster_name}_transitions_out"] = transitions_from_cap

        # Calculate transition rate for this CAP (transitions per episode)
        n_episodes = metrics[f"{cluster_name}_n_episodes"]
        if n_episodes > 0:
            metrics[f"{cluster_name}_transition_rate"] = (
                transitions_from_cap / n_episodes
            )
        else:
            metrics[f"{cluster_name}_transition_rate"] = 0

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


def analyze_cap_metrics_multiple_runs(timeseries_list, run_names, n_clusters=5):
    """Analyze CAP metrics across multiple runs."""
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

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Frequency of occurrence
    ax1 = axes[0, 0]
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

    # 2. Average dwell time
    ax2 = axes[0, 1]
    for i in range(n_caps):
        cap_name = f"CAP_{i+1}"
        dwell_col = f"{cap_name}_avg_dwell_time"
        if dwell_col in metrics_df.columns:
            ax2.bar(
                [f"{run}\n{cap_name}" for run in metrics_df["run"]],
                metrics_df[dwell_col],
                alpha=0.7,
                label=cap_name,
            )

    ax2.set_title("Average Dwell Time (TRs)", fontweight="bold")
    ax2.set_ylabel("Time Points (TRs)")
    ax2.tick_params(axis="x", rotation=45)
    ax2.legend()

    # 3. Transitions out from each CAP
    ax3 = axes[1, 0]
    for i in range(n_caps):
        cap_name = f"CAP_{i+1}"
        trans_col = f"{cap_name}_transitions_out"
        if trans_col in metrics_df.columns:
            ax3.bar(
                [f"{run}\n{cap_name}" for run in metrics_df["run"]],
                metrics_df[trans_col],
                alpha=0.7,
                label=cap_name,
            )

    ax3.set_title("Transitions Out from Each CAP", fontweight="bold")
    ax3.set_ylabel("Number of Transitions")
    ax3.tick_params(axis="x", rotation=45)
    ax3.legend()

    # 4. CAP transition rates
    ax4 = axes[1, 1]
    for i in range(n_caps):
        cap_name = f"CAP_{i+1}"
        rate_col = f"{cap_name}_transition_rate"
        if rate_col in metrics_df.columns:
            ax4.bar(
                [f"{run}\n{cap_name}" for run in metrics_df["run"]],
                metrics_df[rate_col],
                alpha=0.7,
                label=cap_name,
            )

    ax4.set_title("CAP Transition Rates (transitions/episode)", fontweight="bold")
    ax4.set_ylabel("Transitions per Episode")
    ax4.tick_params(axis="x", rotation=45)
    ax4.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"CAP metrics plot saved to: {save_path}")

    plt.close()


def save_cap_metrics(metrics_df, output_dir="caps_results"):
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


def save_results(df, cluster_labels, caps, atlas_labels, output_dir="caps_results"):
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


def main():
    """Main analysis pipeline."""
    print("CAPs Analysis Pipeline")
    print("=" * 50)

    # File path
    bold_file = (
        "sub-pretend_ses-01_task-rest_run-01_space-scan_desc-optcomDenoised_bold.nii.gz"
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
            dwell_key = f"{cap_name}_avg_dwell_time"
            episodes_key = f"{cap_name}_n_episodes"
            trans_out_key = f"{cap_name}_transitions_out"
            trans_rate_key = f"{cap_name}_transition_rate"

            print(f"  {cap_name}:")
            print(f"    Frequency: {cap_metrics[freq_key]:.1f}%")
            print(f"    Avg Dwell Time: {cap_metrics[dwell_key]:.2f} TRs")
            print(f"    Episodes: {cap_metrics[episodes_key]}")
            print(f"    Transitions out: {cap_metrics[trans_out_key]}")
            print(f"    Transition rate: {cap_metrics[trans_rate_key]:.3f} /episode")

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
    print(f"\nIdentified {n_clusters} distinct co-activation patterns (CAPs)")
    print("Check the elbow plot to determine if this is optimal!")

    # Print basic statistics
    print("\nBasic Statistics:")
    print(f"Total time points: {len(timeseries)}")
    print(f"Number of networks: {len(atlas_labels)}")
    print(f"Networks: {', '.join(atlas_labels)}")


if __name__ == "__main__":
    main()
