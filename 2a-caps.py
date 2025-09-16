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


def perform_kmeans_clustering(timeseries, n_clusters=5, random_state=42):
    """Perform k-means clustering on timeseries data."""
    print(f"Performing k-means clustering with {n_clusters} clusters...")

    # Standardize the data
    # Standardizes each network's timeseries individually (temporal standardization)
    scaler = StandardScaler()
    timeseries_scaled = scaler.fit_transform(timeseries)

    # Perform k-means clustering
    # Standardizes across all networks at each timepoint (spatial standardization)

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

    # Perform k-means clustering
    n_clusters = 5  # You can adjust this
    try:
        cluster_labels, caps, kmeans, scaler = perform_kmeans_clustering(
            timeseries, n_clusters=n_clusters
        )
    except Exception as e:
        print(f"Error in clustering: {e}")
        return

    # Plot results
    try:
        plot_caps(caps, atlas_labels, save_path="caps_patterns.png")
        plot_timeseries_with_clusters(
            df, cluster_labels, save_path="timeseries_clusters.png"
        )
    except Exception as e:
        print(f"Error in plotting: {e}")
        print("Continuing without plots...")

    # Save results
    try:
        save_results(df, cluster_labels, caps, atlas_labels)
    except Exception as e:
        print(f"Error saving results: {e}")
        return

    print("\nAnalysis completed successfully!")
    print(f"Identified {n_clusters} distinct co-activation patterns (CAPs)")

    # Print basic statistics
    print("\nBasic Statistics:")
    print(f"Total time points: {len(timeseries)}")
    print(f"Number of networks: {len(atlas_labels)}")
    print(f"Networks: {', '.join(atlas_labels)}")


if __name__ == "__main__":
    main()
