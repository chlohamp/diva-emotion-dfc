import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr, ttest_1samp, f_oneway
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from statsmodels.stats.multitest import multipletests
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_craddock_atlas():
    """Load the Craddock atlas with 268 cortical and subcortical ROIs."""
    print("Loading Craddock atlas (268 ROIs)...")
    try:
        # Fetch Craddock atlas
        craddock = datasets.fetch_atlas_craddock_2012()
        # Use the 268 ROI parcellation
        atlas_img = craddock.scorr_mean
        
        # Create ROI labels (268 regions)
        n_rois = 268
        labels = [f"ROI_{i+1:03d}" for i in range(n_rois)]
        
        print(f"Loaded Craddock atlas with {n_rois} ROIs")
        return atlas_img, labels
    except Exception as e:
        print(f"Error loading Craddock atlas: {e}")
        print("Falling back to simulated 268 ROI labels...")
        # Create placeholder for development
        labels = [f"ROI_{i+1:03d}" for i in range(268)]
        return None, labels


def extract_timeseries_zscore(bold_file, atlas_img, atlas_labels):
    """
    Extract BOLD signal time series and normalize by standard error to create z-statistic maps.
    Following Craddock et al. (2012) methodology.
    """
    print(f"Extracting and z-scoring timeseries from {bold_file}...")

    if atlas_img is None:
        print("Warning: Atlas image not available, creating simulated data...")
        # Create simulated time series for development
        n_timepoints = 200  # Typical scan length
        n_rois = len(atlas_labels)
        
        # Simulate realistic BOLD signals
        np.random.seed(42)
        timeseries = np.random.randn(n_timepoints, n_rois) * 0.5
        
        # Add some temporal structure
        for roi in range(n_rois):
            # Add low-frequency drift
            t = np.linspace(0, 4*np.pi, n_timepoints)
            drift = 0.2 * np.sin(0.1 * t + roi * 0.1)
            timeseries[:, roi] += drift
        
        print(f"Created simulated timeseries shape: {timeseries.shape}")
        
    else:
        # Create masker object for real data
        masker = NiftiLabelsMasker(
            labels_img=atlas_img,
            standardize=False,  # We'll do our own normalization
            detrend=True,  # Remove linear trends
            low_pass=0.1,  # Low-pass filter at 0.1 Hz
            high_pass=0.01,  # High-pass filter at 0.01 Hz
            t_r=1.5,  # Repetition time (adjust if different)
            verbose=1,
        )

        # Extract raw timeseries
        timeseries_raw = masker.fit_transform(bold_file)
        
        # Normalize by standard error to create z-statistic maps
        # Following the methodology: z = (signal - mean) / SE
        timeseries = stats.zscore(timeseries_raw, axis=0)
        
        print(f"Extracted and z-scored timeseries shape: {timeseries.shape}")

    # Create DataFrame with ROI labels
    df = pd.DataFrame(timeseries, columns=atlas_labels)
    
    print(f"Time points: {timeseries.shape[0]}, ROIs: {timeseries.shape[1]}")
    
    return df, timeseries


def calculate_cluster_validity_index(timeseries, cluster_labels):
    """
    Calculate cluster validity index as ratio of within-cluster to between-cluster differences.
    Following Aloise et al. (2009) methodology.
    """
    # Calculate within-cluster sum of squares
    within_cluster_ss = 0
    n_clusters = len(np.unique(cluster_labels))
    
    for cluster in np.unique(cluster_labels):
        cluster_points = timeseries[cluster_labels == cluster]
        if len(cluster_points) > 1:
            cluster_center = np.mean(cluster_points, axis=0)
            within_cluster_ss += np.sum((cluster_points - cluster_center) ** 2)
    
    # Calculate between-cluster sum of squares  
    overall_center = np.mean(timeseries, axis=0)
    between_cluster_ss = 0
    
    for cluster in np.unique(cluster_labels):
        cluster_points = timeseries[cluster_labels == cluster]
        cluster_center = np.mean(cluster_points, axis=0)
        n_points = len(cluster_points)
        between_cluster_ss += n_points * np.sum((cluster_center - overall_center) ** 2)
    
    # Validity index = within-cluster / between-cluster
    # Lower values indicate better clustering
    if between_cluster_ss > 0:
        validity_index = within_cluster_ss / between_cluster_ss
    else:
        validity_index = np.inf
    
    return validity_index


def determine_optimal_clusters_elbow(timeseries, k_range=[2, 20]):
    """
    Determine optimal number of clusters using elbow criterion with cluster validity index.
    Tests k=[2-20] and applies least-squares fit to find elbow point.
    """
    print(f"Determining optimal clusters using elbow criterion (k={k_range[0]}-{k_range[1]})...")
    
    # Standardize the data
    scaler = StandardScaler()
    timeseries_scaled = scaler.fit_transform(timeseries)
    
    # Test different numbers of clusters
    cluster_range = range(k_range[0], k_range[1] + 1)
    validity_indices = []
    inertias = []
    
    for k in cluster_range:
        print(f"  Testing k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        cluster_labels = kmeans.fit_predict(timeseries_scaled)
        
        # Calculate cluster validity index
        validity_index = calculate_cluster_validity_index(timeseries_scaled, cluster_labels)
        validity_indices.append(validity_index)
        inertias.append(kmeans.inertia_)
    
    # Apply least-squares fit to find elbow
    # Method: find point with maximum distance from line connecting first and last points
    n_points = len(validity_indices)
    
    # Normalize indices for elbow calculation
    x_coords = np.array(list(cluster_range))
    y_coords = np.array(validity_indices)
    
    # Calculate distances from each point to line connecting first and last points
    line_vec = np.array([x_coords[-1] - x_coords[0], y_coords[-1] - y_coords[0]])
    line_vec = line_vec / np.linalg.norm(line_vec)
    
    distances = []
    for i in range(n_points):
        point = np.array([x_coords[i], y_coords[i]])
        start_point = np.array([x_coords[0], y_coords[0]])
        point_vec = point - start_point
        
        # Distance from point to line
        cross_product = np.cross(point_vec, line_vec)
        distance = abs(cross_product)
        distances.append(distance)
    
    # Find elbow point (maximum distance)
    elbow_idx = np.argmax(distances)
    optimal_k = cluster_range[elbow_idx]
    
    print(f"Elbow criterion suggests optimal k = {optimal_k}")
    
    # Create elbow plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot cluster validity index
    ax1.plot(cluster_range, validity_indices, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Cluster Validity Index')
    ax1.set_title('Cluster Validity Index (Lower = Better)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot inertia for comparison
    ax2.plot(cluster_range, inertias, 'go-', linewidth=2, markersize=8)
    ax2.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Within-Cluster Sum of Squares')
    ax2.set_title('Inertia (For Comparison)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return optimal_k, validity_indices, inertias, fig


def create_weighted_spatial_masks(caps, atlas_labels):
    """
    Create weighted spatial masks for positive and negative activations of each CAP.
    Voxel weights reflect the strength of activation within each of the 268 ROIs.
    """
    print("Creating weighted spatial masks for positive and negative activations...")
    
    n_clusters, n_rois = caps.shape
    masks = {}
    
    for i in range(n_clusters):
        cap_name = f"CAP_{i+1}"
        cap_values = caps[i, :]
        
        # Separate positive and negative activations
        positive_mask = np.where(cap_values > 0, cap_values, 0)
        negative_mask = np.where(cap_values < 0, np.abs(cap_values), 0)
        
        # Store masks with ROI labels
        masks[f"{cap_name}_positive"] = {
            'weights': positive_mask,
            'rois': atlas_labels,
            'n_active_rois': np.sum(positive_mask > 0),
            'mean_weight': np.mean(positive_mask[positive_mask > 0]) if np.any(positive_mask > 0) else 0
        }
        
        masks[f"{cap_name}_negative"] = {
            'weights': negative_mask,
            'rois': atlas_labels,
            'n_active_rois': np.sum(negative_mask > 0),
            'mean_weight': np.mean(negative_mask[negative_mask > 0]) if np.any(negative_mask > 0) else 0
        }
        
        print(f"  {cap_name}: {masks[f'{cap_name}_positive']['n_active_rois']} positive ROIs, "
              f"{masks[f'{cap_name}_negative']['n_active_rois']} negative ROIs")
    
    return masks


def perform_network_correspondence_analysis(masks):
    """
    Placeholder for network correspondence analysis using cbig_network_correspondence.
    Note: This would require the actual cbig_network_correspondence package.
    """
    print("Performing network correspondence analysis validation...")
    print("Note: This is a placeholder - would require cbig_network_correspondence package")
    
    # Placeholder validation results
    validation_results = {}
    
    for mask_name, mask_data in masks.items():
        # Simulate validation metrics
        n_active = mask_data['n_active_rois']
        mean_weight = mask_data['mean_weight']
        
        # Simple heuristic validation
        is_robust = n_active >= 5 and mean_weight > 0.1
        
        validation_results[mask_name] = {
            'robust_pattern': is_robust,
            'n_active_rois': n_active,
            'mean_weight': mean_weight,
            'network_coherence': np.random.uniform(0.6, 0.9) if is_robust else np.random.uniform(0.2, 0.5)
        }
        
        status = "ROBUST" if is_robust else "WEAK"
        print(f"  {mask_name}: {status} (coherence: {validation_results[mask_name]['network_coherence']:.3f})")
    
    return validation_results


def extract_cap_timeseries(timeseries, masks, validation_results):
    """
    Extract averaged and z-scored time series for positive and negative regions of each CAP.
    Only use validated robust patterns.
    """
    print("Extracting CAP time series from weighted masks...")
    
    cap_timeseries = {}
    n_timepoints = timeseries.shape[0]
    
    for mask_name, mask_data in masks.items():
        # Check if pattern is robust
        if not validation_results[mask_name]['robust_pattern']:
            print(f"  Skipping {mask_name} - not robust enough")
            continue
            
        weights = mask_data['weights']
        
        if np.sum(weights) == 0:
            print(f"  Skipping {mask_name} - no active ROIs")
            continue
        
        # Extract weighted average time series
        # Multiply each ROI's timeseries by its weight, then average
        weighted_timeseries = np.zeros(n_timepoints)
        total_weight = 0
        
        for roi_idx, weight in enumerate(weights):
            if weight > 0:
                weighted_timeseries += timeseries[:, roi_idx] * weight
                total_weight += weight
        
        if total_weight > 0:
            weighted_timeseries /= total_weight
            
            # Z-score the time series
            z_scored_timeseries = stats.zscore(weighted_timeseries)
            
            cap_timeseries[mask_name] = z_scored_timeseries
            
            print(f"  Extracted {mask_name}: mean weight = {total_weight:.3f}")
        else:
            print(f"  Skipping {mask_name} - zero total weight")
    
    print(f"Successfully extracted {len(cap_timeseries)} CAP time series")
    return cap_timeseries


def correlate_caps_with_emotion(cap_timeseries, emotion_timeseries, alpha=0.05):
    """
    Correlate CAP time series with continuous valence and arousal using Spearman correlation.
    Apply multiple comparison correction.
    """
    print("Correlating CAP time series with emotion ratings...")
    
    if emotion_timeseries is None:
        print("No emotion time series provided - creating simulated data for demonstration")
        n_timepoints = len(list(cap_timeseries.values())[0])
        
        # Create simulated emotion time series
        np.random.seed(42)
        valence_ts = np.random.randn(n_timepoints)
        arousal_ts = np.random.randn(n_timepoints)
        
        # Add some correlation between valence and arousal
        arousal_ts = 0.3 * valence_ts + 0.7 * arousal_ts
        
        # Apply some smoothing to make it more realistic
        from scipy.ndimage import gaussian_filter1d
        valence_ts = gaussian_filter1d(valence_ts, sigma=2.0)
        arousal_ts = gaussian_filter1d(arousal_ts, sigma=2.0)
        
        emotion_timeseries = {
            'valence': valence_ts,
            'arousal': arousal_ts
        }
        print("Created simulated emotion time series")
    
    correlation_results = []
    
    # Perform correlations
    for cap_name, cap_ts in cap_timeseries.items():
        for emotion_name, emotion_ts in emotion_timeseries.items():
            # Ensure time series are same length
            min_length = min(len(cap_ts), len(emotion_ts))
            cap_ts_trimmed = cap_ts[:min_length]
            emotion_ts_trimmed = emotion_ts[:min_length]
            
            # Compute Spearman correlation
            correlation, p_value = spearmanr(cap_ts_trimmed, emotion_ts_trimmed)
            
            correlation_results.append({
                'cap_mask': cap_name,
                'emotion': emotion_name,
                'correlation': correlation,
                'p_value': p_value,
                'n_timepoints': min_length
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(correlation_results)
    
    # Apply multiple comparison correction (Bonferroni)
    if len(results_df) > 0:
        p_values = results_df['p_value'].values
        rejected, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method='bonferroni')
        
        results_df['p_corrected'] = p_corrected
        results_df['significant_corrected'] = rejected
        results_df['significant_uncorrected'] = p_values < alpha
        
        print(f"\nCorrelation Results (n={len(results_df)} tests):")
        print(f"Uncorrected significant: {np.sum(results_df['significant_uncorrected'])}")
        print(f"Bonferroni corrected significant: {np.sum(results_df['significant_corrected'])}")
        
        # Display significant results
        significant_results = results_df[results_df['significant_corrected']]
        if len(significant_results) > 0:
            print("\nSignificant correlations (p < 0.05, corrected):")
            for _, row in significant_results.iterrows():
                print(f"  {row['cap_mask']} - {row['emotion']}: r = {row['correlation']:.4f}, "
                      f"p = {row['p_corrected']:.4f}")
        else:
            print("\nNo significant correlations after multiple comparison correction")
    
    return results_df


def calculate_individual_cap_metrics(cluster_labels, onset_times=None):
    """
    Calculate individual-level CAP metrics:
    a) Fraction of time - proportion of total TRs spent in each brain state
    b) Persistence - average number of TRs in each brain state before transitioning
    c) Counts - number of times each brain state occurred
    """
    print("Calculating individual-level CAP metrics...")
    
    n_timepoints = len(cluster_labels)
    unique_clusters = np.unique(cluster_labels)
    
    metrics = {
        'total_timepoints': n_timepoints,
        'unique_states': len(unique_clusters)
    }
    
    for cluster in unique_clusters:
        cap_name = f"CAP_{cluster + 1}"
        
        # a) Fraction of time
        cluster_timepoints = np.sum(cluster_labels == cluster)
        fraction_time = cluster_timepoints / n_timepoints
        metrics[f"{cap_name}_fraction_time"] = fraction_time
        
        # b) Persistence (dwell time)
        # Find continuous segments of this cluster
        cluster_segments = []
        current_segment_length = 0
        
        for i, label in enumerate(cluster_labels):
            if label == cluster:
                current_segment_length += 1
            else:
                if current_segment_length > 0:
                    cluster_segments.append(current_segment_length)
                current_segment_length = 0
        
        # Don't forget the last segment if it ends with the target cluster
        if current_segment_length > 0:
            cluster_segments.append(current_segment_length)
        
        if cluster_segments:
            persistence = np.mean(cluster_segments)
            max_persistence = np.max(cluster_segments)
        else:
            persistence = 0
            max_persistence = 0
            
        metrics[f"{cap_name}_persistence"] = persistence
        metrics[f"{cap_name}_max_persistence"] = max_persistence
        
        # c) Counts (number of occurrences)
        counts = len(cluster_segments)
        metrics[f"{cap_name}_counts"] = counts
        
        print(f"  {cap_name}: fraction={fraction_time:.3f}, persistence={persistence:.2f}, counts={counts}")
    
    # Additional global metrics
    # Total number of transitions
    transitions = 0
    for i in range(1, len(cluster_labels)):
        if cluster_labels[i] != cluster_labels[i-1]:
            transitions += 1
    
    metrics['total_transitions'] = transitions
    metrics['transition_rate'] = transitions / n_timepoints if n_timepoints > 0 else 0
    
    print(f"  Global: {transitions} transitions, rate={metrics['transition_rate']:.4f}")
    
    return metrics


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


def main_caps_emotion_analysis():
    """
    Main CAPs-emotion analysis pipeline following the described methodology.
    """
    print("CAPs-Emotion Analysis Pipeline")
    print("=" * 50)
    print("Following methodology with Craddock atlas (268 ROIs)")
    print("Testing k=[2-20] clusters with elbow criterion")
    
    # Check for reliable runs from inter-rater reliability analysis
    irr_file = "derivatives/caps/interrater/interrater_reliability_results.tsv"
    reliable_runs = []
    
    try:
        irr_results = pd.read_csv(irr_file, sep='\t')
        # Check for runs with significant reliability
        for _, row in irr_results.iterrows():
            if row['valence_reliable'] or row['arousal_reliable']:
                reliable_runs.append(row['run'])
                print(f"Found reliable run: {row['run']} (valence: {row['valence_reliable']}, arousal: {row['arousal_reliable']})")
    except FileNotFoundError:
        print("No inter-rater reliability results found. Proceeding with available data...")
        reliable_runs = ['run-1']  # Default for development
    
    if not reliable_runs:
        print("❌ No reliable runs found. Cannot proceed with CAPs analysis.")
        print("Please run inter-rater reliability analysis first (1a-interrater-reliability.py)")
        return
    
    subject_id = "sub-Bubbles_ses-01"
    print(f"\nAnalyzing subject: {subject_id}")
    print(f"Reliable runs: {reliable_runs}")
    
    # Load Craddock atlas
    try:
        atlas_img, atlas_labels = load_craddock_atlas()
    except Exception as e:
        print(f"Error loading atlas: {e}")
        return
    
    # Simulate BOLD data extraction for development
    # In real analysis, this would load actual fMRI files
    print(f"\nExtracting BOLD time series from {len(reliable_runs)} runs...")
    timeseries_list = []
    
    for run in reliable_runs:
        print(f"Processing {run}...")
        # Simulate data extraction
        df, timeseries = extract_timeseries_zscore(f"simulated_{run}.nii.gz", atlas_img, atlas_labels)
        timeseries_list.append(timeseries)
    
    # Concatenate across all runs to preserve within-individual variability
    print(f"\nConcatenating {len(timeseries_list)} runs...")
    concatenated_timeseries = np.vstack(timeseries_list)
    print(f"Final matrix shape: {concatenated_timeseries.shape[0]} TRs x {concatenated_timeseries.shape[1]} ROIs")
    
    # Determine optimal number of clusters using elbow criterion
    print(f"\nStep 1: Determining optimal number of clusters...")
    try:
        optimal_k, validity_indices, inertias, elbow_fig = determine_optimal_clusters_elbow(
            concatenated_timeseries, k_range=[2, 20]
        )
        
        # Save elbow plot
        output_dir = Path(f"derivatives/caps/cap-analysis/{subject_id.replace('-', '_')}")
        output_dir.mkdir(parents=True, exist_ok=True)
        elbow_fig.savefig(output_dir / "elbow_criterion_plot.png", dpi=300, bbox_inches='tight')
        plt.close(elbow_fig)
        
    except Exception as e:
        print(f"Error in elbow analysis: {e}")
        optimal_k = 8  # Default fallback
        print(f"Using default k = {optimal_k}")
    
    # Perform k-means clustering with optimal k
    print(f"\nStep 2: Performing k-means clustering with k={optimal_k}...")
    cluster_labels, caps, kmeans, scaler = perform_kmeans_clustering(
        concatenated_timeseries, n_clusters=optimal_k, random_state=42
    )
    
    # Create weighted spatial masks
    print(f"\nStep 3: Creating weighted spatial masks...")
    masks = create_weighted_spatial_masks(caps, atlas_labels)
    
    # Perform network correspondence analysis
    print(f"\nStep 4: Validating spatial patterns...")
    validation_results = perform_network_correspondence_analysis(masks)
    
    # Extract CAP time series
    print(f"\nStep 5: Extracting CAP time series...")
    cap_timeseries = extract_cap_timeseries(concatenated_timeseries, masks, validation_results)
    
    # Load emotion time series (from inter-rater reliability results)
    print(f"\nStep 6: Loading emotion time series...")
    emotion_timeseries = None
    try:
        emotion_file = "derivatives/caps/interrater/aggregated_emotion_timeseries.tsv"
        emotion_df = pd.read_csv(emotion_file, sep='\t')
        
        # Extract reliable emotion dimensions
        valence_reliable = emotion_df['valence_reliable'].iloc[0] if 'valence_reliable' in emotion_df.columns else False
        arousal_reliable = emotion_df['arousal_reliable'].iloc[0] if 'arousal_reliable' in emotion_df.columns else False
        
        emotion_timeseries = {}
        if valence_reliable and 'valence_aggregated' in emotion_df.columns:
            valence_data = emotion_df['valence_aggregated'].dropna().values
            emotion_timeseries['valence'] = valence_data
            print(f"Loaded valence time series: {len(valence_data)} timepoints")
            
        if arousal_reliable and 'arousal_aggregated' in emotion_df.columns:
            arousal_data = emotion_df['arousal_aggregated'].dropna().values
            emotion_timeseries['arousal'] = arousal_data
            print(f"Loaded arousal time series: {len(arousal_data)} timepoints")
            
        if not emotion_timeseries:
            print("No reliable emotion time series found")
            emotion_timeseries = None
            
    except FileNotFoundError:
        print("No aggregated emotion time series found - will use simulated data")
        emotion_timeseries = None
    
    # Correlate CAPs with emotion
    print(f"\nStep 7: Correlating CAPs with emotion...")
    correlation_results = correlate_caps_with_emotion(cap_timeseries, emotion_timeseries)
    
    # Calculate individual-level CAP metrics
    print(f"\nStep 8: Calculating individual-level CAP metrics...")
    cap_metrics = calculate_individual_cap_metrics(cluster_labels)
    
    # Save all results
    print(f"\nStep 9: Saving results...")
    
    # Create comprehensive results
    results = {
        'subject_id': subject_id,
        'optimal_k': optimal_k,
        'n_timepoints': len(concatenated_timeseries),
        'n_rois': len(atlas_labels),
        'reliable_runs': reliable_runs,
        **cap_metrics
    }
    
    # Save CAP metrics
    metrics_df = pd.DataFrame([results])
    metrics_df.to_csv(output_dir / "cap_metrics.tsv", sep='\t', index=False)
    
    # Save correlation results
    if len(correlation_results) > 0:
        correlation_results.to_csv(output_dir / "cap_emotion_correlations.tsv", sep='\t', index=False)
    
    # Save CAPs and cluster assignments
    caps_df = pd.DataFrame(caps.T, columns=[f"CAP_{i+1}" for i in range(caps.shape[0])], index=atlas_labels)
    caps_df.to_csv(output_dir / "caps.tsv", sep='\t')
    
    # Save concatenated data with cluster labels
    concatenated_df = pd.DataFrame(concatenated_timeseries, columns=atlas_labels)
    concatenated_df['cluster'] = cluster_labels
    concatenated_df.to_csv(output_dir / "timeseries_with_clusters.tsv", sep='\t', index=False)
    
    # Save mask information
    mask_info = []
    for mask_name, mask_data in masks.items():
        mask_info.append({
            'mask_name': mask_name,
            'n_active_rois': mask_data['n_active_rois'],
            'mean_weight': mask_data['mean_weight'],
            'robust_pattern': validation_results[mask_name]['robust_pattern'],
            'network_coherence': validation_results[mask_name]['network_coherence']
        })
    
    mask_df = pd.DataFrame(mask_info)
    mask_df.to_csv(output_dir / "cap_masks_info.tsv", sep='\t', index=False)
    
    # Create visualizations
    print(f"\nStep 10: Creating visualizations...")
    try:
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # Plot CAPs
        plot_caps_heatmap(caps, atlas_labels, save_path=figures_dir / "caps_heatmap.png")
        
        # Plot correlation results
        if len(correlation_results) > 0:
            plot_emotion_correlations(correlation_results, save_path=figures_dir / "cap_emotion_correlations.png")
        
        # Plot CAP metrics
        plot_cap_metrics_individual(results, save_path=figures_dir / "individual_cap_metrics.png")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    print(f"\nCAPs-Emotion Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"\nKey findings:")
    print(f"- Identified {optimal_k} distinct brain states (CAPs)")
    print(f"- Analyzed {len(concatenated_timeseries)} total timepoints")
    print(f"- Extracted {len(cap_timeseries)} robust CAP time series")
    if len(correlation_results) > 0:
        n_sig = np.sum(correlation_results['significant_corrected'])
        print(f"- Found {n_sig} significant CAP-emotion correlations (corrected)")
    
    return results, correlation_results, caps


def plot_caps_heatmap(caps, atlas_labels, save_path=None):
    """Plot CAPs as heatmap with proper ROI labeling."""
    print("Creating CAPs heatmap...")
    
    n_clusters = caps.shape[0]
    
    # Create heatmap
    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    
    # Plot heatmap
    im = ax.imshow(caps, cmap='RdBu_r', aspect='auto', vmin=-3, vmax=3)
    
    # Set labels
    ax.set_xlabel('ROIs', fontsize=12)
    ax.set_ylabel('CAPs', fontsize=12)
    ax.set_title('Co-Activation Patterns (CAPs) - Craddock Atlas (268 ROIs)', fontsize=14, fontweight='bold')
    
    # Set ticks
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels([f"CAP {i+1}" for i in range(n_clusters)])
    
    # Only show some ROI labels to avoid overcrowding
    n_rois = len(atlas_labels)
    step = max(1, n_rois // 20)  # Show ~20 labels max
    ax.set_xticks(range(0, n_rois, step))
    ax.set_xticklabels([atlas_labels[i] for i in range(0, n_rois, step)], rotation=45, ha='right')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Z-score', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"CAPs heatmap saved to: {save_path}")
    
    plt.close()


def plot_emotion_correlations(correlation_results, save_path=None):
    """Plot CAP-emotion correlation results."""
    print("Creating CAP-emotion correlation plot...")
    
    # Create correlation matrix
    caps = correlation_results['cap_mask'].unique()
    emotions = correlation_results['emotion'].unique()
    
    corr_matrix = np.zeros((len(caps), len(emotions)))
    p_matrix = np.zeros((len(caps), len(emotions)))
    
    for i, cap in enumerate(caps):
        for j, emotion in enumerate(emotions):
            mask = (correlation_results['cap_mask'] == cap) & (correlation_results['emotion'] == emotion)
            if np.any(mask):
                corr_matrix[i, j] = correlation_results.loc[mask, 'correlation'].iloc[0]
                p_matrix[i, j] = correlation_results.loc[mask, 'p_corrected'].iloc[0]
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    
    # Plot heatmap
    im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Add significance markers
    for i in range(len(caps)):
        for j in range(len(emotions)):
            if p_matrix[i, j] < 0.05:
                ax.text(j, i, '*', ha='center', va='center', color='black', fontsize=20, fontweight='bold')
    
    # Set labels
    ax.set_xlabel('Emotion Dimensions', fontsize=12)
    ax.set_ylabel('CAP Masks', fontsize=12)
    ax.set_title('CAP-Emotion Correlations\n(* p < 0.05, corrected)', fontsize=14, fontweight='bold')
    
    ax.set_xticks(range(len(emotions)))
    ax.set_xticklabels(emotions)
    ax.set_yticks(range(len(caps)))
    ax.set_yticklabels(caps, rotation=0)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Spearman Correlation', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation plot saved to: {save_path}")
    
    plt.close()


def plot_cap_metrics_individual(results, save_path=None):
    """Plot individual-level CAP metrics."""
    print("Creating individual CAP metrics plot...")
    
    # Extract CAP metrics
    cap_names = []
    fractions = []
    persistences = []
    counts = []
    
    for key, value in results.items():
        if '_fraction_time' in key:
            cap_name = key.replace('_fraction_time', '')
            cap_names.append(cap_name)
            fractions.append(value)
            persistences.append(results.get(f"{cap_name}_persistence", 0))
            counts.append(results.get(f"{cap_name}_counts", 0))
    
    if not cap_names:
        print("No CAP metrics found to plot")
        return
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Fraction of time
    axes[0].bar(cap_names, fractions, alpha=0.7, color='skyblue')
    axes[0].set_title('Fraction of Time', fontweight='bold')
    axes[0].set_ylabel('Proportion')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # Persistence
    axes[1].bar(cap_names, persistences, alpha=0.7, color='lightcoral')
    axes[1].set_title('Persistence (Dwell Time)', fontweight='bold')
    axes[1].set_ylabel('Average TRs')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    # Counts
    axes[2].bar(cap_names, counts, alpha=0.7, color='lightgreen')
    axes[2].set_title('Occurrence Counts', fontweight='bold')
    axes[2].set_ylabel('Number of Occurrences')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Individual-Level CAP Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"CAP metrics plot saved to: {save_path}")
    
    plt.close()


def main_subject_level():
    """Updated main function to call the new CAPs-emotion analysis."""
    return main_caps_emotion_analysis()


def main():
    """Updated main function to call the new CAPs-emotion analysis."""
    return main_caps_emotion_analysis()
