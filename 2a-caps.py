#!/usr/bin/env python3
"""
CAP Analysis Script
Converted from 2a-caps.ipynb
Performs time series extraction and k-means clustering to identify co-activation patterns (CAPs)
"""

import pandas as pd
import numpy as np
import pingouin as pg
from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import os.path as op
import re
import nibabel as nib

# Additional imports for neuroimaging and atlas processing
from nilearn import datasets, masking, input_data, plotting
from nilearn.connectome import ConnectivityMeasure
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def main():
    print("Starting CAP Analysis...")
    
    # Setup directories and plotting theme
    RELI_DIR = Path("dset/derivatives/caps/interrater")
    FIGURES_DIR = Path("dset/derivatives/figures")
    OUT_DIR = Path("dset/derivatives/caps")
    KMEANS_DIR = OUT_DIR / "kmeans"
    KMEANS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {OUT_DIR}")
    print(f"K-means directory: {KMEANS_DIR}")
    
    # Load Craddock atlas from local file
    print("Loading Craddock atlas from local file...")
    
    atlas_filename = Path("dset/craddock2012_tcorr05_2level_270_2mm.nii")
    print(f"Atlas loaded: {atlas_filename}")
    
    # Load the atlas to check its properties
    atlas_img = nib.load(atlas_filename)
    atlas_data = atlas_img.get_fdata()
    
    # Get unique ROI labels (excluding background/0)
    unique_labels = np.unique(atlas_data)
    print(f"Unique labels before filtering: {unique_labels}")
    unique_labels = unique_labels[unique_labels != 0]  # Remove background
    n_rois = len(unique_labels)
    
    print(f"Atlas shape: {atlas_data.shape}")
    print(f"Number of ROIs: {n_rois}")
    print(f"ROI labels range: {unique_labels.min()} to {unique_labels.max()}")
    
    # Background statistics
    background_voxels = np.sum(atlas_data == 0)
    total_voxels = np.prod(atlas_data.shape)
    background_percentage = (background_voxels / total_voxels) * 100
    
    print(f"Background Statistics:")
    print(f"• Background voxels: {background_voxels:,}")
    print(f"• Total voxels: {total_voxels:,}")
    print(f"• Background coverage: {background_percentage:.1f}%")
    
    # Create masker for extracting time series from ROIs
    masker = input_data.NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=True,  # z-scores the time series
        memory='nilearn_cache',
        mask_type="whole-brain",
        verbose=1
    )
    
    # Define all runs for each participant and episode
    participant_data = {
        "sub-Blossom": {
            "episode_1": [1, 2, 3],
            "episode_2": [1, 2, 3, 4, 5, 6, 7],  # Available runs for episode 2
            "episode_3": [1, 5],
            "episode_4": [1, 2, 3, 4, 5, 6]
        },
        "sub-Bubbles": {
            "episode_1": [1, 2, 3],
            "episode_2": [1, 2, 3, 4, 5, 6, 7],  # Available runs for episode 2
            "episode_3": [1, 5],
            "episode_4": [1, 2, 3, 4, 5, 6]
        },
        "sub-Buttercup": {
            "episode_1": [1, 2, 3],
            "episode_2": [1, 2, 3, 4, 5, 6, 7],  # Available runs for episode 2
            "episode_3": [1, 5],
            "episode_4": [1, 2, 3, 4, 5, 6]
        }
        # Add other participants as needed
    }
    
    # Extract BOLD time series and create z-scored participant matrices
    print("Extracting BOLD time series...")
    
    # Initialize storage for participant-level matrices
    all_participant_matrices = {}
    
    for sub_id, episodes in participant_data.items():
        print(f"\n{'='*60}")
        print(f"PROCESSING {sub_id}")
        print(f"{'='*60}")
        
        participant_timeseries = []
        
        for episode_key, run_numbers in episodes.items():
            ep_num = int(episode_key.split('_')[1])
            
            print(f"\nProcessing Episode {ep_num} with {len(run_numbers)} runs...")
            
            for run_num in run_numbers:
                print(f"  Processing run {run_num}...")
                
                TASK_DIR = Path(f"dset/{sub_id}/ses-{ep_num:02d}/func") 
                
                # Construct the filename - note that run number is NOT zero-padded
                task_filename = f"{sub_id}_ses-{ep_num:02d}_task-strangerthings_run-{run_num}_part-mag_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
                task_filepath = TASK_DIR / task_filename
                
                # Extract time series from 268 ROIs
                print(f"    Extracting time series from {task_filepath.name}...")
                time_series = masker.fit_transform(task_filepath)
                    
                print(f"    Time series shape: {time_series.shape} (TRs x ROIs)")
                    
                # Store z-scored time series for this run
                participant_timeseries.append(time_series)
        
        if participant_timeseries:
            # Concatenate all runs for this participant
            print(f"\nConcatenating {len(participant_timeseries)} runs for {sub_id}...")
            participant_matrix = np.vstack(participant_timeseries)
            
            print(f"Final participant matrix shape: {participant_matrix.shape}")
            print(f"  - Total TRs across all runs: {participant_matrix.shape[0]}")
            print(f"  - Number of ROIs (Craddock): {participant_matrix.shape[1]}")
            
            # Store the participant-level matrix
            all_participant_matrices[sub_id] = participant_matrix
            
            # Save the participant matrix   
            output_file = KMEANS_DIR / f"{sub_id}_zscore_strangerthings_matrix.npy"
            np.save(output_file, participant_matrix)
            print(f"Saved participant matrix to: {output_file}")
            
            # Also save as CSV for easier inspection
            output_csv = KMEANS_DIR / f"{sub_id}_zscore_strangerthings_matrix.csv"
            df_matrix = pd.DataFrame(participant_matrix, 
                                    columns=[f"{unique_labels[i]}" for i in range(participant_matrix.shape[1])])
            df_matrix.to_csv(output_csv, index=False)
            print(f"Saved participant matrix (CSV) to: {output_csv}")
    
    print(f"\n{'='*60}")
    print("TIME SERIES EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Processed {len(all_participant_matrices)} participants:")
    for sub_id, matrix in all_participant_matrices.items():
        print(f"  {sub_id}: {matrix.shape[0]} TRs × {matrix.shape[1]} ROIs")
    
    # Perform k-means clustering
    print("\n" + "="*60)
    print("STARTING K-MEANS CLUSTERING")
    print("="*60)
    
    masks_output_dir = OUT_DIR / "caps_masks"
    masks_output_dir.mkdir(parents=True, exist_ok=True)
    
    for sub_id, episodes in participant_data.items():
        print(f"\n{'='*60}")
        print(f"PROCESSING {sub_id}")
        print(f"{'='*60}")
        
        timeseries_file = KMEANS_DIR / f"{sub_id}_zscore_strangerthings_matrix.npy"
        
        if timeseries_file.exists():
            # Load the participant time series matrix
            participant_matrix = np.load(timeseries_file)
            print(f"Loaded time series matrix: {participant_matrix.shape}")
            print(f"  - Time points (TRs): {participant_matrix.shape[0]}")
            print(f"  - ROIs: {participant_matrix.shape[1]}")
            
            # Determine optimal number of clusters using elbow method and silhouette score
            print("\nDetermining optimal number of clusters...")
            k_range = range(2, 21)  # Test 2 to 20 clusters
            inertias = []
            silhouette_scores = []
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(participant_matrix)
                
                inertias.append(kmeans.inertia_)
                sil_score = silhouette_score(participant_matrix, cluster_labels)
                silhouette_scores.append(sil_score)
                
                print(f"  k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.3f}")
            
            # Find optimal k based on elbow method
            # Calculate the rate of change (differences) in inertia
            inertia_diffs = np.diff(inertias)
            
            # Calculate the second derivative (rate of change of the rate of change)
            second_diffs = np.diff(inertia_diffs)
            
            # Find the elbow point - where the second derivative is maximum
            # (greatest change in the rate of decrease)
            elbow_idx = np.argmax(second_diffs) + 2  # +2 because we lost 2 points in double diff
            optimal_k = k_range[elbow_idx]
            
            # Also report the silhouette score for this k
            corresponding_silhouette = silhouette_scores[elbow_idx]
            
            print(f"\nElbow method results:")
            print(f"  - Optimal number of clusters (elbow): {optimal_k}")
            print(f"  - Silhouette score at elbow k: {corresponding_silhouette:.3f}")
            print(f"  - Inertia at elbow k: {inertias[elbow_idx]:.2f}")
            
            # For comparison, also show the k with best silhouette score
            best_sil_k = k_range[np.argmax(silhouette_scores)]
            max_silhouette = max(silhouette_scores)
            print(f"\nFor comparison:")
            print(f"  - Best silhouette k: {best_sil_k}")
            print(f"  - Best silhouette score: {max_silhouette:.3f}")
            
            print(f"\nOptimal number of clusters: {optimal_k}")
            print(f"Silhouette score at optimal k: {corresponding_silhouette:.3f}")
            
            # Perform final clustering with optimal k (from elbow method)
            print(f"\nPerforming final k-means clustering with k={optimal_k} (elbow method)...")
            final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            final_labels = final_kmeans.fit_predict(participant_matrix)
            
            # Get cluster information
            cluster_unique_labels, label_counts = np.unique(final_labels, return_counts=True)
            
            print(f"Clustering results:")
            print(f"  - Total time points: {len(final_labels)}")
            print(f"  - Number of clusters: {len(cluster_unique_labels)}")
            
            for i, (label, count) in enumerate(zip(cluster_unique_labels, label_counts)):
                percentage = (count / len(final_labels)) * 100
                print(f"  - CAP {label+1}: {count} time points ({percentage:.1f}%)")
            
            # Save clustering results
            labels_csv_file = KMEANS_DIR / f"{sub_id}_cluster_labels.csv"
            df_labels = pd.DataFrame({
                'timepoint': range(len(final_labels)),
                'cluster_label': final_labels
            })
            df_labels.to_csv(labels_csv_file, index=False)
            print(f"Saved cluster labels (CSV) to: {labels_csv_file}")
            
            # Save cluster centers (CAPs) as CSV for easy inspection
            centers_csv_file = KMEANS_DIR / f"{sub_id}_cluster_centers.csv"
            
            df_centers = pd.DataFrame(
                final_kmeans.cluster_centers_,
                columns=[f"{unique_labels[i]}" for i in range(final_kmeans.cluster_centers_.shape[1])],
                index=[f"CAP_{i+1}" for i in range(final_kmeans.cluster_centers_.shape[0])]
            )
            df_centers.to_csv(centers_csv_file)
            print(f"Saved cluster centers (CSV) to: {centers_csv_file}")
            
            # Save clustering metadata
            metadata = {
                'n_clusters': optimal_k,
                'method': 'elbow',
                'silhouette_score': corresponding_silhouette,
                'inertia': final_kmeans.inertia_,
                'cluster_sizes': label_counts.tolist(),
                'total_timepoints': len(final_labels),
                'best_silhouette_k': best_sil_k,
                'max_silhouette_score': max_silhouette
            }
            
            metadata_file = KMEANS_DIR / f"{sub_id}_clustering_metadata.txt"
            with open(metadata_file, 'w') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
            print(f"Saved clustering metadata to: {metadata_file}")
            
            # Create weighted CAP masks
            print(f"\nCreating weighted CAP masks for {sub_id}...")
            
            PCT_OF_MAX = 0.0  # keep |z| >= (PCT_OF_MAX/100) * max|z| per CAP
            
            centers_df = pd.read_csv(centers_csv_file)
            
            for i, cap in centers_df.iterrows():
                cap_df = pd.DataFrame([cap], columns=centers_df.columns)
                cap_name = cap_df.iloc[0, 0]
                roi_labels = [float(c) for c in cap_df.columns[1:]]
                z_vals = cap_df.iloc[0, 1:].astype(float).values
                
                print(f"  Processing {cap_name}...")
                
                # Z-score normalize to [-1, 1] using max |activation|
                vabs_full = float(np.nanmax(np.abs(z_vals)))
                if vabs_full == 0 or not np.isfinite(vabs_full):
                    print(f"    Skipping {cap_name}: max|z| is zero or NaN.")
                    continue
                z_vals_norm = z_vals / vabs_full  # Now in [-1, 1]
                
                # Apply threshold on normalized values
                thr_norm = PCT_OF_MAX / 100.0  # e.g., 0.15 for 15%
                z_vals_thresh = np.where(np.abs(z_vals_norm) >= thr_norm, z_vals_norm, 0.0)
                
                # Map normalized & thresholded values to atlas (stay in [-1, 1])
                atlas_data = np.asanyarray(atlas_img.dataobj)
                mask_data = np.zeros_like(atlas_data, dtype=np.float32)
                for roi, z_norm in zip(roi_labels, z_vals_thresh):
                    if np.isfinite(z_norm) and z_norm != 0.0:
                        mask_data[atlas_data == int(roi)] = z_norm  # values in [-1, 1]
                
                # Save z-score normalized weighted CAP (values in [-1, 1])
                mask_img = nib.Nifti1Image(mask_data, atlas_img.affine, atlas_img.header)
                mask_path = masks_output_dir / f"{sub_id}_zscore-weighted-{int(PCT_OF_MAX)}_{cap_name}.nii.gz"
                nib.save(mask_img, mask_path)
                print(f"    Saved z-score weighted NIfTI: {mask_path}")
                print(f"    Original range:    [{np.min(z_vals):.3f}, {np.max(z_vals):.3f}]")
                nonzero = mask_data[mask_data != 0]
                if nonzero.size:
                    print(f"    Normalized (kept): [{np.min(nonzero):.3f}, {np.max(nonzero):.3f}]")
                else:
                    print("    Normalized (kept): [empty after threshold]")
                
                # Split into positive and negative parts
                pos_data = np.where(mask_data > 0, mask_data, 0.0)
                neg_data = np.where(mask_data < 0, mask_data, 0.0)
                
                pos_img = nib.Nifti1Image(pos_data, atlas_img.affine, atlas_img.header)
                neg_img = nib.Nifti1Image(neg_data, atlas_img.affine, atlas_img.header)
                
                # Save separate positive and negative weighted maps (still in [-1, 1])
                pos_path = masks_output_dir / f"{sub_id}_zscore-weighted-{int(PCT_OF_MAX)}_{cap_name}_pos.nii.gz"
                neg_path = masks_output_dir / f"{sub_id}_zscore-weighted-{int(PCT_OF_MAX)}_{cap_name}_neg.nii.gz"
                nib.save(pos_img, pos_path)
                nib.save(neg_img, neg_path)
                print(f"    Saved positive weights: {pos_path}")
                print(f"    Saved negative weights: {neg_path}")
                
        else:
            print(f"ERROR: Time series file not found: {timeseries_file}")
    
    print(f"\n{'='*60}")
    print("CLUSTERING COMPLETE")
    print(f"{'='*60}")
    print("All CAP analysis completed successfully!")

if __name__ == "__main__":
    main()