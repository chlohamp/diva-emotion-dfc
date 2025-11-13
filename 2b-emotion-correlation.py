#!/usr/bin/env python3
"""
Emotion Correlation Analysis Script
Converted from 2b-emotion-correlation.ipynb
Extracts CAP timeseries using weighted masks and correlates with emotion ratings
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
import json

# Additional imports for neuroimaging and atlas processing
from nilearn import datasets, masking, input_data, plotting
from nilearn.maskers import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def main():
    print("Starting Emotion Correlation Analysis...")
    
    # Setup directories and plotting theme
    RELI_DIR = Path("dset/derivatives/caps/interrater")
    FIGURES_DIR = Path("dset/derivatives/figures")
    # will use loop later to run over all subjects
    OUT_DIR = Path("dset/derivatives/caps/emotion-correlation")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {OUT_DIR}")
    print(f"Figures directory: {FIGURES_DIR}")
    
    # Define all runs for each participant and episode
    participant_data = {
        "sub-Blossom": {
            "S01E01": [1, 2, 3], 
            "S01E02": [1, 2, 3, 4, 5, 6, 7], 
        },
        "sub-Bubbles": {
            "S01E01": [1, 2, 3], 
            "S01E02": [1, 2, 3, 4, 5, 6, 7], 
            "S01E03": [1, 5], 
            "S01E04": [1, 2, 3, 4, 5, 6]   
        },
        "sub-Buttercup": {
            "S01E01": [1, 2, 3],           
            "S01E02": [1, 2, 3, 4, 5, 6, 7], 
            "S01E03": [1, 5],             
            "S01E04": [1, 2, 3, 4, 5, 6]  
        }
    }
    
    # Define CAP masks to process - participant-specific
    def get_cap_masks_for_participant(sub_id):
        cap_masks = []
        if sub_id == "sub-Bubbles":
            cap_range = range(1, 6)  # CAPs 1-5 for Bubbles
        elif sub_id == "sub-Buttercup":
            cap_range = range(1, 5)  # CAPs 1-4 for Buttercup
        elif sub_id == "sub-Blossom":
            cap_range = range(1, 5)  # CAPs 1-4 for Blossom
        else:
            cap_range = range(1, 5)  # Default
            
        for cap_num in cap_range:
            for polarity in ['pos', 'neg']:
                mask_path = Path(f"dset/derivatives/caps/caps_masks/{sub_id}_zscore-weighted-0_CAP_{cap_num}_{polarity}.nii.gz")
                if mask_path.exists():
                    cap_masks.append({
                        'cap_num': cap_num,
                        'polarity': polarity,
                        'path': mask_path,
                        'name': f"CAP{cap_num}_{polarity}"
                    })
        return cap_masks

    removed_clips_df = pd.read_csv(Path("dset/derivatives/annotations/removed_clips_log.csv"))

    # Initialize storage for all participant-episode combinations
    all_combined_timeseries = {}

    print(f"\n{'='*80}")
    print("EXTRACTING CAP TIMESERIES")
    print(f"{'='*80}")
    
    # Process each participant-episode combination
    for sub_id, episodes in participant_data.items():
        print(f"\n{'='*80}")
        print(f"PROCESSING {sub_id}")
        print(f"{'='*80}")
        
        for episode_key, run_numbers in episodes.items():
            # Handle both episode naming conventions
            if episode_key.startswith('S01E'):
                ep_num = int(episode_key[-2:])  # Extract from S01E02
            else:
                ep_num = int(episode_key.split('_')[1])  # Extract from episode_2
            
            output_csv = OUT_DIR / f"{sub_id}_{episode_key}_all_caps_timeseries.csv"
            if output_csv.exists():
                print(f"\nSkipping {sub_id} {episode_key} — CSV already exists: {output_csv}")
                continue

            print(f"\n{'='*60}")
            print(f"Processing {episode_key} with {len(run_numbers)} runs...")
            print(f"{'='*60}")
            
            # Get excluded clips for this episode
            excluded_clips_episode = removed_clips_df[removed_clips_df['episode'] == episode_key]['episode_position'].tolist()
            print(f"Found {len(excluded_clips_episode)} excluded clips for {episode_key}: {excluded_clips_episode[:10] if len(excluded_clips_episode) > 10 else excluded_clips_episode}...")
            
            participant_episode_key = f"{sub_id}_{episode_key}"
            all_cap_timeseries = {}
            
            # Get CAP masks for this specific participant
            cap_masks = get_cap_masks_for_participant(sub_id)
            print(f"Found {len(cap_masks)} CAP masks for {sub_id}:")
            for mask_info in cap_masks:
                print(f"  - {mask_info['name']}: {mask_info['path']}")
            
            for mask_info in cap_masks:
                print(f"\n  Processing {mask_info['name']}...")
                
                mask_img = nib.load(mask_info['path'])
                mask_data = mask_img.get_fdata()

                caps_masker = NiftiMapsMasker(
                    maps_img=mask_img,
                    standardize=True,
                    memory='nilearn_cache',
                    mask_type="whole-brain",
                    verbose=0
                )
                
                participant_timeseries = []
                all_clip_positions = []
                
                for run_num in run_numbers:
                    print(f"    Processing run {run_num}...")
                    
                    TASK_DIR = Path(f"dset/{sub_id}/ses-{ep_num:02d}/func") 
                    task_filename = f"{sub_id}_ses-{ep_num:02d}_task-strangerthings_run-{run_num}_part-mag_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
                    task_filepath = TASK_DIR / task_filename

                    time_series = caps_masker.fit_transform(task_filepath)
                            
                    if time_series.shape[0] > 5:
                        time_series_trimmed = time_series[5:]
                    else:
                        print(f"    WARNING: Run has only {time_series.shape[0]} TRs, cannot remove 5 TRs!")
                        time_series_trimmed = time_series
                    
                    if run_num == 1:
                        run_start_clip = 1
                    else:
                        run_start_clip = len(all_clip_positions) + 1
                    
                    tr_positions = list(range(run_start_clip, run_start_clip + time_series_trimmed.shape[0]))
                    all_clip_positions.extend(tr_positions)
                    participant_timeseries.append(time_series_trimmed)

                if participant_timeseries:
                    participant_matrix = np.vstack(participant_timeseries)
                    exclude_mask = np.array([pos in excluded_clips_episode for pos in all_clip_positions])
                    keep_mask = ~exclude_mask
                    excluded_count = np.sum(exclude_mask)
                    
                    if np.any(keep_mask):
                        participant_matrix_filtered = participant_matrix[keep_mask]
                    else:
                        print("Warning: All TRs would be excluded! Using original matrix.")
                        participant_matrix_filtered = participant_matrix
                    
                    all_cap_timeseries[mask_info['name']] = participant_matrix_filtered.flatten()
                    print(f"    {mask_info['name']}: {participant_matrix_filtered.shape} -> {len(participant_matrix_filtered.flatten())} timepoints")
            
            if all_cap_timeseries:
                timeseries_lengths = [len(ts) for ts in all_cap_timeseries.values()]
                min_length = min(timeseries_lengths) if timeseries_lengths else 0
                
                if len(set(timeseries_lengths)) > 1:
                    print(f"  Warning: Different timeseries lengths found: {set(timeseries_lengths)}")
                    print(f"  Truncating all to minimum length: {min_length}")
                    
                combined_df_data = {cap_name: ts[:min_length] for cap_name, ts in all_cap_timeseries.items()}
                combined_df = pd.DataFrame(combined_df_data)
                
                combined_df.to_csv(output_csv, index=False)
                print(f"\nSaved combined timeseries CSV: {output_csv}")
                print(f"Shape: {combined_df.shape} (timepoints x CAPs)")
                print(f"Columns: {list(combined_df.columns)}")
                
                all_combined_timeseries[participant_episode_key] = combined_df
                
                exclusion_info = {
                    'episode': episode_key,
                    'participant': sub_id,
                    'runs_processed': run_numbers,
                    'original_trs_before_trimming': int(participant_matrix.shape[0]) + (len(run_numbers) * 5),
                    'trs_removed_for_t1_equilibration': len(run_numbers) * 5,
                    'original_trs_after_trimming': int(participant_matrix.shape[0]),
                    'excluded_trs': int(excluded_count),
                    'final_trs': min_length,
                    'excluded_positions': [float(pos) for pos in excluded_clips_episode],
                    'cap_columns': list(combined_df.columns)
                }
                
                exclusion_file = OUT_DIR / f"{sub_id}_{episode_key}_exclusion_info.json"
                with open(exclusion_file, 'w') as f:
                    json.dump(exclusion_info, f, indent=2)
                print(f"Saved exclusion info: {exclusion_file}")

    print(f"\n{'='*80}")
    print("CAP TIMESERIES EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"Processed {len(all_combined_timeseries)} participant-episode combinations:")
    for key, df in all_combined_timeseries.items():
        print(f"  {key}: {df.shape[0]} timepoints × {df.shape[1]} CAPs")
        print(f"    Columns: {list(df.columns)}")

    print(f"\n{'='*80}")
    print("STARTING EPISODE-LEVEL CORRELATION PLOTTING")
    print(f"{'='*80}")

    # Plot CAP weighted timeseries with emotion ratings using combined CSV files
    print("Loading combined CAP timeseries files for plotting...")

    for sub_id, episodes in participant_data.items():
        print(f"\n{'='*80}")
        print(f"PLOTTING {sub_id}")
        print(f"{'='*80}")

        for episode_key, run_numbers in episodes.items():
            # Handle both episode naming conventions
            if episode_key.startswith('S01E'):
                ep_num = int(episode_key[-2:])  # Extract from S01E02
            else:
                ep_num = int(episode_key.split('_')[1])  # Extract from episode_2
            
            print(f"\n{'='*60}")
            print(f"Processing {episode_key}")
            print(f"{'='*60}")

            # Load combined CAP timeseries data
            combined_csv_path = OUT_DIR / f"{sub_id}_{episode_key}_all_caps_timeseries.csv"
            emotion_csv_path = RELI_DIR / f"S01E{ep_num:02d}_avg.csv"
            
            if not combined_csv_path.exists():
                print(f"Warning: Combined CAP data not found at {combined_csv_path}")
                continue
                
            if not emotion_csv_path.exists():
                print(f"Warning: Emotion data not found at {emotion_csv_path}")
                continue
            
            # Load the data
            caps_df = pd.read_csv(combined_csv_path)
            emotion_df = pd.read_csv(emotion_csv_path)
            
            print(f"Loaded CAP data: {caps_df.shape} (timepoints x CAPs)")
            print(f"CAP columns: {list(caps_df.columns)}")
            print(f"Loaded emotion data: {emotion_df.shape}")
            
            # Get emotion data
            valence = emotion_df['valence']
            arousal = emotion_df['arousal']
            
            # Ensure same length across all data
            min_length = min(len(valence), len(arousal), len(caps_df))
            valence = valence[:min_length]
            arousal = arousal[:min_length]
            caps_df_trimmed = caps_df.iloc[:min_length]
            
            print(f"Using {min_length} timepoints for analysis")
            
            # Plot each CAP column
            for cap_name in caps_df.columns:
                print(f"\n{'='*50}")
                print(f"Plotting {cap_name}")
                print(f"{'='*50}")
                
                # Get CAP timeseries for this column
                cap_timeseries = caps_df_trimmed[cap_name]
                
                # Calculate correlations for display
                val_corr, val_p = spearmanr(cap_timeseries, valence)
                aro_corr, aro_p = spearmanr(cap_timeseries, arousal)

                # Create the plot
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
                
                # Define colors
                colors = ['#CF9397', "#606933", '#E5D28E']
                
                # Valence plot
                ax1.axhspan(1, 4, alpha=0.1, color='red', zorder=0)
                ax1.axhspan(4, 7, alpha=0.1, color='green', zorder=0)
                
                # Plot valence ratings
                ax1.plot(range(len(valence)), valence, color=colors[0], marker='o', markersize=4, 
                         linewidth=2, alpha=0.8, label='Valence Ratings')
                
                # Plot CAP weighted timeseries on secondary y-axis
                ax1_twin = ax1.twinx()
                ax1_twin.plot(range(len(cap_timeseries)), cap_timeseries, color=colors[1], marker='s', markersize=4, 
                              linewidth=2, alpha=0.8, label=f'{cap_name} Weighted')
                
                ax1.set_title(f'{sub_id} {episode_key} - Valence vs {cap_name} Weighted', fontsize=16, fontweight='bold')
                ax1.set_xlabel(f'Clips ({len(valence)} total clips after exclusions)', fontsize=12)
                ax1.set_ylabel('Valence Rating', fontsize=12, color=colors[0])
                ax1_twin.set_ylabel(f'{cap_name} Weighted', fontsize=12, color=colors[1])
                ax1.set_ylim(0, 8)
                
                # Add reference lines for valence
                ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=2)
                ax1.axhline(y=4, color='gray', linestyle='--', alpha=0.7, linewidth=2)
                ax1.axhline(y=7, color='green', linestyle='--', alpha=0.7, linewidth=2)
                
                # Add text labels
                x_right = ax1.get_xlim()[1]
                ax1.text(x_right, 1, 'Negative Valence', ha='left', va='bottom', fontsize=10, color='red')
                ax1.text(x_right, 4, 'Neutral', ha='left', va='bottom', fontsize=10, color='gray')
                ax1.text(x_right, 7, 'Positive Valence', ha='left', va='bottom', fontsize=10, color='green')
                
                # Combined legend
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax1_twin.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                
                # Add correlation results
                stats_text = [f"Spearman ρ: {val_corr:.3f}"]
                p_str = f"{val_p:.3f}" if val_p >= 0.001 else "<0.001"
                stats_text.append(f"p: {p_str}")
                
                ax1.text(0.02, 0.98, '\n'.join(stats_text), transform=ax1.transAxes,
                         fontsize=11, verticalalignment='top', horizontalalignment='left',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

                # Arousal plot  
                # Plot arousal ratings
                ax2.plot(range(len(arousal)), arousal, color=colors[0], marker='o', markersize=4, 
                         linewidth=2, alpha=0.8, label='Arousal Ratings')
                
                # Plot CAP weighted timeseries on secondary y-axis
                ax2_twin = ax2.twinx()
                ax2_twin.plot(range(len(cap_timeseries)), cap_timeseries, color=colors[1], marker='s', markersize=4, 
                              linewidth=2, alpha=0.8, label=f'{cap_name} Weighted')
                
                ax2.set_title(f'{sub_id} {episode_key} - Arousal vs {cap_name} Weighted', fontsize=16, fontweight='bold')
                ax2.set_xlabel(f'Clips ({len(arousal)} total clips after exclusions)', fontsize=12)
                ax2.set_ylabel('Arousal Rating', fontsize=12, color=colors[0])
                ax2_twin.set_ylabel(f'{cap_name} Weighted', fontsize=12, color=colors[1])
                ax2.set_ylim(0, 8)
                
                # Add reference lines for arousal
                ax2.axhline(y=1, color='black', linestyle='--', alpha=0.7, linewidth=2)
                ax2.axhline(y=7, color='black', linestyle='--', alpha=0.7, linewidth=2)
                
                # Add text labels
                x_right = ax2.get_xlim()[1]
                ax2.text(x_right, 1, 'Low Arousal', ha='left', va='bottom', fontsize=10, color='black')
                ax2.text(x_right, 7, 'High Arousal', ha='left', va='bottom', fontsize=10, color='black')
                
                # Combined legend
                lines1, labels1 = ax2.get_legend_handles_labels()
                lines2, labels2 = ax2_twin.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                
                # Add correlation results
                stats_text = [f"Spearman ρ: {aro_corr:.3f}"]
                p_str = f"{aro_p:.3f}" if aro_p >= 0.001 else "<0.001"
                stats_text.append(f"p: {p_str}")
                
                ax2.text(0.02, 0.98, '\n'.join(stats_text), transform=ax2.transAxes,
                         fontsize=11, verticalalignment='top', horizontalalignment='left',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

                plt.tight_layout()
                
                # Save figure with CAP name
                figure_filename = f"{sub_id}_{episode_key}_{cap_name}_emotion_correlation.png"
                figure_path = FIGURES_DIR / figure_filename
                fig.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"Figure saved: {figure_path}")
                
                plt.close(fig)  # Close figure to save memory
                
                print(f"Data lengths: CAP={len(cap_timeseries)}, Valence={len(valence)}, Arousal={len(arousal)}")
                print(f"Correlations: Valence r={val_corr:.3f} (p={val_p:.3f}), Arousal r={aro_corr:.3f} (p={aro_p:.3f})")

    print(f"\n{'='*80}")
    print("EPISODE-LEVEL PLOTTING COMPLETE")
    print(f"{'='*80}")

    print(f"\n{'='*80}")
    print("STARTING RUN-LEVEL CORRELATION PLOTTING")
    print(f"{'='*80}")

    # Plot CAP weighted timeseries with emotion ratings BY INDIVIDUAL RUNS
    print("Processing CAP timeseries by individual runs using combined data...")

    PLOT_ABS_R_THRESHOLD = 0.35  # only plot when |rho| > 0.35

    def passes_threshold(rho, thr=PLOT_ABS_R_THRESHOLD):
        # Return True only if rho is finite and above threshold in magnitude
        return np.isfinite(rho) and (abs(rho) > thr)

    for sub_id, episodes in participant_data.items():
        print(f"\n{'='*80}")
        print(f"PLOTTING {sub_id} - BY RUNS")
        print(f"{'='*80}")

        for episode_key, run_numbers in episodes.items():
            # Handle both episode naming conventions
            if episode_key.startswith('S01E'):
                ep_num = int(episode_key[-2:])  # Extract from S01E02
            else:
                ep_num = int(episode_key.split('_')[1])  # Extract from episode_2
            
            print(f"\n{'='*60}")
            print(f"Processing {episode_key} - Individual Runs")
            print(f"{'='*60}")

            # Load combined CAP timeseries data that we already computed
            combined_csv_path = OUT_DIR / f"{sub_id}_{episode_key}_all_caps_timeseries.csv"
            emotion_csv_path = RELI_DIR / f"S01E{ep_num:02d}_avg.csv"
            
            if not combined_csv_path.exists():
                print(f"Warning: Combined CAP data not found at {combined_csv_path}")
                continue
                
            if not emotion_csv_path.exists():
                print(f"Warning: Emotion data not found at {emotion_csv_path}")
                continue
            
            # Load the data
            caps_df = pd.read_csv(combined_csv_path)
            emotion_df = pd.read_csv(emotion_csv_path)
            
            print(f"Loaded CAP data: {caps_df.shape} (timepoints x CAPs)")
            print(f"Loaded emotion data with run info: {emotion_df.shape}")
            print(f"Available runs in emotion data: {sorted(emotion_df['run'].unique())}")
            
            # Ensure same length across all data
            min_length = min(len(emotion_df), len(caps_df))
            emotion_df_trimmed = emotion_df.iloc[:min_length]
            caps_df_trimmed = caps_df.iloc[:min_length]
            
            print(f"Using {min_length} timepoints for analysis")
            
            # Process each run individually
            for run_num in run_numbers:
                print(f"\n{'='*50}")
                print(f"Processing {episode_key} - Run {run_num}")
                print(f"{'='*50}")
                
                # Get rows for this specific run from emotion data
                run_identifier = f"{episode_key}R{run_num:02d}"
                run_mask = emotion_df_trimmed['run'] == run_identifier
                
                if not run_mask.any():
                    print(f"    Warning: No data found for run {run_identifier}")
                    continue
                
                # Extract data for this run using the mask
                run_emotion_data = emotion_df_trimmed[run_mask].reset_index(drop=True)
                run_caps_data = caps_df_trimmed[run_mask].reset_index(drop=True)
                
                run_valence = run_emotion_data['valence']
                run_arousal = run_emotion_data['arousal']
                
                print(f"    Found {len(run_emotion_data)} clips for run {run_identifier}")
                print(f"    Run {run_num} data: {len(run_valence)} valence, {len(run_arousal)} arousal, {len(run_caps_data)} CAP timepoints")
                
                # Plot each CAP for this run
                for cap_name in run_caps_data.columns:
                    cap_timeseries = run_caps_data[cap_name]
                    
                    # Calculate correlations
                    val_corr, val_p = spearmanr(cap_timeseries, run_valence)
                    aro_corr, aro_p = spearmanr(cap_timeseries, run_arousal)

                    # Only plot if either |rho| exceeds threshold (treat NaN as not passing)
                    if not (passes_threshold(val_corr) or passes_threshold(aro_corr)):
                        # Format safely even if NaN
                        val_str = "nan" if not np.isfinite(val_corr) else f"{val_corr:.3f}"
                        aro_str = "nan" if not np.isfinite(aro_corr) else f"{aro_corr:.3f}"
                        print(f"    Skipping {cap_name} (|ρ_val|={val_str}, |ρ_aro|={aro_str} ≤ {PLOT_ABS_R_THRESHOLD})")
                        continue

                    # Create the plot
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
                    
                    # Define colors
                    colors = ['#CF9397', "#606933", '#E5D28E']
                    
                    # Valence plot
                    ax1.axhspan(1, 4, alpha=0.1, color='red', zorder=0)
                    ax1.axhspan(4, 7, alpha=0.1, color='green', zorder=0)
                    
                    # Plot valence ratings
                    ax1.plot(range(len(run_valence)), run_valence, color=colors[0], marker='o', markersize=4, 
                             linewidth=2, alpha=0.8, label='Valence Ratings')
                    
                    # Plot CAP weighted timeseries on secondary y-axis
                    ax1_twin = ax1.twinx()
                    ax1_twin.plot(range(len(cap_timeseries)), cap_timeseries, color=colors[1], marker='s', markersize=4, 
                                  linewidth=2, alpha=0.8, label=f'{cap_name} Weighted')
                    
                    ax1.set_title(f'{sub_id} {episode_key} Run {run_num} - Valence vs {cap_name} Weighted', fontsize=16, fontweight='bold')
                    ax1.set_xlabel(f'Clips ({len(run_valence)} clips)', fontsize=12)
                    ax1.set_ylabel('Valence Rating', fontsize=12, color=colors[0])
                    ax1_twin.set_ylabel(f'{cap_name} Weighted', fontsize=12, color=colors[1])
                    ax1.set_ylim(0, 8)
                    
                    # Add reference lines for valence
                    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=2)
                    ax1.axhline(y=4, color='gray', linestyle='--', alpha=0.7, linewidth=2)
                    ax1.axhline(y=7, color='green', linestyle='--', alpha=0.7, linewidth=2)
                    
                    # Add text labels
                    x_right = ax1.get_xlim()[1]
                    ax1.text(x_right, 1, 'Negative Valence', ha='left', va='bottom', fontsize=10, color='red')
                    ax1.text(x_right, 4, 'Neutral', ha='left', va='bottom', fontsize=10, color='gray')
                    ax1.text(x_right, 7, 'Positive Valence', ha='left', va='bottom', fontsize=10, color='green')
                    
                    # Combined legend
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax1_twin.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                    
                    # Add correlation results (safe formatting)
                    val_p_str = "nan" if not np.isfinite(val_p) else (f"{val_p:.3f}" if val_p >= 0.001 else "<0.001")
                    val_corr_str = "nan" if not np.isfinite(val_corr) else f"{val_corr:.3f}"
                    ax1.text(0.02, 0.98, f"Spearman ρ: {val_corr_str}\np: {val_p_str}",
                             transform=ax1.transAxes,
                             fontsize=11, verticalalignment='top', horizontalalignment='left',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

                    # Arousal plot  
                    ax2.plot(range(len(run_arousal)), run_arousal, color=colors[0], marker='o', markersize=4, 
                             linewidth=2, alpha=0.8, label='Arousal Ratings')
                    
                    # Plot CAP weighted timeseries on secondary y-axis
                    ax2_twin = ax2.twinx()
                    ax2_twin.plot(range(len(cap_timeseries)), cap_timeseries, color=colors[1], marker='s', markersize=4, 
                                  linewidth=2, alpha=0.8, label=f'{cap_name} Weighted')
                    
                    ax2.set_title(f'{sub_id} {episode_key} Run {run_num} - Arousal vs {cap_name} Weighted', fontsize=16, fontweight='bold')
                    ax2.set_xlabel(f'Clips ({len(run_arousal)} clips)', fontsize=12)
                    ax2.set_ylabel('Arousal Rating', fontsize=12, color=colors[0])
                    ax2_twin.set_ylabel(f'{cap_name} Weighted', fontsize=12, color=colors[1])
                    ax2.set_ylim(0, 8)
                    
                    # Add reference lines for arousal
                    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.7, linewidth=2)
                    ax2.axhline(y=7, color='black', linestyle='--', alpha=0.7, linewidth=2)
                    
                    # Add text labels
                    x_right = ax2.get_xlim()[1]
                    ax2.text(x_right, 1, 'Low Arousal', ha='left', va='bottom', fontsize=10, color='black')
                    ax2.text(x_right, 7, 'High Arousal', ha='left', va='bottom', fontsize=10, color='black')
                    
                    # Combined legend
                    lines1, labels1 = ax2.get_legend_handles_labels()
                    lines2, labels2 = ax2_twin.get_legend_handles_labels()
                    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                    
                    # Add correlation results (safe formatting)
                    aro_p_str = "nan" if not np.isfinite(aro_p) else (f"{aro_p:.3f}" if aro_p >= 0.001 else "<0.001")
                    aro_corr_str = "nan" if not np.isfinite(aro_corr) else f"{aro_corr:.3f}"
                    ax2.text(0.02, 0.98, f"Spearman ρ: {aro_corr_str}\np: {aro_p_str}",
                             transform=ax2.transAxes,
                             fontsize=11, verticalalignment='top', horizontalalignment='left',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

                    plt.tight_layout()
                    
                    # Save figure with run-specific filename
                    figure_filename = f"{sub_id}_{episode_key}_run{run_num}_{cap_name}_emotion_correlation.png"
                    figure_path = FIGURES_DIR / figure_filename
                    fig.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
                    print(f"    Figure saved: {figure_path}")
                    
                    plt.close(fig)  # Close figure to save memory
                    
                    print(f"    Run {run_num} {cap_name}: Valence r={val_corr_str} (p={val_p_str}), Arousal r={aro_corr_str} (p={aro_p_str})")

    print(f"\n{'='*80}")
    print("ALL RUN-LEVEL PLOTTING COMPLETE")
    print(f"{'='*80}")
    print("All emotion correlation analysis completed successfully!")

if __name__ == "__main__":
    main()