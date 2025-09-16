import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

# For ICC calculations
try:
    import pingouin as pg

    HAS_PINGOUIN = True
except ImportError:
    print("Warning: pingouin not available. Install with: " "pip install pingouin")
    HAS_PINGOUIN = False

# Set non-interactive backend for matplotlib
plt.switch_backend("Agg")


def generate_multi_session_data(
    n_subjects=3, n_sessions=2, n_runs_per_session=3, n_caps=5, n_episodes=4
):
    """
    Generate simulated multi-session CAP and emotion data.

    This simulates the realistic scenario where:
    - Multiple subjects participate in multiple sessions
    - Each session has multiple runs (different episodes)
    - Each episode has different arousal/valence profiles
    - CAP metrics vary across runs but show some consistency within subjects
    """
    print("Generating multi-session data:")
    print(
        f"  {n_subjects} subjects × {n_sessions} sessions × "
        f"{n_runs_per_session} runs"
    )
    print(f"  {n_caps} CAPs, {n_episodes} different episodes")

    # Subject IDs
    subject_ids = ["sub-Blossom", "sub-Bubbles", "sub-Buttercup"][:n_subjects]

    # Episode names (different Stranger Things episodes)
    episode_names = [
        "episode-1-pilot",
        "episode-2-weirdo",
        "episode-3-holly",
        "episode-4-body",
    ][:n_episodes]

    all_data = []

    # Set random seed for reproducible individual differences
    np.random.seed(42)

    # Create subject-specific baseline patterns (individual differences)
    subject_patterns = {}
    for subj_id in subject_ids:
        subject_patterns[subj_id] = {
            "valence_baseline": np.random.uniform(3, 6),
            "arousal_baseline": np.random.uniform(2, 5),
            "cap_preferences": np.random.uniform(0.8, 1.2, n_caps),
        }

    # Generate data for each subject, session, and run
    for subj_idx, subject_id in enumerate(subject_ids):
        subj_pattern = subject_patterns[subject_id]

        for session in range(1, n_sessions + 1):
            for run in range(1, n_runs_per_session + 1):
                # Select episode for this run (different episodes across runs)
                episode_idx = (run - 1) % len(episode_names)
                episode = episode_names[episode_idx]

                run_id = f"{subject_id}_ses-{session:02d}_run-{run:02d}_" f"{episode}"

                # Generate episode-specific emotion ratings
                # Different episodes have different emotional profiles
                episode_effects = {
                    "episode-1-pilot": {"val_boost": 0.5, "ar_boost": -0.3},
                    "episode-2-weirdo": {"val_boost": -0.8, "ar_boost": 1.2},
                    "episode-3-holly": {"val_boost": 0.2, "ar_boost": 0.8},
                    "episode-4-body": {"val_boost": -1.0, "ar_boost": 1.5},
                }

                episode_effect = episode_effects.get(
                    episode, {"val_boost": 0, "ar_boost": 0}
                )

                # Subject's emotional response to this episode
                valence_score = (
                    subj_pattern["valence_baseline"]
                    + episode_effect["val_boost"]
                    + np.random.normal(0, 0.5)
                )

                arousal_score = (
                    subj_pattern["arousal_baseline"]
                    + episode_effect["ar_boost"]
                    + np.random.normal(0, 0.5)
                )

                # Clip to valid range
                valence_score = np.clip(valence_score, 1, 7)
                arousal_score = np.clip(arousal_score, 1, 7)

                # Generate CAP metrics with some relationship to emotions
                cap_metrics = {
                    "run": run_id,
                    "subject_id": subject_id,
                    "session": session,
                    "run_number": run,
                    "episode": episode,
                    "valence_rating": valence_score,
                    "arousal_rating": arousal_score,
                    "total_timepoints": np.random.randint(250, 350),
                }

                # Generate CAP metrics with some true relationship to emotions
                for cap_idx in range(1, n_caps + 1):
                    cap_name = f"CAP_{cap_idx}"

                    # Create some caps that correlate with valence, others with arousal
                    if cap_idx <= 2:  # CAPs 1-2 relate to valence
                        emotion_influence = (
                            valence_score - 4
                        ) * 2  # Center around 4, scale
                    else:  # CAPs 3-5 relate to arousal
                        emotion_influence = (
                            arousal_score - 3
                        ) * 1.5  # Center around 3, scale

                    # Subject-specific CAP patterns
                    subject_multiplier = subj_pattern["cap_preferences"][cap_idx - 1]

                    # Session consistency (session 2 should be similar to session 1)
                    session_consistency = (
                        1.0 if session == 1 else np.random.uniform(0.85, 1.15)
                    )

                    # Base frequency with emotion and individual influences
                    base_frequency = 15 + emotion_influence + np.random.normal(0, 3)
                    frequency = (
                        base_frequency * subject_multiplier * session_consistency
                    )
                    frequency = np.clip(frequency, 5, 35)  # Reasonable range

                    cap_metrics[f"{cap_name}_frequency_pct"] = frequency

                    # Dwell time (somewhat independent of frequency)
                    dwell_time = np.random.uniform(2, 6) + (emotion_influence * 0.1)
                    cap_metrics[f"{cap_name}_avg_dwell_time"] = max(1.5, dwell_time)

                    # Episodes and transitions
                    episodes = max(1, int(frequency * 0.4 + np.random.poisson(2)))
                    cap_metrics[f"{cap_name}_n_episodes"] = episodes

                    transitions = np.random.poisson(episodes * 0.7)
                    cap_metrics[f"{cap_name}_transitions_out"] = transitions

                    if episodes > 0:
                        trans_rate = transitions / episodes
                    else:
                        trans_rate = 0
                    cap_metrics[f"{cap_name}_transition_rate"] = trans_rate

                all_data.append(cap_metrics)

    df = pd.DataFrame(all_data)

    print(f"Generated {len(df)} observations")
    print(f"Sessions: {df['session'].unique()}")
    print(f"Episodes: {df['episode'].unique()}")
    print(f"Subjects: {df['subject_id'].unique()}")

    return df


def calculate_session_correlations(
    data, emotion_cols=["valence_rating", "arousal_rating"]
):
    """Calculate CAP-emotion correlations within each session."""
    print("Calculating CAP-emotion correlations by session...")

    # Get CAP metric columns
    cap_cols = [
        col
        for col in data.columns
        if "CAP_" in col
        and col.endswith(
            (
                "_frequency_pct",
                "_avg_dwell_time",
                "_transitions_out",
                "_transition_rate",
            )
        )
    ]

    correlation_results = []

    # Calculate correlations for each session
    for session in data["session"].unique():
        session_data = data[data["session"] == session]
        print(f"  Session {session}: {len(session_data)} observations")

        for emotion_col in emotion_cols:
            for cap_col in cap_cols:
                # Calculate correlation
                valid_data = session_data.dropna(subset=[emotion_col, cap_col])

                if len(valid_data) >= 3:
                    r, p = pearsonr(valid_data[emotion_col], valid_data[cap_col])

                    # Extract CAP and metric info
                    cap_parts = cap_col.split("_")
                    cap_name = f"{cap_parts[0]}_{cap_parts[1]}"
                    metric_name = "_".join(cap_parts[2:])

                    correlation_results.append(
                        {
                            "session": session,
                            "emotion": emotion_col.replace("_rating", ""),
                            "cap": cap_name,
                            "metric": metric_name,
                            "correlation": r,
                            "p_value": p,
                            "n": len(valid_data),
                        }
                    )

    return pd.DataFrame(correlation_results)


def calculate_replication_icc(correlation_results):
    """Calculate ICC between sessions for replication analysis."""
    print("Calculating replication ICC between sessions...")

    if not HAS_PINGOUIN:
        print("Pingouin not available, using simple correlation for replication")
        return calculate_simple_replication_correlation(correlation_results)

    icc_results = []

    # Get unique combinations of emotion, cap, and metric
    unique_combinations = (
        correlation_results.groupby(["emotion", "cap", "metric"]).size().reset_index()
    )

    for _, combo in unique_combinations.iterrows():
        # Get data for this combination across sessions
        combo_data = correlation_results[
            (correlation_results["emotion"] == combo["emotion"])
            & (correlation_results["cap"] == combo["cap"])
            & (correlation_results["metric"] == combo["metric"])
        ]

        if len(combo_data) >= 2:  # Need at least 2 sessions
            # Prepare data for ICC
            # Create subject-session pairs (for ICC we treat each correlation as from a "subject")
            icc_data = []
            for _, row in combo_data.iterrows():
                icc_data.append(
                    {
                        "measurement": row["correlation"],
                        "session": row["session"],
                        "combination": f"{combo['emotion']}_{combo['cap']}_{combo['metric']}",
                    }
                )

            icc_df = pd.DataFrame(icc_data)

            # Add artificial subject IDs for ICC calculation
            # (In real data, you'd have multiple subjects per session)
            icc_df["subject"] = range(len(icc_df))

            try:
                # Calculate ICC(3,1) for consistency across sessions
                icc_result = pg.intraclass_corr(
                    data=icc_df,
                    targets="subject",
                    raters="session",
                    ratings="measurement",
                )

                # Get ICC(3,1) - consistency, single measurement
                icc_31 = icc_result[icc_result["Type"] == "ICC3"]["ICC"].iloc[0]
                icc_31_ci_low = icc_result[icc_result["Type"] == "ICC3"]["CI95%"].iloc[
                    0
                ][0]
                icc_31_ci_high = icc_result[icc_result["Type"] == "ICC3"]["CI95%"].iloc[
                    0
                ][1]

                icc_results.append(
                    {
                        "emotion": combo["emotion"],
                        "cap": combo["cap"],
                        "metric": combo["metric"],
                        "icc": icc_31,
                        "icc_ci_low": icc_31_ci_low,
                        "icc_ci_high": icc_31_ci_high,
                        "n_sessions": len(combo_data),
                    }
                )

            except Exception as e:
                print(
                    f"ICC calculation failed for {combo['emotion']}_{combo['cap']}_{combo['metric']}: {e}"
                )
                continue

    return pd.DataFrame(icc_results)


def calculate_simple_replication_correlation(correlation_results):
    """Simple correlation between session 1 and session 2 correlations."""
    print("Using simple correlation for replication analysis...")

    # Pivot to get session 1 vs session 2 correlations
    pivot_data = correlation_results.pivot_table(
        index=["emotion", "cap", "metric"],
        columns="session",
        values="correlation",
        fill_value=np.nan,
    )

    # Calculate correlation between session 1 and session 2
    if 1 in pivot_data.columns and 2 in pivot_data.columns:
        valid_data = pivot_data.dropna(subset=[1, 2])

        if len(valid_data) >= 3:
            replication_r, replication_p = pearsonr(valid_data[1], valid_data[2])

            print(
                f"Replication correlation between sessions: r = {replication_r:.3f}, p = {replication_p:.3f}"
            )

            return {
                "replication_correlation": replication_r,
                "replication_p_value": replication_p,
                "n_comparisons": len(valid_data),
            }

    return None


def build_emotion_prediction_models(
    data, target_emotions=["valence_rating", "arousal_rating"]
):
    """Build models to predict emotions from CAP metrics."""
    print("Building emotion prediction models...")

    # Get CAP feature columns
    feature_cols = [
        col
        for col in data.columns
        if "CAP_" in col
        and col.endswith(
            (
                "_frequency_pct",
                "_avg_dwell_time",
                "_transitions_out",
                "_transition_rate",
            )
        )
    ]

    prediction_results = []

    for target_emotion in target_emotions:
        print(f"\n--- Predicting {target_emotion} ---")

        # Separate sessions for train/test
        train_data = data[data["session"] == 1]
        test_data = data[data["session"] == 2]

        if len(train_data) == 0 or len(test_data) == 0:
            print(f"Insufficient data for train/test split for {target_emotion}")
            continue

        # Prepare features and target
        X_train = train_data[feature_cols].fillna(0)
        y_train = train_data[target_emotion].fillna(train_data[target_emotion].mean())

        X_test = test_data[feature_cols].fillna(0)
        y_test = test_data[target_emotion].fillna(test_data[target_emotion].mean())

        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Features: {len(feature_cols)}")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Test multiple models
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=0.1),
            "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
        }

        for model_name, model in models.items():
            try:
                # Train model
                if "Forest" in model_name:
                    model.fit(X_train, y_train)  # Random Forest doesn't need scaling
                    y_pred = model.predict(X_test)
                    feature_importance = model.feature_importances_
                else:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    if hasattr(model, "coef_"):
                        feature_importance = np.abs(model.coef_)
                    else:
                        feature_importance = np.zeros(len(feature_cols))

                # Calculate performance metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # Correlation between predicted and actual
                pred_corr, pred_p = pearsonr(y_test, y_pred)

                print(
                    f"{model_name}: R² = {r2:.3f}, RMSE = {rmse:.3f}, r = {pred_corr:.3f}"
                )

                # Get top features
                feature_importance_df = pd.DataFrame(
                    {"feature": feature_cols, "importance": feature_importance}
                ).sort_values("importance", ascending=False)

                top_features = feature_importance_df.head(5)["feature"].tolist()

                prediction_results.append(
                    {
                        "emotion": target_emotion,
                        "model": model_name,
                        "r2_score": r2,
                        "rmse": rmse,
                        "mae": mae,
                        "prediction_correlation": pred_corr,
                        "prediction_p_value": pred_p,
                        "top_features": ", ".join(top_features),
                        "n_train": len(X_train),
                        "n_test": len(X_test),
                    }
                )

            except Exception as e:
                print(f"Model {model_name} failed for {target_emotion}: {e}")
                continue

    return pd.DataFrame(prediction_results)


def plot_replication_results(correlation_results, save_path=None):
    """Plot replication analysis results."""
    print("Plotting replication results...")

    # Create pivot table for heatmap
    pivot_data = correlation_results.pivot_table(
        index=["cap", "metric"],
        columns=["emotion", "session"],
        values="correlation",
        fill_value=0,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Session 1 vs Session 2 correlations
    for i, emotion in enumerate(["valence", "arousal"]):
        if (emotion, 1) in pivot_data.columns and (emotion, 2) in pivot_data.columns:
            session1_data = pivot_data[(emotion, 1)]
            session2_data = pivot_data[(emotion, 2)]

            ax = ax1 if i == 0 else ax2

            # Scatter plot
            ax.scatter(session1_data, session2_data, alpha=0.7, s=50)

            # Add diagonal line
            min_val = min(session1_data.min(), session2_data.min())
            max_val = max(session1_data.max(), session2_data.max())
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                "r--",
                alpha=0.7,
                label="Perfect replication",
            )

            # Calculate and display correlation
            valid_mask = ~(session1_data.isna() | session2_data.isna())
            if valid_mask.sum() >= 3:
                r, p = pearsonr(session1_data[valid_mask], session2_data[valid_mask])
                ax.text(
                    0.05,
                    0.95,
                    f"r = {r:.3f}, p = {p:.3f}",
                    transform=ax.transAxes,
                    fontsize=12,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

            ax.set_xlabel(f"Session 1 Correlations ({emotion.title()})")
            ax.set_ylabel(f"Session 2 Correlations ({emotion.title()})")
            ax.set_title(f"{emotion.title()}-CAP Correlation Replication")
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.suptitle(
        "CAP-Emotion Correlation Replication Across Sessions",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Replication plot saved to: {save_path}")

    plt.close()


def plot_prediction_results(prediction_results, save_path=None):
    """Plot prediction model results."""
    print("Plotting prediction results...")

    if len(prediction_results) == 0:
        print("No prediction results to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    emotions = prediction_results["emotion"].unique()

    for i, emotion in enumerate(emotions):
        emotion_data = prediction_results[prediction_results["emotion"] == emotion]

        # R² scores
        ax1 = axes[i, 0]
        models = emotion_data["model"]
        r2_scores = emotion_data["r2_score"]
        colors = ["skyblue" if r2 > 0 else "lightcoral" for r2 in r2_scores]

        bars1 = ax1.bar(models, r2_scores, color=colors, alpha=0.7)
        ax1.set_title(f'{emotion.replace("_rating", "").title()} - R² Scores')
        ax1.set_ylabel("R² Score")
        ax1.tick_params(axis="x", rotation=45)
        ax1.axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # Add value labels on bars
        for bar, r2 in zip(bars1, r2_scores):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{r2:.3f}",
                ha="center",
                va="bottom",
            )

        # Prediction correlations
        ax2 = axes[i, 1]
        pred_corrs = emotion_data["prediction_correlation"]
        colors2 = [
            "darkgreen" if corr > 0.3 else "orange" if corr > 0 else "red"
            for corr in pred_corrs
        ]

        bars2 = ax2.bar(models, pred_corrs, color=colors2, alpha=0.7)
        ax2.set_title(
            f'{emotion.replace("_rating", "").title()} - Prediction Correlations'
        )
        ax2.set_ylabel("Correlation (r)")
        ax2.tick_params(axis="x", rotation=45)
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # Add value labels on bars
        for bar, corr in zip(bars2, pred_corrs):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{corr:.3f}",
                ha="center",
                va="bottom",
            )

    plt.suptitle("Emotion Prediction Model Performance", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Prediction plot saved to: {save_path}")

    plt.close()


def generate_comprehensive_report(
    correlation_results,
    replication_results,
    prediction_results,
    output_file="replication_prediction_analysis_report.txt",
):
    """Generate comprehensive analysis report."""
    print("Generating comprehensive analysis report...")

    with open(output_file, "w") as f:
        f.write("REPLICATION & PREDICTION ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write(
            "OBJECTIVE: Test within-person replication and prediction of CAP-emotion relationships\n"
        )
        f.write("DESIGN: Multi-session design with different episodes per run\n")
        f.write(
            "ANALYSIS: Session 1 vs Session 2 replication + Session 1→Session 2 prediction\n\n"
        )

        # 1. Replication Analysis
        f.write("1. REPLICATION ANALYSIS\n")
        f.write("-" * 30 + "\n")

        if len(correlation_results) > 0:
            # Session-wise correlation summary
            session_summary = (
                correlation_results.groupby("session")
                .agg(
                    {
                        "correlation": ["mean", "std", "count"],
                        "p_value": lambda x: (x < 0.05).sum(),
                    }
                )
                .round(3)
            )

            f.write("Session-wise CAP-emotion correlation summary:\n")
            f.write(
                f"  Session 1: Mean r = {session_summary.loc[1, ('correlation', 'mean')]:.3f} "
                f"(SD = {session_summary.loc[1, ('correlation', 'std')]:.3f})\n"
            )
            f.write(
                f"  Session 2: Mean r = {session_summary.loc[2, ('correlation', 'mean')]:.3f} "
                f"(SD = {session_summary.loc[2, ('correlation', 'std')]:.3f})\n"
            )

            f.write(
                f"  Significant correlations - Session 1: {session_summary.loc[1, ('p_value', '<lambda>')]}\n"
            )
            f.write(
                f"  Significant correlations - Session 2: {session_summary.loc[2, ('p_value', '<lambda>')]}\n\n"
            )

            # Cross-session replication
            if isinstance(replication_results, dict):
                f.write("Cross-session replication:\n")
                f.write(
                    f"  Correlation between Session 1 and Session 2 effect sizes: "
                    f"r = {replication_results['replication_correlation']:.3f}, "
                    f"p = {replication_results['replication_p_value']:.3f}\n"
                )
                f.write(
                    f"  Number of CAP-emotion associations compared: {replication_results['n_comparisons']}\n\n"
                )
            elif len(replication_results) > 0:
                f.write("ICC-based replication results:\n")
                mean_icc = replication_results["icc"].mean()
                f.write(f"  Mean ICC across CAP-emotion associations: {mean_icc:.3f}\n")
                reliable_associations = (replication_results["icc"] > 0.5).sum()
                f.write(
                    f"  Associations with good reliability (ICC > 0.5): {reliable_associations}/{len(replication_results)}\n\n"
                )
        else:
            f.write("No replication results available.\n\n")

        # 2. Prediction Analysis
        f.write("2. PREDICTION ANALYSIS\n")
        f.write("-" * 30 + "\n")

        if len(prediction_results) > 0:
            f.write("Model performance summary (Session 1 → Session 2 prediction):\n\n")

            for emotion in prediction_results["emotion"].unique():
                emotion_clean = emotion.replace("_rating", "").title()
                emotion_data = prediction_results[
                    prediction_results["emotion"] == emotion
                ]

                f.write(f"{emotion_clean} Prediction:\n")

                # Find best model
                best_model = emotion_data.loc[emotion_data["r2_score"].idxmax()]

                for _, model_result in emotion_data.iterrows():
                    model_name = model_result["model"]
                    r2 = model_result["r2_score"]
                    rmse = model_result["rmse"]
                    corr = model_result["prediction_correlation"]

                    best_indicator = (
                        " (BEST)" if model_name == best_model["model"] else ""
                    )

                    f.write(
                        f"  {model_name}: R² = {r2:.3f}, RMSE = {rmse:.3f}, "
                        f"r = {corr:.3f}{best_indicator}\n"
                    )

                f.write(f"  Top features: {best_model['top_features']}\n\n")

            # Overall prediction success
            best_r2_per_emotion = prediction_results.groupby("emotion")[
                "r2_score"
            ].max()
            successful_predictions = (best_r2_per_emotion > 0.1).sum()

            f.write(f"Prediction Summary:\n")
            f.write(
                f"  Emotions with successful prediction (R² > 0.1): {successful_predictions}/{len(best_r2_per_emotion)}\n"
            )
            f.write(
                f"  Mean best R² across emotions: {best_r2_per_emotion.mean():.3f}\n"
            )
            f.write(f"  Max R² achieved: {best_r2_per_emotion.max():.3f}\n\n")
        else:
            f.write("No prediction results available.\n\n")

        # 3. Interpretation
        f.write("3. INTERPRETATION & IMPLICATIONS\n")
        f.write("-" * 40 + "\n")
        f.write("REPLICATION FINDINGS:\n")
        if isinstance(replication_results, dict):
            repl_r = replication_results["replication_correlation"]
            if repl_r > 0.7:
                f.write(
                    "- EXCELLENT replication: CAP-emotion associations highly consistent across sessions\n"
                )
            elif repl_r > 0.5:
                f.write(
                    "- GOOD replication: CAP-emotion associations moderately consistent across sessions\n"
                )
            elif repl_r > 0.3:
                f.write(
                    "- FAIR replication: Some consistency in CAP-emotion associations across sessions\n"
                )
            else:
                f.write(
                    "- POOR replication: Low consistency in CAP-emotion associations across sessions\n"
                )

        f.write("\nPREDICTION FINDINGS:\n")
        if len(prediction_results) > 0:
            max_r2 = prediction_results["r2_score"].max()
            if max_r2 > 0.3:
                f.write(
                    "- STRONG predictive power: CAPs can predict emotions across sessions\n"
                )
            elif max_r2 > 0.1:
                f.write(
                    "- MODERATE predictive power: CAPs show some predictive validity\n"
                )
            else:
                f.write(
                    "- WEAK predictive power: Limited ability to predict emotions from CAPs\n"
                )

        f.write("\nCONCLUSIONS:\n")
        f.write(
            "- Within-person CAP-emotion relationships tested across multiple sessions\n"
        )
        f.write(
            "- Results inform reliability and generalizability of CAP-based emotion models\n"
        )
        f.write("- Framework applicable to real multi-session neuroimaging studies\n")

        f.write("\nAnalysis completed.\n")

    print(f"Comprehensive report saved to: {output_file}")


def main():
    """Main replication and prediction analysis."""
    print("Starting Replication & Prediction Analysis...")
    print("=" * 50)

    # 1. Generate multi-session data
    data = generate_multi_session_data(
        n_subjects=3,
        n_sessions=2,
        n_runs_per_session=4,  # More runs for better statistical power
        n_caps=5,
        n_episodes=4,
    )

    # Save the generated dataset
    data.to_csv("multi_session_cap_emotion_data.tsv", sep="\t", index=False)
    print(f"Multi-session dataset saved to: multi_session_cap_emotion_data.tsv")

    # 2. Calculate session-wise correlations
    correlation_results = calculate_session_correlations(data)

    # 3. Calculate replication metrics
    replication_results = calculate_replication_icc(correlation_results)

    # 4. Build prediction models
    prediction_results = build_emotion_prediction_models(data)

    # 5. Generate visualizations
    plot_replication_results(correlation_results, "replication_analysis.png")
    plot_prediction_results(prediction_results, "prediction_analysis.png")

    # 6. Save detailed results
    correlation_results.to_csv("session_correlations.tsv", sep="\t", index=False)
    print("Session correlations saved to: session_correlations.tsv")

    if isinstance(replication_results, pd.DataFrame) and len(replication_results) > 0:
        replication_results.to_csv("replication_icc_results.tsv", sep="\t", index=False)
        print("Replication ICC results saved to: replication_icc_results.tsv")

    if len(prediction_results) > 0:
        prediction_results.to_csv("prediction_model_results.tsv", sep="\t", index=False)
        print("Prediction results saved to: prediction_model_results.tsv")

    # 7. Generate comprehensive report
    generate_comprehensive_report(
        correlation_results, replication_results, prediction_results
    )

    print("\nReplication & Prediction Analysis completed successfully!")
    print("=" * 50)

    # Summary statistics
    print("\nSUMMARY:")
    print(
        f"- Generated {len(data)} observations across {data['session'].nunique()} sessions"
    )
    print(f"- Calculated {len(correlation_results)} CAP-emotion correlations")
    print(f"- Tested {len(prediction_results)} prediction models")
    print("- Framework ready for real multi-session neuroimaging data")

    return {
        "data": data,
        "correlations": correlation_results,
        "replication": replication_results,
        "prediction": prediction_results,
    }


if __name__ == "__main__":
    main()
