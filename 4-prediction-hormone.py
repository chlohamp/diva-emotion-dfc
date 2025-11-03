import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# Statsmodels for mixed-effects models
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import mixedlm
    from statsmodels.stats.diagnostic import acorr_ljungbox
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available. Install with: pip install statsmodels")


plt.switch_backend("Agg")


# ------------------------------
# Data loading and preparation
# ------------------------------

PARTICIPANTS = ["sub-Blossom", "sub-Bubbles", "sub-Buttercup"]
CONTRACEPTIVE_STATUS = {
    "sub-Blossom": "OCP",
    "sub-Bubbles": "Natural",
    "sub-Buttercup": "OCP",
}


def create_simulated_emotion_timeseries(participant_id: str, n_timepoints: int = 240) -> pd.DataFrame:
    np.random.seed(hash(participant_id) % 10000)
    t = np.linspace(0, 6 * np.pi, n_timepoints)
    val = 0.7 * np.sin(0.35 * t) + 0.3 * np.sin(0.95 * t) + np.random.normal(0, 0.3, n_timepoints)
    aro = 0.8 * np.cos(0.4 * t) + 0.2 * np.sin(1.1 * t) + np.random.normal(0, 0.3, n_timepoints)
    return pd.DataFrame(
        {
            "onset": np.arange(0, n_timepoints * 1.5, 1.5),
            "valence_aggregated": stats.zscore(val),
            "arousal_aggregated": stats.zscore(aro),
            "valence_reliable": True,
            "arousal_reliable": True,
        }
    )


def load_participant_data() -> dict:
    """Load or simulate per-participant runs with CAP activations, emotion, motion, session/run ids."""
    data = {}
    for pid in PARTICIPANTS:
        # Attempt to read aggregated emotion series (from 1a)
        try:
            emo_df = pd.read_csv(
                "derivatives/caps/interrater/aggregated_emotion_timeseries.tsv", sep="\t"
            )
        except FileNotFoundError:
            emo_df = create_simulated_emotion_timeseries(pid, 240)

        # Determine sessions and runs (match earlier script conventions)
        if pid == "sub-Blossom":
            n_sessions, runs_per_session = 2, 3
        else:
            n_sessions, runs_per_session = 4, 3
        n_runs = n_sessions * runs_per_session
        base_len = len(emo_df) // n_runs if len(emo_df) >= n_runs else 60

        # CAP masks (try to load else simulate)
        try:
            cap_file = f"derivatives/caps/cap-analysis/{pid.replace('-', '_')}/cap_emotion_correlations.tsv"
            cap_df = pd.read_csv(cap_file, sep="\t")
            cap_masks = cap_df["cap_mask"].unique().tolist()
        except Exception:
            cap_masks = [f"CAP_{i}_{sign}" for i in range(1, 6) for sign in ["positive", "negative"]]

        runs = []
        for s in range(1, n_sessions + 1):
            for r in range(1, runs_per_session + 1):
                run_id = f"ses-{s:02d}_run-{r:02d}"
                # slice or sample emotion
                if len(emo_df) >= base_len:
                    start = ((s - 1) * runs_per_session + (r - 1)) * base_len
                    start = min(start, max(0, len(emo_df) - base_len))
                    emo_sub = emo_df.iloc[start : start + base_len].reset_index(drop=True)
                else:
                    emo_sub = emo_df.sample(n=base_len, replace=True).reset_index(drop=True)

                df = pd.DataFrame(
                    {
                        "participant_id": pid,
                        "session": s,
                        "run_in_session": r,
                        "run_id": run_id,
                        "timepoint": np.arange(base_len),
                        "time_seconds": np.arange(base_len) * 1.5,
                        "valence_zscore": stats.zscore(emo_sub["valence_aggregated"].values),
                        "arousal_zscore": stats.zscore(emo_sub["arousal_aggregated"].values),
                        "valence_reliable": True,
                        "arousal_reliable": True,
                    }
                )

                # CAP activations with mild temporal dependency
                rng = np.random.default_rng(abs(hash(f"{pid}-{run_id}")) % (2**32))
                for cap in cap_masks:
                    x = rng.normal(size=base_len)
                    for k in range(1, base_len):
                        x[k] += 0.35 * x[k - 1]
                    # couple a bit with emotions
                    if "positive" in cap:
                        x = x + 0.2 * df["valence_zscore"].values + 0.2 * df["arousal_zscore"].values
                    df[f"{cap}_activation"] = stats.zscore(x)

                # motion
                fd = rng.exponential(0.1, base_len)
                df["framewise_displacement"] = np.convolve(fd, np.ones(3) / 3, mode="same")
                runs.append(df)

        data[pid] = runs

    return data


def load_hormone_levels() -> pd.DataFrame:
    """Load hormone concentration per participant-session.

    Expected columns if file exists:
    participant_id, session, hormone, hormone_unit

    If no file is found, simulate by contraceptive status:
    - OCP: mean ~ 50, sd ~ 5 (stable)
    - Natural: mean ~ 100, sd ~ 30 (variable across sessions)
    Values are arbitrary units for illustration; replace with real assays if available.
    """
    possible_paths = [
        Path("derivatives/caps/cap-analysis/hormones.tsv"),
        Path("derivatives/simulated/hormones.tsv"),
    ]
    for p in possible_paths:
        if p.exists():
            df = pd.read_csv(p, sep="\t") if p.suffix == ".tsv" else pd.read_csv(p)
            # normalize columns
            cols = {c.lower(): c for c in df.columns}
            # ensure required
            req = ["participant_id", "session", "hormone"]
            if all(any(k == c.lower() for c in df.columns) for k in req):
                # harmonize names
                df = df.rename(columns={cols.get("participant_id", "participant_id"): "participant_id",
                                        cols.get("session", "session"): "session",
                                        cols.get("hormone", "hormone"): "hormone"})
                return df[["participant_id", "session", "hormone"]]

    # simulate
    records = []
    for pid in PARTICIPANTS:
        if pid == "sub-Blossom":
            n_sessions = 2
        else:
            n_sessions = 4
        status = CONTRACEPTIVE_STATUS[pid]
        rng = np.random.default_rng(abs(hash(pid)) % (2**32))
        if status == "OCP":
            base, sd = 50, 6
        else:
            base, sd = 100, 28
        for s in range(1, n_sessions + 1):
            val = float(np.clip(rng.normal(base, sd), 5, None))
            records.append({"participant_id": pid, "session": s, "hormone": val})
    return pd.DataFrame(records)


def prepare_model_data(all_run_data: pd.DataFrame, emotion: str) -> pd.DataFrame:
    cap_cols = [c for c in all_run_data.columns if c.endswith("_activation")]
    pos = [c for c in cap_cols if "positive" in c]
    neg = [c for c in cap_cols if "negative" in c]
    df = all_run_data.copy()
    df["CAP_positive"] = all_run_data[pos].mean(axis=1) if len(pos) else 0.0
    df["CAP_negative"] = all_run_data[neg].mean(axis=1) if len(neg) else 0.0
    keep = [f"{emotion}_zscore", "CAP_positive", "CAP_negative", "framewise_displacement", "run_id", "session"]
    df = df.dropna(subset=[f"{emotion}_zscore"]).copy()
    return df[keep]


# -----------------------------------------
# Primary model (without hormones) – LOSO
# -----------------------------------------

def loso_by_session_primary_model(model_df: pd.DataFrame, emotion: str, participant_id: str):
    """Leave-one-session-out CV using MixedLM without hormone predictor.
    Returns per-session RMSE and predictions.
    """
    if not HAS_STATSMODELS:
        raise RuntimeError("statsmodels is required for MixedLM")

    ses_ids = sorted(model_df["session"].unique())
    session_rmse = []
    fold_details = []

    for test_s in ses_ids:
        train = model_df[model_df["session"] != test_s].copy()
        test = model_df[model_df["session"] == test_s].copy()
        if len(train) < 30 or len(test) < 10:
            continue

        ycol = f"{emotion}_zscore"
        formula = f"{ycol} ~ CAP_positive + CAP_negative + framewise_displacement"
        try:
            md = mixedlm(formula, train, groups=train["session"])  # random intercept by session
            mdf = md.fit()

            # Autocorrelation diagnostic
            ac = acorr_ljungbox(mdf.resid, lags=[1], return_df=True)
            has_ar1 = ac["lb_pvalue"].iloc[0] < 0.05

            # Predict (fixed effects part)
            X_test = test[["CAP_positive", "CAP_negative", "framewise_displacement"]].copy()
            # Ensure intercept name matches MixedLM fe_params index ('Intercept')
            X_test.insert(0, "Intercept", 1.0)
            y_true = test[ycol].values
            y_pred = (X_test @ mdf.fe_params).values

            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            r2 = float(r2_score(y_true, y_pred))
            fold_details.append(
                {
                    "participant_id": participant_id,
                    "session": int(test_s),
                    "emotion": emotion,
                    "rmse": rmse,
                    "r2": r2,
                    "ar1_detected": bool(has_ar1),
                    "n_test": len(test),
                }
            )
            session_rmse.append((int(test_s), rmse))
        except Exception as e:
            print(f"[{participant_id} {emotion}] LOSO session {test_s} failed: {e}")
            continue

    return session_rmse, fold_details


# -----------------------------------------
# Hormone-extended model (fixed + interactions)
# -----------------------------------------

def fit_hormone_extended_model(model_df: pd.DataFrame, hormone_df: pd.DataFrame, emotion: str, participant_id: str):
    """Fit LME with hormone as fixed effect and CAP*hormone interactions.
    Random intercept by session. Random slope for hormone is not identifiable when
    hormone is constant within session; we therefore include fixed effects and
    interactions, and allow the intercept to vary by session.
    """
    if not HAS_STATSMODELS:
        raise RuntimeError("statsmodels is required for MixedLM")

    df = model_df.merge(hormone_df, on=["participant_id", "session"], how="left")
    ycol = f"{emotion}_zscore"
    formula = (
        f"{ycol} ~ CAP_positive + CAP_negative + framewise_displacement + hormone "
        "+ CAP_positive:hormone + CAP_negative:hormone"
    )
    try:
        md = mixedlm(formula, df, groups=df["session"])  # random intercept by session
        mdf = md.fit()

        # Autocorrelation diagnostic
        ac = acorr_ljungbox(mdf.resid, lags=[1], return_df=True)
        has_ar1 = ac["lb_pvalue"].iloc[0] < 0.05

        # Collect fixed effects table
        fe = mdf.fe_params.rename("coef").to_frame()
        fe["pval"] = mdf.pvalues.reindex(fe.index)
        fe.reset_index(inplace=True)
        fe.rename(columns={"index": "term"}, inplace=True)

        return {
            "participant_id": participant_id,
            "emotion": emotion,
            "ar1_detected": bool(has_ar1),
            "n_obs": int(len(df)),
            "fixed_effects": fe,
            "model": mdf,
        }
    except Exception as e:
        print(f"[{participant_id} {emotion}] Hormone model failed: {e}")
        return None


# ------------------------------
# Visualization helpers
# ------------------------------

def plot_rmse_hormone_scatter(corr_df: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    for (pid, emo), df in corr_df.groupby(["participant_id", "emotion"]):
        if df["hormone"].nunique() < 2:
            continue
        plt.figure(figsize=(5.5, 4.2))
        sns.regplot(
            data=df,
            x="hormone",
            y="rmse",
            scatter_kws=dict(s=40, alpha=0.8),
            line_kws=dict(color="black", alpha=0.6),
            ci=None,
        )
        rho, p = spearmanr(df["hormone"], df["rmse"])
        plt.title(f"{pid} • {emo.title()}\nSpearman ρ={rho:.2f}, p={p:.3f}")
        plt.xlabel("Hormone concentration (a.u.)")
        plt.ylabel("LOSO RMSE (no-hormone model)")
        plt.tight_layout()
        plt.savefig(outdir / f"{pid}_{emo}_rmse_vs_hormone.png", dpi=300)
        plt.close()


def save_fixed_effects_tables(results: list, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for res in results:
        if res is None:
            continue
        fe = res["fixed_effects"].copy()
        fe.insert(0, "participant_id", res["participant_id"])
        fe.insert(1, "emotion", res["emotion"])
        rows.append(fe)
    if rows:
        all_fe = pd.concat(rows, ignore_index=True)
        all_fe.to_csv(outdir / "hormone_lme_fixed_effects.tsv", sep="\t", index=False)


# ------------------------------
# Main analysis orchestration
# ------------------------------

def main():
    print("Hormone-extended CPM analysis")
    print("-" * 34)
    if not HAS_STATSMODELS:
        print("This script requires statsmodels. Please install it and retry.")
        return

    figures_dir = Path("derivatives/caps/cap-analysis/figures")
    results_dir = Path("derivatives/caps/cap-analysis")
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading participant runs and emotions…")
    participant_data = load_participant_data()
    print("Loading hormone levels…")
    hormone_levels = load_hormone_levels()

    # Ensure types
    hormone_levels["session"] = hormone_levels["session"].astype(int)

    # Containers for outputs
    corr_records = []
    hormone_model_results = []

    for pid, runs in participant_data.items():
        print(f"\nParticipant: {pid} ({CONTRACEPTIVE_STATUS[pid]})")
        all_df = pd.concat(runs, ignore_index=True)
        # keep participant id and session columns
        all_df = all_df.copy()
        all_df["participant_id"] = pid

        pid_horm = hormone_levels[hormone_levels["participant_id"] == pid].copy()
        if pid_horm.empty:
            print("  No hormone entries found – skipping participant.")
            continue

        for emotion in ["valence", "arousal"]:
            print(f"  Emotion: {emotion}")
            model_df = prepare_model_data(all_df, emotion)
            if model_df.empty:
                print("    No model data.")
                continue

            # Add participant id for merge
            model_df.insert(0, "participant_id", pid)

            # 1) Primary model (no hormone), leave-one-session-out CV
            session_rmse, folds = loso_by_session_primary_model(model_df, emotion, pid)

            # Store fold details
            if folds:
                fold_df = pd.DataFrame(folds)
                fold_df.to_csv(
                    results_dir / f"{pid}_{emotion}_primary_loso_folds.tsv",
                    sep="\t",
                    index=False,
                )

            # Correlate per-session RMSE with hormone concentration (pre-session assay)
            for s, rmse in session_rmse:
                h = pid_horm.loc[pid_horm["session"] == s, "hormone"]
                if not h.empty:
                    corr_records.append(
                        {
                            "participant_id": pid,
                            "emotion": emotion,
                            "session": int(s),
                            "hormone": float(h.iloc[0]),
                            "rmse": float(rmse),
                        }
                    )

            # 2) Hormone-extended model with interactions
            res = fit_hormone_extended_model(model_df, pid_horm, emotion, pid)
            hormone_model_results.append(res)

    # Aggregate and compute Spearman correlations
    if corr_records:
        corr_df = pd.DataFrame(corr_records)
        corr_df.to_csv(results_dir / "session_rmse_vs_hormone.tsv", sep="\t", index=False)

        # Per-participant/emotion correlation summary
        rows = []
        for (pid, emo), df in corr_df.groupby(["participant_id", "emotion"]):
            if df["hormone"].nunique() >= 2 and len(df) >= 3:
                rho, p = spearmanr(df["hormone"], df["rmse"])
                rows.append({
                    "participant_id": pid,
                    "emotion": emo,
                    "n_sessions": int(df["session"].nunique()),
                    "spearman_rho": float(rho),
                    "p_value": float(p),
                })
        if rows:
            pd.DataFrame(rows).to_csv(
                results_dir / "session_rmse_vs_hormone_summary.tsv", sep="\t", index=False
            )

        # Plots
        plot_rmse_hormone_scatter(corr_df, figures_dir)

    # Save fixed effects from hormone-extended models
    save_fixed_effects_tables(hormone_model_results, results_dir)

    print("\nAnalysis complete. Outputs written to:")
    print(f"  - {results_dir}")
    print(f"  - {figures_dir}")


if __name__ == "__main__":
    main()
