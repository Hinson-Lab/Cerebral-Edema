"""
Descriptive diagnostics for admission glucose and blood alcohol
(Supplementary Fig. 3). Ported from the original stand-alone script; the
plotting functions are unchanged apart from reading thresholds and units from
``config`` and taking baseline CE directly from the analysis dataset.

Outputs (written to the output directory)
-----------------------------------------
glucose_by_edema_trajectory.png          Supp. Fig. 3A
alcohol_modified_z_by_trajectory.png     Supp. Fig. 3B
alcohol_detectability_worsening_rate.png Worsening rate by detectable alcohol (Wilson CIs)
alcohol_above_300_outcome_counts.png     Outcome counts among alcohol > cutoff
glucose_alcohol_figure_summary.csv       Thresholds and counts used in the figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config


def make_trajectory(df: pd.DataFrame) -> pd.Series:
    """
    Combine TRUE baseline CE status with the 6-hour worsening outcome.

    Four groups are retained because patients without CE at baseline can
    subsequently worsen/develop CE by 6 hours.
    """
    baseline = df["Baseline CE"]
    worse = df["Edema Worse"]

    conditions = [
        baseline.eq(0) & worse.eq(0),
        baseline.eq(0) & worse.eq(1),
        baseline.eq(1) & worse.eq(0),
        baseline.eq(1) & worse.eq(1),
    ]
    labels = [
        "No baseline CE,\nno worsening",
        "No baseline CE,\nworsened",
        "Baseline CE,\nno worsening",
        "Baseline CE,\nworsened",
    ]
    trajectory = np.select(conditions, labels, default="Missing/other")
    return pd.Series(trajectory, index=df.index, name="Edema trajectory")

def wilson_interval(events: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return the Wilson 95% confidence interval for a binomial proportion."""
    if total <= 0:
        return np.nan, np.nan
    p = events / total
    denominator = 1 + (z**2 / total)
    center = (p + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * np.sqrt((p * (1 - p) / total) + (z**2 / (4 * total**2)))
        / denominator
    )
    return center - half_width, center + half_width


def add_jittered_points(
    ax: plt.Axes,
    values: pd.Series,
    positions: pd.Series,
    rng: np.random.Generator,
    jitter_width: float = 0.12,
) -> None:
    """Add jittered individual observations to an axis."""
    x = positions.to_numpy(dtype=float)
    y = values.to_numpy(dtype=float)
    jitter = rng.uniform(-jitter_width, jitter_width, size=len(values))
    ax.scatter(x + jitter, y, s=28, alpha=0.75, edgecolors="none")


def plot_glucose(
    df: pd.DataFrame,
    output_dir: Path,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    plot_df = df.dropna(
        subset=["Glucose", "Baseline CE", "Edema Worse"]
    ).copy()

    glucose = plot_df["Glucose"]
    q1 = float(glucose.quantile(0.25))
    q3 = float(glucose.quantile(0.75))
    iqr = q3 - q1
    upper_3iqr = q3 + config.EXTREME_VALUE_IQR_MULTIPLIER * iqr

    preferred_order = [
        "No baseline CE,\nno worsening",
        "No baseline CE,\nworsened",
        "Baseline CE,\nno worsening",
        "Baseline CE,\nworsened",
        "Missing/other",
    ]
    observed = set(plot_df["Edema trajectory"])
    order = [label for label in preferred_order if label in observed]

    position_map = {label: i + 1 for i, label in enumerate(order)}
    plot_df["plot_position"] = plot_df["Edema trajectory"].map(position_map)

    grouped_values = [
        plot_df.loc[plot_df["Edema trajectory"].eq(label), "Glucose"].to_numpy()
        for label in order
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

    ax.boxplot(
        grouped_values,
        positions=np.arange(1, len(order) + 1),
        widths=0.48,
        showfliers=False,
        medianprops={"linewidth": 1.8},
    )
    add_jittered_points(
        ax=ax,
        values=plot_df["Glucose"],
        positions=plot_df["plot_position"],
        rng=rng,
    )

    ax.axhline(
        upper_3iqr,
        linestyle="--",
        linewidth=1.5,
        label=f"Q3 + {config.EXTREME_VALUE_IQR_MULTIPLIER:g}xIQR = {upper_3iqr:.0f} {config.GLUCOSE_UNIT}",
    )

    labels_with_n = []
    for label in order:
        group = plot_df.loc[plot_df["Edema trajectory"].eq(label)]
        n_total = len(group)
        n_extreme = int(group["Glucose"].gt(upper_3iqr).sum())
        labels_with_n.append(f"{label}\n(n={n_total})")

        y_text = max(float(group["Glucose"].max()), upper_3iqr)
        ax.text(
            position_map[label],
            y_text + 10,
            f"{n_extreme} above\n{config.EXTREME_VALUE_IQR_MULTIPLIER:g}xIQR",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels(labels_with_n)
    ax.set_ylabel(f"Admission glucose ({config.GLUCOSE_UNIT})")
    ax.set_title("Admission glucose by baseline CE and 6-hour trajectory")
    ax.legend(frameon=False, loc="lower left")
    ax.margins(y=0.15)

    fig.savefig(
        output_dir / "glucose_by_edema_trajectory.png",
        dpi=config.FIG_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    return {
        "glucose_n": int(glucose.notna().sum()),
        "glucose_q1": q1,
        "glucose_q3": q3,
        "glucose_iqr": iqr,
        "glucose_upper_3iqr": upper_3iqr,
        "glucose_n_above_upper_3iqr": int(glucose.gt(upper_3iqr).sum()),
    }


def plot_alcohol_modified_z(
    df: pd.DataFrame,
    output_dir: Path,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    positive = df.loc[
        df["Alcohol"].gt(0)
        & df["Alcohol"].notna()
        & df["Baseline CE"].notna()
        & df["Edema Worse"].notna()
    ].copy()

    median_positive = float(positive["Alcohol"].median())
    mad_positive = float(
        np.median(np.abs(positive["Alcohol"].to_numpy() - median_positive))
    )
    if mad_positive == 0:
        raise ValueError(
            "MAD among detectable alcohol values is zero; modified z-scores "
            "cannot be calculated."
        )

    positive["Modified z"] = (
        config.MODIFIED_Z_CONSTANT
        * (positive["Alcohol"] - median_positive)
        / mad_positive
    )

    upper_raw_threshold = (
        median_positive + config.MODIFIED_Z_THRESHOLD * mad_positive / config.MODIFIED_Z_CONSTANT
    )

    preferred_order = [
        "No baseline CE,\nno worsening",
        "No baseline CE,\nworsened",
        "Baseline CE,\nno worsening",
        "Baseline CE,\nworsened",
        "Missing/other",
    ]
    observed = set(positive["Edema trajectory"])
    order = [label for label in preferred_order if label in observed]

    position_map = {label: i + 1 for i, label in enumerate(order)}
    positive["plot_position"] = positive["Edema trajectory"].map(position_map)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

    add_jittered_points(
        ax=ax,
        values=positive["Modified z"],
        positions=positive["plot_position"],
        rng=rng,
    )

    ax.axhline(config.MODIFIED_Z_THRESHOLD, linestyle="--", linewidth=1.5)
    ax.axhline(-config.MODIFIED_Z_THRESHOLD, linestyle="--", linewidth=1.5)
    ax.text(
        0.02,
        0.87,
        (
            "Modified z calculated among non-zero alcohol values only\n"
            f"|z| = {config.MODIFIED_Z_THRESHOLD}; upper raw-value equivalent ≈ "
            f"{upper_raw_threshold:.0f} {config.ALCOHOL_UNIT}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    labels_with_n = []
    for label in order:
        n_group = int(positive["Edema trajectory"].eq(label).sum())
        labels_with_n.append(f"{label}\n(n={n_group})")

    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels(labels_with_n)
    ax.set_ylabel("Modified z-score for admission alcohol")
    ax.set_title("Alcohol outlier assessment by baseline CE and 6-hour trajectory")
    ax.set_ylim(-4.2, 4.2)

    fig.savefig(
        output_dir / "alcohol_modified_z_by_trajectory.png",
        dpi=config.FIG_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    return {
        "alcohol_positive_n": int(len(positive)),
        "alcohol_positive_median": median_positive,
        "alcohol_positive_mad": mad_positive,
        "alcohol_modified_z_upper_raw_threshold": float(upper_raw_threshold),
        "alcohol_max": float(positive["Alcohol"].max()),
        "alcohol_max_modified_z": float(positive["Modified z"].max()),
        "alcohol_n_abs_modified_z_gt_3_5": int(
            positive["Modified z"].abs().gt(config.MODIFIED_Z_THRESHOLD).sum()
        ),
    }


def plot_alcohol_detectability_rate(
    df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, float | int]:
    observed = df.dropna(subset=["Alcohol", "Edema Worse"]).copy()
    observed["Alcohol group"] = np.where(
        observed["Alcohol"].gt(0),
        "Detectable alcohol",
        "No detectable alcohol",
    )

    order = ["No detectable alcohol", "Detectable alcohol"]
    rows = []
    for label in order:
        group = observed.loc[observed["Alcohol group"].eq(label)]
        events = int(group["Edema Worse"].eq(1).sum())
        total = int(len(group))
        rate = events / total
        lower, upper = wilson_interval(events, total)
        rows.append(
            {
                "group": label,
                "events": events,
                "total": total,
                "rate": rate,
                "lower": lower,
                "upper": upper,
            }
        )

    rate_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    x = np.arange(len(rate_df))
    rates_pct = rate_df["rate"].to_numpy() * 100
    lower_err = (rate_df["rate"] - rate_df["lower"]).to_numpy() * 100
    upper_err = (rate_df["upper"] - rate_df["rate"]).to_numpy() * 100

    ax.bar(
        x,
        rates_pct,
        yerr=np.vstack([lower_err, upper_err]),
        capsize=5,
        width=0.62,
    )

    for i, row in rate_df.iterrows():
        ax.text(
            i,
            row["rate"] * 100 + upper_err[i] + 2,
            f'{row["events"]}/{row["total"]}\n({row["rate"] * 100:.1f}%)',
            ha="center",
            va="bottom",
            fontsize=10,
        )

    missing_n = int(df["Alcohol"].isna().sum())
    ax.set_xticks(x)
    ax.set_xticklabels(rate_df["group"])
    ax.set_ylabel("Cerebral-edema worsening rate (%)")
    ax.set_title("Six-hour worsening by detectable admission alcohol")
    ax.set_ylim(0, max(55, float((rates_pct + upper_err).max() + 12)))
    fig.savefig(
        output_dir / "alcohol_detectability_worsening_rate.png",
        dpi=config.FIG_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    return {
        "alcohol_observed_n": int(len(observed)),
        "alcohol_missing_n": missing_n,
        "alcohol_zero_n": int(observed["Alcohol"].eq(0).sum()),
        "alcohol_detectable_n": int(observed["Alcohol"].gt(0).sum()),
        "worsening_rate_alcohol_zero": float(
            observed.loc[observed["Alcohol"].eq(0), "Edema Worse"].mean()
        ),
        "worsening_rate_alcohol_detectable": float(
            observed.loc[observed["Alcohol"].gt(0), "Edema Worse"].mean()
        ),
    }


def plot_alcohol_above_300(
    df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, int]:
    observed = df.dropna(subset=["Alcohol", "Edema Worse"]).copy()
    high = observed.loc[observed["Alcohol"].gt(config.ALCOHOL_HIGH_CUTOFF)].copy()

    counts = (
        high["Edema Worse"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
        .astype(int)
    )

    fig, ax = plt.subplots(figsize=(5.8, 4.5), constrained_layout=True)
    x = np.arange(2)
    heights = counts.to_numpy()

    ax.bar(x, heights, width=0.62)
    for i, count in enumerate(heights):
        ax.text(i, count + 0.12, str(int(count)), ha="center", va="bottom", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(["No 6-hour worsening", "6-hour worsening"])
    ax.set_ylabel(f"Patients with alcohol >300 {config.ALCOHOL_UNIT}")
    ax.set_title("Outcome split among patients with alcohol >300")
    ax.set_ylim(0, max(5, int(heights.max()) + 1))
    ax.text(
        0.01,
        0.98,
        f"Total n={len(high)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )

    fig.savefig(
        output_dir / "alcohol_above_300_outcome_counts.png",
        dpi=config.FIG_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    above_350 = observed.loc[observed["Alcohol"].gt(350), "Edema Worse"]
    return {
        "alcohol_n_above_300": int(len(high)),
        "alcohol_above_300_no_worsening": int(counts.loc[0]),
        "alcohol_above_300_worsening": int(counts.loc[1]),
        "alcohol_n_above_350": int(len(above_350)),
        "alcohol_above_350_no_worsening": int(above_350.eq(0).sum()),
        "alcohol_above_350_worsening": int(above_350.eq(1).sum()),
    }


def run_diagnostics(df: pd.DataFrame, output_dir: Path | str) -> pd.DataFrame:
    """
    Build the four figures and the summary table from the analysis dataset
    (subjects with a 6-hour follow-up CT). Returns the summary as a DataFrame.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = df[df[config.COL_WORSE].notna()].copy()
    for col in ("Glucose", "Alcohol"):
        if col not in data.columns:
            raise ValueError(f"'{col}' is not in the analysis dataset; add its REDCap label to "
                             "CLINICAL_COLUMN_MAP in config.py.")
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["Baseline CE"] = pd.to_numeric(data[config.COL_CE_BASELINE], errors="coerce")
    data["Edema Worse"] = pd.to_numeric(data[config.COL_WORSE], errors="coerce")
    data["Edema trajectory"] = make_trajectory(data)
    rng = np.random.default_rng(config.DIAGNOSTIC_JITTER_SEED)

    summary: dict[str, float | int] = {
        "analytic_n": int(len(data)),
        "baseline_ce_absent_n": int(data["Baseline CE"].eq(0).sum()),
        "baseline_ce_present_n": int(data["Baseline CE"].eq(1).sum()),
        "worsening_no_n": int(data["Edema Worse"].eq(0).sum()),
        "worsening_yes_n": int(data["Edema Worse"].eq(1).sum()),
    }
    for trajectory, count in data["Edema trajectory"].value_counts().items():
        key = ("trajectory_" + trajectory.replace("\n", "_").replace(",", "")
               .replace(" ", "_").replace("/", "_").lower() + "_n")
        summary[key] = int(count)

    summary.update(plot_glucose(data, output_dir, rng))
    summary.update(plot_alcohol_modified_z(data, output_dir, rng))
    summary.update(plot_alcohol_detectability_rate(data, output_dir))
    summary.update(plot_alcohol_above_300(data, output_dir))

    summary_df = pd.DataFrame([{"metric": k, "value": v} for k, v in summary.items()])
    summary_df.to_csv(output_dir / "glucose_alcohol_figure_summary.csv", index=False)
    return summary_df
