import math
import os
import base64
import binascii
import configparser
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from DataTaking import countD
from flask import jsonify, request

try:
    from scipy.stats import chi2 as chi2_dist
except ImportError:  # pragma: no cover - optional dependency fallback
    chi2_dist = None

from dash import Dash, dcc, html, Input, Output, State, dash_table, no_update


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "detector_measurements.csv"
ANGLE_PARAMETERS_FILE = BASE_DIR / "def_param.cfi"
EXPORT_DIR = BASE_DIR / "exports"
POISSON_ACQUISITION_DURATION = 60
POISSON_ACQUISITION_ANGLE = 90

# Simple translations map for UI strings
TRANSLATIONS = {
    "en": {
        "app_title": "Cosmic Detector Analyzer",
        "home_h2": "Cosmic Detector Analyzer",
        "home_p": "Launch a measurement and inspect the detector data across dedicated pages.",
        "link_home": "Home",
        "link_poisson": "Poisson law",
        "link_trends": "Trend plots",
        "link_angle": "Angle dependence",
        "export_images": "Export images",
        "exporting_images": "Exporting…",
        "export_no_charts": "No charts available to export.",
        "export_success": "Saved {} images in exports/",
        "export_error": "Image export failed.",
        "last_measurement_summary": "Last measurement summary",
        "measurement_time_label": "Recorded",
        "measurement_operator_label": "Operator",
        "measurement_duration_label": "Duration",
        "measurement_angle_label": "Angle",
        "poisson_scope": "60-second acquisition at 90°",
        "run_measurement_hint": "Run a new measurement to update the summary.",
        "duration_label": "Duration (s): ",
        "person_label": "Operator name: ",
        "person_placeholder": "Enter your name",
        "angle_label": "Angle (deg): ",
        "run_button": "Run measurement",
        "progress_label": "Measurement progress: {}%",
        "latest_measurements": "All Measurements",
        "poisson_h2": "Poisson law check (60-second acquisitions at 90°)",
        "current_distributions": "Current distributions",
        "activate_fit": "Activate fit",
        "click_fit_hint": "Click the fit button to overlay a Poisson fit on each histogram.",
        "fitted_parameter": "Fitted parameter: μ = {}",
        "chi2_probability": "χ² probability: {}",
        "angle_fit_title": "A·cos(θ−π/2)^n fit",
        "angle_fit_function": "Model: f(θ) = A·cos(θ−π/2)^n",
        "angle_fit_uncertainty": "Fit uncertainty",
        "angle_plot_title": "Mean coincidence rate per minute versus angle",
        "angle_ylabel": "Coincidences per minute",
        "angle_a_history_title": "Fitted A as measurements are added",
        "angle_n_history_title": "Fitted n as measurements are added",
        "angle_history_x": "Measurements included",
        "angle_a_axis": "A (coincidences/min)",
        "angle_n_axis": "n",
        "angle_fit_summary": "Fit result: A = {} ± {} | n = {} ± {}",
        "animated_history": "Animated history",
        "trend_h2": "Trend plots (detector in vertical position)",
        "rate_trend_title": "Cumulative mean count rate per minute (detector in vertical position)",
        "run_number_label": "Measurement number",
        "rate_ylabel": "Counts per minute",
        "angle_h2": "Angle dependence",
        "angle_tendency_curve": "Tendency curve",
        "no_data": "No data yet",
        "no_angle_data": "No measurements at 90°",
        "no_60s": "No 60-second runs at 90° for {}",
        "no_timestamps": "No valid timestamps for {}",
        "histogram_title": "Histogram of {} (60-second runs at 90°)",
        "histogram_over_time": "Histogram of {} over time (60-second runs at 90°)",
        "count_label": "Count",
        "mean_annotation": "Mean = {} ± {}",
        "play_label": "Play",
        "date_prefix": "Date: ",
        "waiting_first": "Waiting for the first measurement.",
        "first_measurement": "No measurement available yet.",
        "compatibility_none": "No previous data to compare.",
        "compatibility_prefix": "Compatibility with previous measurements: {}.",
        "count1_title": "Count 1",
        "count2_title": "Count 2",
        "coincidences_title": "Coincidences",
        "measurement_saved": "Measurement saved: {} | operator={} | duration={}s | angle={}°",
    },
    "fr": {
        "app_title": "Analyseur du détecteur cosmique",
        "home_h2": "Analyseur du détecteur cosmique",
        "home_p": "Lancer une mesure et inspecter les données du détecteur sur des pages dédiées.",
        "link_home": "Accueil",
        "link_poisson": "Loi de Poisson",
        "link_trends": "Évolutions",
        "link_angle": "Dépendance angulaire",
        "export_images": "Exporter les images",
        "exporting_images": "Export en cours…",
        "export_no_charts": "Aucun graphique à exporter.",
        "export_success": "{} images enregistrées dans exports/",
        "export_error": "Échec de l'export des images.",
        "last_measurement_summary": "Résumé de la dernière mesure",
        "measurement_time_label": "Date et heure",
        "measurement_operator_label": "Opérateur",
        "measurement_duration_label": "Durée",
        "measurement_angle_label": "Angle",
        "poisson_scope": "Prise de 60 s à 90°",
        "run_measurement_hint": "Exécutez une nouvelle mesure pour mettre à jour le résumé.",
        "duration_label": "Durée (s) : ",
        "person_label": "Nom de l'opérateur : ",
        "person_placeholder": "Entrez votre nom",
        "angle_label": "Angle (°) : ",
        "run_button": "Exécuter la mesure",
        "progress_label": "Progression de la mesure : {}%",
        "latest_measurements": "Toutes les mesures",
        "poisson_h2": "Vérification de la loi de Poisson (prises de 60 secondes à 90°)",
        "current_distributions": "Distributions actuelles",
        "activate_fit": "Activer l'ajustement",
        "click_fit_hint": "Cliquez sur le bouton d'ajustement pour superposer un ajustement de Poisson sur chaque histogramme.",
        "fitted_parameter": "Paramètre ajusté : μ = {}",
        "chi2_probability": "Probabilité χ² : {}",
        "angle_fit_title": "A·cos(θ−π/2)^n ajustement",
        "angle_fit_function": "Modèle : f(θ) = A·cos(θ−π/2)^n",
        "angle_fit_uncertainty": "Incertitude de l'ajustement",
        "angle_plot_title": "Taux moyen de coïncidences par minute en fonction de l'angle",
        "angle_ylabel": "Coïncidences par minute",
        "angle_a_history_title": "Évolution de A selon les mesures ajoutées",
        "angle_n_history_title": "Évolution de n selon les mesures ajoutées",
        "angle_history_x": "Nombre de mesures incluses",
        "angle_a_axis": "A (coïncidences/min)",
        "angle_n_axis": "n",
        "angle_fit_summary": "Résultat de l'ajustement : A = {} ± {} | n = {} ± {}",
        "animated_history": "Historique animé",
        "trend_h2": "Graphiques d'évolution en fonction du temps (détecteur en position verticale)",
        "rate_trend_title": "Moyenne cumulée des décomptes par minute (détecteur en position verticale)",
        "run_number_label": "Numéro de la prise",
        "rate_ylabel": "Décomptes par minute",
        "angle_h2": "Dépendance angulaire",
        "angle_tendency_curve": "Courbe de tendance",
        "no_data": "Pas encore de données",
        "no_angle_data": "Aucune mesure à 90°",
        "no_60s": "Aucune prise de 60 secondes à 90° pour {}",
        "no_timestamps": "Aucun horodatage valide pour {}",
        "histogram_title": "Histogramme de {} (prises de 60 secondes à 90°)",
        "histogram_over_time": "Histogramme de {} au fil du temps (prises de 60 secondes à 90°)",
        "count_label": "Nombre",
        "mean_annotation": "Moyenne = {} ± {}",
        "play_label": "Lire",
        "date_prefix": "Date : ",
        "waiting_first": "Nous attendons que vous prenez votre première mesure.",
        "first_measurement": "No measurement available yet.",
        "compatibility_none": "No previous data to compare.",
        "compatibility_prefix": "Compatibilité avec les mesures précédentes : {}.",
        "count1_title": "Taux de comptage du détecteur 1",
        "count2_title": "Taux de comptage du détecteur 2",
        "coincidences_title": "Coïncidences",
        "measurement_saved": "Measurement saved: {} | opérateur={} | durée={}s | angle={}°",
    },
}


def poisson_pmf(k_values: np.ndarray, mu: float) -> np.ndarray:
    k_values = np.asarray(k_values, dtype=int)
    if mu <= 0:
        return np.where(k_values == 0, 1.0, 0.0).astype(float)
    log_terms = -mu + k_values * np.log(mu) - np.array([math.lgamma(int(k) + 1) for k in k_values])
    return np.exp(log_terms)


def ensure_data_file(csv_path: Union[Path, str] = DATA_FILE) -> Path:
    path = Path(csv_path)
    if not path.exists():
        pd.DataFrame(
            columns=[
                "time",
                "person",
                "duration",
                "angle",
                "count_1",
                "count_2",
                "coincidences",
            ]
        ).to_csv(path, index=False)
    return path


def append_measurement(
    csv_path: Union[Path, str] = DATA_FILE,
    duration: float = 10.0,
    angle: float = 0.0,
    count_1: int = 0,
    count_2: int = 0,
    coincidences: int = 0,
    person: str = "",
) -> Dict[str, Any]:
    path = ensure_data_file(csv_path)
    row = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "person": str(person),
        "duration": float(duration),
        "angle": float(angle),
        "count_1": int(count_1),
        "count_2": int(count_2),
        "coincidences": int(coincidences),
    }
    df = pd.read_csv(path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)
    return row


def load_measurements(csv_path: Union[Path, str] = DATA_FILE) -> pd.DataFrame:
    path = ensure_data_file(csv_path)
    df = pd.read_csv(path)
    if df.empty:
        return df
    for col in ["duration", "angle", "count_1", "count_2", "coincidences"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _chi2_survival_function(chi2_stat: float, dof: int) -> float:
    if chi2_stat <= 0 or dof <= 0:
        return 1.0
    if chi2_dist is not None:
        return float(chi2_dist.sf(chi2_stat, dof))

    # Wilson-Hilferty approximation for the chi-square upper-tail probability.
    z = ((chi2_stat / dof) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * dof))) / math.sqrt(2.0 / (9.0 * dof))
    probability = 0.5 * math.erfc(z / math.sqrt(2.0))
    return float(min(1.0, max(0.0, probability)))


def build_poisson_summary(csv_path: Union[Path, str] = DATA_FILE) -> Dict[str, Any]:
    df = load_measurements(csv_path)
    if df.empty:
        return {
            "mean_count": 0.0,
            "pmf_series": pd.Series(dtype=float),
            "observed_counts": pd.Series(dtype=float),
        }

    observed_counts = df["coincidences"].astype(float).to_numpy()
    mean_count = float(np.mean(observed_counts))
    max_count = int(max(5, np.ceil(mean_count) + 5))
    pmf = poisson_pmf(np.arange(max_count + 1), mean_count)
    pmf_series = pd.Series(pmf, index=np.arange(max_count + 1), name="probability")
    return {
        "mean_count": mean_count,
        "pmf_series": pmf_series,
        "observed_counts": pd.Series(observed_counts, name="coincidences"),
    }


def _load_trend_measurements(csv_path: Union[Path, str]) -> pd.DataFrame:
    df = load_measurements(csv_path)
    if df.empty:
        return df
    return df.loc[df["angle"] == 90].copy()


def _poisson_rate_uncertainties(counts: pd.Series, durations: pd.Series) -> np.ndarray:
    return 60 * np.sqrt(counts.astype(float)) / durations.astype(float)


def build_trend_plot(csv_path: Union[Path, str] = DATA_FILE, lang: str = "fr"):
    t = TRANSLATIONS.get(lang, TRANSLATIONS["fr"])
    df = _load_trend_measurements(csv_path)
    if df.empty:
        return px.scatter(title=t.get("no_angle_data", "No measurements at 90°"))
    df = df.copy()
    df["index"] = np.arange(1, len(df) + 1)
    for column in ["count_1", "count_2", "coincidences"]:
        df[f"{column}_per_minute"] = df[column] / df["duration"] * 60

    rate_columns = [
        ("count_1_per_minute", "count_1"),
        ("count_2_per_minute", "count_2"),
        ("coincidences_per_minute", "coincidences"),
    ]
    series_names = {
        "count_1_per_minute": f"{t.get('count1_title', 'Count 1')}/min",
        "count_2_per_minute": f"{t.get('count2_title', 'Count 2')}/min",
        "coincidences_per_minute": f"{t.get('coincidences_title', 'Coincidences')}/min",
    }
    fig = go.Figure()
    for column, count_column in rate_columns:
        fig.add_trace(go.Scatter(
            x=df["index"],
            y=df[column],
            mode="lines+markers",
            name=series_names[column],
            error_y=dict(
                type="data",
                array=_poisson_rate_uncertainties(df[count_column], df["duration"]),
                visible=True,
            ),
        ))
    fig.update_layout(
        title=t.get("trend_h2", "Évolutions des décomptes dans le temps"),
        xaxis_title=t.get("run_number_label", "Measurement number"),
        yaxis_title=t.get("rate_ylabel", "Counts per minute"),
        template="plotly_white",
    )
    return fig


def build_rate_trend_plot(csv_path: Union[Path, str] = DATA_FILE, lang: str = "fr"):
    df = _load_trend_measurements(csv_path)
    t = TRANSLATIONS.get(lang, TRANSLATIONS["fr"])
    if df.empty:
        return px.scatter(title=t.get("no_angle_data", "No measurements at 90°"))

    df = df.copy()
    df["index"] = np.arange(1, len(df) + 1)
    df["count_1_rate"] = df["count_1"] / df["duration"] * 60
    df["count_2_rate"] = df["count_2"] / df["duration"] * 60
    df["coincidences_rate"] = df["coincidences"] / df["duration"] * 60

    fig = go.Figure()
    for column, count_column, name in [
        ("count_1_rate", "count_1", f"{t.get('count1_title')}/min"),
        ("count_2_rate", "count_2", f"{t.get('count2_title')}/min"),
        ("coincidences_rate", "coincidences", f"{t.get('coincidences_title')}/min"),
    ]:
        values = df[column].astype(float)
        measurement_uncertainties = _poisson_rate_uncertainties(df[count_column], df["duration"])
        running_means = []
        running_uncertainties = []
        for index in range(len(values)):
            subset = values.iloc[: index + 1]
            subset_uncertainties = measurement_uncertainties[: index + 1]
            running_means.append(float(subset.mean()))
            running_uncertainties.append(
                float(np.sqrt(np.square(subset_uncertainties).sum()) / len(subset))
            )

        fig.add_trace(go.Scatter(
            x=df["index"],
            y=running_means,
            mode="lines+markers",
            name=name,
            error_y=dict(type="data", array=running_uncertainties, visible=True),
        ))

    fig.update_layout(
        title=t.get("rate_trend_title", "Moyenne cumulée des décomptes par minute"),
        xaxis_title=t.get("run_number_label", "Numéro de la prise"),
        yaxis_title=t.get("rate_ylabel", "Décomptes par minute"),
        template="plotly_white",
    )
    return fig


def build_angle_fit_result(df: Union[pd.DataFrame, Path, str], lang: str = "fr") -> Dict[str, Any]:
    if not isinstance(df, pd.DataFrame):
        df = load_measurements(df)
    if df.empty:
        return {"A": 0.0, "n": 0.0, "A_unc": 0.0, "n_unc": 0.0, "angle": np.array([]), "fit_rate": np.array([])}

    df = df.copy()
    df["coincidences_per_minute"] = df["coincidences"] / df["duration"] * 60
    grouped = df.groupby("angle")["coincidences_per_minute"]
    angle_groups = pd.DataFrame({
        "angle": list(grouped.groups.keys()),
        "mean_rate": [float(values.mean()) for _, values in grouped],
    }).sort_values("angle")

    angles = angle_groups["angle"].to_numpy(dtype=float)
    rates = angle_groups["mean_rate"].to_numpy(dtype=float)
    if len(angles) == 0:
        return {"A": 0.0, "n": 0.0, "A_unc": 0.0, "n_unc": 0.0, "angle": angles, "fit_rate": np.array([])}

    cos_theta = np.cos(np.deg2rad(angles) - np.pi/2)

    def fit_for_n(n_value: float) -> Optional[Dict[str, Any]]:
        x = np.sign(cos_theta) * np.power(np.abs(cos_theta), n_value)
        try:
            params, residuals, rank, s = np.linalg.lstsq(x[:, np.newaxis], rates, rcond=None)
        except np.linalg.LinAlgError:
            return None
        A_val = float(params[0])
        fit = A_val * x
        ssr = float(np.sum((rates - fit) ** 2))
        return {"A": A_val, "n": float(n_value), "ssr": ssr, "x": x}

    best_fit = None
    for n_value in np.linspace(0.0, 5.0, 101):
        result = fit_for_n(n_value)
        if result is None:
            continue
        if best_fit is None or result["ssr"] < best_fit["ssr"]:
            best_fit = result

    if best_fit is None:
        return {"A": 0.0, "n": 0.0, "A_unc": 0.0, "n_unc": 0.0, "angle": angles, "fit_rate": np.array([])}

    dof = len(rates) - 1
    sigma2 = best_fit["ssr"] / dof if dof > 0 else 0.0
    A_variance = 0.0
    if dof > 0:
        x_norm_squared = float(np.dot(best_fit["x"], best_fit["x"]))
        if x_norm_squared > 0:
            A_variance = sigma2 / x_norm_squared

    A_unc = float(math.sqrt(A_variance))

    n_unc = 0.0
    dense_n = np.linspace(max(0.0, best_fit["n"] - 0.25), min(5.0, best_fit["n"] + 0.25), 21)
    ssr_values = []
    for n_value in dense_n:
        result = fit_for_n(n_value)
        ssr_values.append(result["ssr"] if result is not None else float("inf"))
    min_index = int(np.argmin(ssr_values))
    if len(ssr_values) >= 3 and sigma2 > 0:
        h = dense_n[1] - dense_n[0]
        if min_index == 0:
            second_derivative = (ssr_values[2] - 2.0 * ssr_values[1] + ssr_values[0]) / (h * h)
        elif min_index == len(ssr_values) - 1:
            second_derivative = (ssr_values[-1] - 2.0 * ssr_values[-2] + ssr_values[-3]) / (h * h)
        else:
            second_derivative = (ssr_values[min_index + 1] - 2.0 * ssr_values[min_index] + ssr_values[min_index - 1]) / (h * h)
        if second_derivative > 0:
            n_unc = float(math.sqrt(2.0 * sigma2 / second_derivative))

    fit_rate = best_fit["A"] * best_fit["x"]
    return {
        "A": best_fit["A"],
        "n": best_fit["n"],
        "A_unc": A_unc,
        "n_unc": n_unc,
        "angle": angles,
        "fit_rate": fit_rate,
    }


def build_angle_fit_history(df: Union[pd.DataFrame, Path, str]) -> Dict[str, np.ndarray]:
    if not isinstance(df, pd.DataFrame):
        df = load_measurements(df)
    if df.empty:
        return {key: np.array([]) for key in ("measurement", "A", "A_unc", "n", "n_unc")}

    ordered = df.dropna(subset=["time", "angle", "duration", "coincidences"]).copy()
    ordered["time"] = pd.to_datetime(ordered["time"], errors="coerce")
    ordered = ordered.dropna(subset=["time"]).sort_values("time", kind="stable").reset_index(drop=True)

    history = {key: [] for key in ("measurement", "A", "A_unc", "n", "n_unc")}
    for index in range(len(ordered)):
        cumulative = ordered.iloc[:index + 1]
        if cumulative["angle"].nunique() < 3:
            continue
        fit = build_angle_fit_result(cumulative)
        if len(fit["angle"]) < 3:
            continue
        history["measurement"].append(index + 1)
        for parameter in ("A", "n"):
            history[parameter].append(fit[parameter])
            history[f"{parameter}_unc"].append(fit[f"{parameter}_unc"])

    return {key: np.asarray(values, dtype=float) for key, values in history.items()}


def build_angle_fit_history_plot(
    history: Dict[str, np.ndarray], parameter: str, lang: str = "fr"
):
    t = TRANSLATIONS.get(lang, TRANSLATIONS["fr"])
    if parameter == "A":
        title = t["angle_a_history_title"]
        yaxis_title = t["angle_a_axis"]
    else:
        title = t["angle_n_history_title"]
        yaxis_title = t["angle_n_axis"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history["measurement"],
        y=history[parameter],
        mode="lines+markers",
        name=parameter,
        error_y=dict(type="data", array=history[f"{parameter}_unc"], visible=True),
    ))
    fig.update_layout(
        title=title,
        xaxis_title=t["angle_history_x"],
        yaxis_title=yaxis_title,
        template="plotly_white",
    )
    return fig


def load_angle_curve_parameters() -> Dict[str, float]:
    parser = configparser.ConfigParser()
    parser.read(ANGLE_PARAMETERS_FILE)
    return {
        "A": parser.getfloat("angle_curve", "A", fallback=0.5),
        "n": parser.getfloat("angle_curve", "n", fallback=2.0),
        "B": parser.getfloat("angle_curve", "B", fallback=0.0),
    }


def build_angle_plot(csv_path: Union[Path, str] = DATA_FILE, fit_result: Optional[Dict[str, Any]] = None, lang: str = "fr"):
    df = csv_path.copy() if isinstance(csv_path, pd.DataFrame) else load_measurements(csv_path)
    t = TRANSLATIONS.get(lang, TRANSLATIONS["fr"])
    fig = go.Figure()
    curve_angles = np.linspace(0.0, 180.0, 361)
    curve_parameters = load_angle_curve_parameters()
    curve_rates = 60 * (
        curve_parameters["A"] * np.cos(np.deg2rad(curve_angles) - np.pi / 2) ** curve_parameters["n"]
        + curve_parameters["B"]
    )
    fig.add_trace(go.Scatter(
        x=curve_angles,
        y=curve_rates,
        mode="lines",
        name=t.get("angle_tendency_curve", "Tendency curve"),
        line=dict(color="#c7c7c7", width=2),
    ))
    if df.empty:
        fig.update_layout(
            title=t.get("no_data", "Pas encore de données"),
            xaxis_title=t.get("angle_label", "Angle (°)"),
            yaxis_title=t.get("angle_ylabel", "Coïncidences par minute"),
            template="plotly_white",
        )
        return fig

    df = df.copy()
    df["coincidences_per_minute"] = df["coincidences"] / df["duration"] * 60
    df["poisson_uncertainty_per_minute"] = _poisson_rate_uncertainties(
        df["coincidences"], df["duration"]
    )

    grouped = df.groupby("angle")
    angle_groups = pd.DataFrame({
        "angle": list(grouped.groups.keys()),
        "mean_rate": [float(values["coincidences_per_minute"].mean()) for _, values in grouped],
        "uncertainty": [
            float(
                np.sqrt(np.square(values["poisson_uncertainty_per_minute"]).sum())
                / len(values)
            )
            for _, values in grouped
        ],
    })
    angle_summary = angle_groups.sort_values("angle")

    fig.add_trace(go.Scatter(
        x=angle_summary["angle"],
        y=angle_summary["mean_rate"],
        mode="lines+markers",
        name=f"{t.get('coincidences_title','Coïncidences')}/min",
        error_y=dict(type="data", array=angle_summary["uncertainty"], visible=True),
    ))
    if fit_result is not None and len(fit_result.get("angle", [])) > 0:
        fit_cosine = np.cos(np.deg2rad(curve_angles) - np.pi / 2)
        abs_fit_cosine = np.abs(fit_cosine)
        fit_basis = np.sign(fit_cosine) * np.power(abs_fit_cosine, fit_result["n"])
        basis_n_derivative = np.zeros_like(fit_basis)
        nonzero_cosine = abs_fit_cosine > 0
        basis_n_derivative[nonzero_cosine] = (
            fit_basis[nonzero_cosine] * np.log(abs_fit_cosine[nonzero_cosine])
        )
        fit_rate = fit_result["A"] * fit_basis
        fit_variance = (
            np.square(fit_basis * fit_result["A_unc"])
            + np.square(fit_result["A"] * basis_n_derivative * fit_result["n_unc"])
        )
        fit_uncertainty = np.sqrt(np.maximum(fit_variance, 0.0))
        fig.add_trace(go.Scatter(
            x=curve_angles,
            y=fit_rate + fit_uncertainty,
            mode="lines",
            line=dict(color="rgba(217,83,79,0)"),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=curve_angles,
            y=fit_rate - fit_uncertainty,
            mode="lines",
            line=dict(color="rgba(217,83,79,0)"),
            fill="tonexty",
            fillcolor="rgba(217,83,79,0.18)",
            name=t.get("angle_fit_uncertainty", "Fit uncertainty"),
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=curve_angles,
            y=fit_rate,
            mode="lines",
            name=t.get("angle_fit_title", "A·cos(θ)^n fit"),
            line=dict(color="#d9534f"),
        ))
    fig.update_layout(
        title=t.get("angle_plot_title", "Taux moyen de coïncidences par minute en fonction de l'angle"),
        xaxis_title=t.get("angle_label", "Angle (°)"),
        yaxis_title=t.get("angle_ylabel", "Coïncidences par minute"),
        template="plotly_white",
    )
    return fig


def build_measurement_controls(visible: bool = True, lang: str = "fr"):
    t = TRANSLATIONS.get(lang, TRANSLATIONS["fr"])
    return html.Div(
        className="measurement-controls",
        style={"display": "flex" if visible else "none", "gap": "12px", "flexWrap": "wrap", "alignItems": "flex-end", "marginBottom": "16px"},
        children=[
            html.Div(
                className="control-field",
                children=[html.Label(t["person_label"]), dcc.Input(id="person", type="text", value="", placeholder=t["person_placeholder"])],
            ),
            html.Div(
                className="control-field control-field--duration",
                children=[html.Label(t["duration_label"]), dcc.Input(id="duration", type="number", value=60, min=1, step=1)],
            ),
            html.Div(
                className="control-field control-field--angle",
                children=[html.Label(t["angle_label"]), dcc.Dropdown(
                    id="angle",
                    options=[{"label": str(v), "value": v} for v in [10*i for i in range(19)]],
                    value=90,
                    clearable=False,
                )],
            ),
            html.Button(
                t["run_button"],
                id="run-button",
                n_clicks=0,
                type="button",
            ),
        ],
    )



def build_home_page(status_text: str = None, df: pd.DataFrame = None, lang: str = "fr"):
    if df is None:
        df = load_measurements(DATA_FILE)
    t = TRANSLATIONS.get(lang, TRANSLATIONS["fr"])
    if status_text is None:
        status_text = t["waiting_first"]
    return html.Div(
        className="page-content page-content--home",
        children=[
            html.H2(t["home_h2"]),
            html.P(t["home_p"]),
            html.Div(
                className="page-nav",
                children=[
                    dcc.Link(t["link_home"], href="/"),
                    dcc.Link(t["link_poisson"], href="/poisson"),
                    dcc.Link(t["link_trends"], href="/trends"),
                    dcc.Link(t["link_angle"], href="/angle"),
                ],
            ),
            html.Div(id="status-output", className="status-message", children=status_text),
            html.Hr(className="section-rule"),
            html.Div(
                className="summary-panel summary-panel--hint",
                children=[
                    html.B(t["last_measurement_summary"]),
                    html.Div(t["run_measurement_hint"]),
                ],
            ),
            build_measurement_controls(visible=True, lang=lang),
            html.Div(
                id="measurement-progress-container",
                className="measurement-progress",
                style={"maxHeight": "0", "opacity": 0, "overflow": "hidden", "margin": "0 24px", "transition": "max-height 150ms ease, opacity 150ms ease"},
                children=[
                    html.Div(id="measurement-progress-label", className="measurement-progress__label"),
                    html.Div(
                        className="measurement-progress__track",
                        children=html.Div(
                            id="measurement-progress-bar",
                            className="measurement-progress__bar",
                            style={"height": "100%", "width": "0%", "transition": "width 100ms linear"},
                        ),
                    ),
                ],
            ),
            html.H3(t["latest_measurements"], className="section-heading"),
            dcc.Loading(children=[
                dash_table.DataTable(
                    id="measurements-table",
                    data=df.iloc[::-1].to_dict("records"),
                    columns=[{"name": col, "id": col} for col in df.columns],
                    page_size=10,
                    style_table={"overflowX": "auto"},
                    style_cell={"padding": "8px", "textAlign": "left"},
                    style_header={"backgroundColor": "#f2f2f2", "fontWeight": "bold"},
                )
            ]),
        ],
    )


def _filter_poisson_measurements(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "duration" not in df.columns or "angle" not in df.columns:
        return df.iloc[0:0].copy()
    return df.loc[
        (df["duration"] == POISSON_ACQUISITION_DURATION)
        & (df["angle"] == POISSON_ACQUISITION_ANGLE)
    ].copy()


def build_poisson_page(df: pd.DataFrame = None, fit_requested: bool = False, lang: str = "fr"):
    if df is None:
        df = load_measurements(DATA_FILE)
    df = _filter_poisson_measurements(df)
    t = TRANSLATIONS.get(lang, TRANSLATIONS["fr"])
    summary = build_comparison_summary(df)
    last_row = df.iloc[-1] if not df.empty else None
    if last_row is None:
        last_measurement_content = html.Div(
            summary["last_measurement"],
            className="poisson-summary-empty",
        )
    else:
        summary_fields = [
            (t["measurement_time_label"], str(last_row["time"])),
            (t["measurement_operator_label"], str(last_row["person"])),
            (t["measurement_duration_label"], f"{float(last_row['duration']):g} s"),
            (t["measurement_angle_label"], f"{float(last_row['angle']):g}°"),
            (t["count1_title"], str(int(last_row["count_1"]))),
            (t["count2_title"], str(int(last_row["count_2"]))),
            (t["coincidences_title"], str(int(last_row["coincidences"]))),
        ]
        last_measurement_content = html.Div(
            className="poisson-summary-grid",
            children=[
                html.Div(
                    className="poisson-summary-field",
                    children=[
                        html.Span(label, className="poisson-summary-field__label"),
                        html.Strong(value, className="poisson-summary-field__value"),
                    ],
                )
                for label, value in summary_fields
            ],
        )

    fit_results = []
    titles = [t.get("count1_title"), t.get("count2_title"), t.get("coincidences_title")]
    if fit_requested:
        for (column, _), title in zip([("count_1", ""), ("count_2", ""), ("coincidences", "")], titles):
            fit_results.append(build_poisson_fit_result(df, column, title))

    static_histograms = [
        dcc.Graph(figure=build_static_histogram_figure(df, "count_1", titles[0], fit_result=fit_results[0] if fit_requested else None, lang=lang)),
        dcc.Graph(figure=build_static_histogram_figure(df, "count_2", titles[1], fit_result=fit_results[1] if fit_requested else None, lang=lang)),
        dcc.Graph(figure=build_static_histogram_figure(df, "coincidences", titles[2], fit_result=fit_results[2] if fit_requested else None, lang=lang)),
    ]
    animated_histograms = [
        dcc.Graph(figure=build_histogram_figure(df, "count_1", titles[0], fit_result=fit_results[0] if fit_requested else None, lang=lang)),
        dcc.Graph(figure=build_histogram_figure(df, "count_2", titles[1], fit_result=fit_results[1] if fit_requested else None, lang=lang)),
        dcc.Graph(figure=build_histogram_figure(df, "coincidences", titles[2], fit_result=fit_results[2] if fit_requested else None, lang=lang)),
    ]

    fit_summary = []
    if fit_requested:
        fit_summary = [
            html.Div(
                style={"padding": "12px", "border": "1px solid #ddd", "borderRadius": "6px"},
                children=[
                    html.B(result["title"]),
                    html.Div(t["fitted_parameter"].format(result['mu'])),
                    html.Div(t["chi2_probability"].format(f"{result['chi2_probability']:.3f}")),
                ],
            )
            for result in fit_results
        ]
    else:
        fit_summary = [
            html.Div(
                className="fit-note",
                children="Cliquez sur le bouton d'ajustement pour superposer un ajustement de Poisson sur chaque histogramme.",
            )
        ]

    return html.Div(
        className="page-content",
        children=[
            html.H2(t["poisson_h2"], className="page-title"),
            html.Div(
                className="page-nav",
                children=[
                    dcc.Link(t["link_home"], href="/"),
                    dcc.Link(t["link_poisson"], href="/poisson"),
                    dcc.Link(t["link_trends"], href="/trends"),
                    dcc.Link(t["link_angle"], href="/angle"),
                ],
            ),
            html.Div(
                className="summary-panel summary-panel--poisson",
                children=[
                    html.Div(
                        className="poisson-summary-heading",
                        children=[
                            html.B(t["last_measurement_summary"]),
                            html.Span(t["poisson_scope"], className="poisson-summary-scope"),
                        ],
                    ),
                    last_measurement_content,
                    html.Div(summary["compatibility"], className="poisson-summary-compatibility"),
                ],
            ),
            build_measurement_controls(visible=False, lang=lang),
            html.H3(t["current_distributions"], className="section-heading"),
            html.Div(
                className="fit-action-row",
                children=[
                    dcc.Link(
                        t["activate_fit"],
                        href="/poisson?fit=1",
                        className="action-link",
                    )
                ],
            ),
            html.Div(
                className="chart-grid chart-grid--three",
                children=static_histograms,
            ),
            html.Div(
                className="fit-summary-grid",
                children=fit_summary,
            ),
            html.H3(t["animated_history"], className="section-heading"),
            html.Div(
                className="chart-grid chart-grid--three",
                children=animated_histograms,
            ),
        ],
    )


def build_trend_page(df: pd.DataFrame = None, lang: str = "fr"):
    if df is None:
        df = load_measurements(DATA_FILE)
    t = TRANSLATIONS.get(lang, TRANSLATIONS["fr"])
    summary = build_comparison_summary(df)
    return html.Div(
        className="page-content",
        children=[
            html.H2(t.get("trend_h2", "Graphiques d'évolution"), className="page-title"),
            html.Div(
                className="page-nav",
                children=[
                    dcc.Link(t["link_home"], href="/"),
                    dcc.Link(t["link_poisson"], href="/poisson"),
                    dcc.Link(t["link_trends"], href="/trends"),
                    dcc.Link(t["link_angle"], href="/angle"),
                ],
            ),
            html.Div(
                className="summary-panel",
                children=[
                    html.B(t["last_measurement_summary"]),
                    html.Div(summary["last_measurement"]),
                    html.Div(summary["compatibility"]),
                ],
            ),
            build_measurement_controls(visible=False, lang=lang),
            html.Div(className="chart-panel", children=dcc.Graph(className="plot-panel", figure=build_trend_plot(DATA_FILE, lang=lang))),
            html.Div(className="chart-panel", children=dcc.Graph(className="plot-panel", figure=build_rate_trend_plot(DATA_FILE, lang=lang))),
        ],
    )


def build_angle_page(df: pd.DataFrame = None, fit_requested: bool = False, lang: str = "fr"):
    if df is None:
        df = load_measurements(DATA_FILE)
    t = TRANSLATIONS.get(lang, TRANSLATIONS["fr"])
    summary = build_comparison_summary(df)
    fit_result = build_angle_fit_result(df, lang=lang) if fit_requested else None
    fit_history = build_angle_fit_history(df)
    return html.Div(
        className="page-content",
        children=[
            html.H2(t.get("angle_h2", "Dépendance angulaire"), className="page-title"),
            html.Div(
                className="page-nav",
                children=[
                    dcc.Link(t["link_home"], href="/"),
                    dcc.Link(t["link_poisson"], href="/poisson"),
                    dcc.Link(t["link_trends"], href="/trends"),
                    dcc.Link(t["link_angle"], href="/angle"),
                ],
            ),
            html.Div(
                className="summary-panel",
                children=[
                    html.B(t["last_measurement_summary"]),
                    html.Div(summary["last_measurement"]),
                    html.Div(summary["compatibility"]),
                ],
            ),
            html.Div(
                className="fit-action-row",
                children=[
                    dcc.Link(
                        t["activate_fit"],
                        href="/angle?fit=1",
                        className="action-link",
                    )
                ],
            ),
            html.Div(
                className="fit-note",
                children=[
                    html.Div(t["click_fit_hint"]),
                    html.Div(t["angle_fit_function"]) if fit_requested else None,
                    html.Div(
                        t["angle_fit_summary"].format(
                            f"{fit_result['A']:.2f}",
                            f"{fit_result['A_unc']:.2f}",
                            f"{fit_result['n']:.2f}",
                            f"{fit_result['n_unc']:.2f}",
                        )
                    ) if fit_requested else None,
                ],
            ),
            build_measurement_controls(visible=False, lang=lang),
            html.Div(className="chart-panel", children=dcc.Graph(className="plot-panel", figure=build_angle_plot(df, fit_result=fit_result, lang=lang))),
            html.Div(
                className="chart-grid chart-grid--history",
                children=[
                    html.Div(className="chart-panel", children=dcc.Graph(className="plot-panel", figure=build_angle_fit_history_plot(fit_history, "A", lang=lang))),
                    html.Div(className="chart-panel", children=dcc.Graph(className="plot-panel", figure=build_angle_fit_history_plot(fit_history, "n", lang=lang))),
                ],
            ),
        ],
    )


def create_app() -> Dash:
    app = Dash(__name__, title=TRANSLATIONS.get("fr")["app_title"])
    app.layout = html.Div(
        className="app-shell",
        children=[
            dcc.Location(id="url", refresh=False),
            html.Div(
                className="topbar",
                children=[
                    html.Div(
                        className="brand-lockup",
                        children=[
                            html.Span("CD", className="brand-mark"),
                            html.Div([
                                html.Div(TRANSLATIONS["fr"]["app_title"], className="brand-title"),
                                html.Div("DETECTOR LAB / DATA SYSTEM", className="brand-caption"),
                            ]),
                        ],
                    ),
                    html.Div(
                        className="topbar-actions",
                        children=[
                            html.Button(
                                TRANSLATIONS["fr"]["export_images"],
                                id="export-page-button",
                                type="button",
                                className="export-button",
                                style={"display": "none"},
                            ),
                            html.Span(id="export-status", className="export-status", **{"aria-live": "polite"}),
                            html.Div(
                                className="language-control",
                                children=[
                                    html.Label("Lang:"),
                                    dcc.Dropdown(
                                        id="lang-select",
                                        options=[{"label": "Français", "value": "fr"}, {"label": "English", "value": "en"}],
                                        value="fr",
                                        clearable=False,
                                        className="language-select",
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(id="page-content", className="page-slot", children=build_home_page(lang="fr")),
            dcc.Store(id="measurement-complete", data=0),
        ],
    )

    @app.server.route("/api/export-page-images", methods=["POST"])
    def export_page_images():
        payload = request.get_json(silent=True) or {}
        page_name = payload.get("page")
        images = payload.get("images")
        if page_name not in {"poisson", "trends", "angle"}:
            return jsonify({"error": "Unsupported page"}), 400
        if not isinstance(images, list) or not images or len(images) > 12:
            return jsonify({"error": "Invalid image list"}), 400

        png_images = []
        prefix = "data:image/png;base64,"
        for image in images:
            data_url = image.get("data", "") if isinstance(image, dict) else ""
            if not data_url.startswith(prefix) or len(data_url) > 20_000_000:
                return jsonify({"error": "Invalid PNG data"}), 400
            try:
                image_bytes = base64.b64decode(data_url[len(prefix):], validate=True)
            except (binascii.Error, ValueError):
                return jsonify({"error": "Invalid PNG data"}), 400
            if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                return jsonify({"error": "Invalid PNG signature"}), 400
            png_images.append(image_bytes)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        try:
            EXPORT_DIR.mkdir(parents=True, exist_ok=True)
            filenames = []
            for index, image_bytes in enumerate(png_images, start=1):
                filename = f"{page_name}_{timestamp}_{index:02d}.png"
                (EXPORT_DIR / filename).write_bytes(image_bytes)
                filenames.append(filename)
        except OSError:
            return jsonify({"error": "Could not save exported images"}), 500

        return jsonify({"directory": EXPORT_DIR.name, "files": filenames})

    @app.callback(
        Output("export-page-button", "children"),
        Output("export-page-button", "style"),
        Input("url", "pathname"),
        Input("lang-select", "value"),
    )
    def update_export_button(pathname, lang):
        t = TRANSLATIONS.get(lang, TRANSLATIONS["fr"])
        visible = pathname in {"/poisson", "/trends", "/angle"}
        return t["export_images"], {"display": "inline-flex" if visible else "none"}

    @app.callback(
        Output("page-content", "children"),
        Output("measurement-complete", "data"),
        [Input("url", "pathname"), Input("url", "search"), Input("run-button", "n_clicks"), Input("lang-select", "value")],
        [State("person", "value"), State("duration", "value"), State("angle", "value")],
    )
    def update_dashboard(pathname, search, n_clicks, lang, person, duration, angle):
        t = TRANSLATIONS.get(lang, TRANSLATIONS["fr"])
        if pathname == "/poisson" and "fit=1" in (search or ""):
            return build_poisson_page(fit_requested=True, lang=lang), no_update
        if pathname == "/angle" and "fit=1" in (search or ""):
            return build_angle_page(fit_requested=True, lang=lang), no_update

        if n_clicks is not None and n_clicks >= 1:
            # code generated fake data - disactivated
            #factor = float(np.cos(np.deg2rad(float(angle or 0.0)))) ** 2
            #count_1 = int(np.random.poisson(max(1.0, 10 + 5 * factor)))
            #count_2 = int(np.random.poisson(max(1.0, 8 + 4 * factor)))
            #coincidences = int(np.random.poisson(max(0.0, 2 + 10 * factor)))
            # code calling the function to take data
            res = countD(duration,'0')
            #print(res)
            row = append_measurement(
                DATA_FILE,
                duration=float(duration or 60.0),
                angle=float(angle or 0.0),
                #fill with fake data
                #count_1=count_1,
                #count_2=count_2,
                #coincidences=coincidences,
                #fill with real data
                count_1 = res['C1'],
                count_2 = res['C2'],
                coincidences = res['COINC'],
                person=str(person or ""),
            )
            df = load_measurements(DATA_FILE)
            status = t.get("measurement_saved", "Measurement saved: {} | operator={} | duration={}s | angle={}°").format(row["time"], row["person"], row["duration"], row["angle"])
            return build_home_page(
                status_text=status,
                df=df,
                lang=lang,
            ), n_clicks

        if pathname == "/poisson":
            return build_poisson_page(fit_requested=False, lang=lang), no_update
        if pathname == "/trends":
            return build_trend_page(lang=lang), no_update
        if pathname == "/angle":
            return build_angle_page(fit_requested=False, lang=lang), no_update
        return build_home_page(lang=lang), no_update

    return app


def build_comparison_summary(df: pd.DataFrame) -> Dict[str, str]:
    if df.empty:
        return {
            "last_measurement": "No measurement available yet.",
            "compatibility": "No previous data to compare.",
        }

    last_row = df.iloc[-1]
    previous_rows = df.iloc[:-1]
    last_measurement = (
        f"Temps : {last_row['time']} | Durée : {last_row['duration']} s | Angle : {last_row['angle']}° | "
        f"Taux de comptage du détecteur 1 : {int(last_row['count_1'])} | Taux de comptage du détecteur 2 : {int(last_row['count_2'])} | Coïncidences : {int(last_row['coincidences'])}"
    )

    if previous_rows.empty:
        compatibility = "This is the first measurement, so there is no previous result to compare with."
    else:
        probabilities = []
        for column in ["count_1", "count_2", "coincidences"]:
            values = previous_rows[column].astype(float).dropna().to_numpy()
            if values.size == 0:
                probabilities.append((column, None))
                continue
            mean_value = float(np.mean(values))
            observed = int(last_row[column])
            if mean_value <= 0:
                prob = 1.0 if observed == 0 else 0.0
            else:
                max_count = int(max(5, np.ceil(mean_value) + 5))
                pmf = poisson_pmf(np.arange(max_count + 1), mean_value)
                if observed < len(pmf):
                    prob = float(pmf[observed])
                else:
                    prob = 0.0
            probabilities.append((column, prob))

        probability_text = ", ".join(
            f"P({name}={int(last_row[name])}) = {prob * 100:.2f}%"
            if prob is not None else f"P({name}={int(last_row[name])}) = n/a"
            for name, prob in probabilities
        )
        compatibility = f"Compatibilité avec les mesures précédentes : {probability_text}."

    return {"last_measurement": last_measurement, "compatibility": compatibility}


def build_poisson_fit_result(df: Union[pd.DataFrame, Path, str], column: str, title: str) -> Dict[str, Any]:
    if not isinstance(df, pd.DataFrame):
        df = load_measurements(df)
    if df.empty or column not in df.columns:
        return {"title": title, "mu": 0.0, "chi2_probability": 1.0, "expected_counts": [], "bin_centers": []}

    filtered_df = _filter_poisson_measurements(df)
    if filtered_df.empty:
        return {"title": title, "mu": 0.0, "chi2_probability": 1.0, "expected_counts": [], "bin_centers": []}

    filtered_df = filtered_df.copy()
    filtered_df["time"] = pd.to_datetime(filtered_df["time"], errors="coerce")
    filtered_df = filtered_df.dropna(subset=["time"]).sort_values("time")
    values = filtered_df[column].dropna().astype(float)
    if values.empty:
        return {"title": title, "mu": 0.0, "chi2_probability": 1.0, "expected_counts": [], "bin_centers": []}

    mu = float(values.mean())
    observed_counts = np.rint(values).astype(int).to_numpy()
    max_observed = int(observed_counts.max())
    fit_max = max(max_observed, int(np.ceil(mu + 6 * np.sqrt(mu))), 10)
    count_values = np.arange(fit_max + 1)
    probabilities = poisson_pmf(count_values, mu)
    expected_counts = len(values) * probabilities

    observed_histogram = np.bincount(observed_counts, minlength=fit_max + 1)
    tail_expected = len(values) * max(0.0, 1.0 - float(probabilities.sum()))
    observed_for_chi2 = np.append(observed_histogram, 0)
    expected_for_chi2 = np.append(expected_counts, tail_expected)
    non_zero = expected_for_chi2 > 0
    chi2_stat = 0.0
    if np.any(non_zero):
        observed = observed_for_chi2[non_zero]
        expected = expected_for_chi2[non_zero]
        chi2_stat = float(np.sum(np.square(observed - expected) / expected))

    dof = max(1, int(np.count_nonzero(non_zero)) - 2)
    chi2_probability = _chi2_survival_function(chi2_stat, dof)

    return {
        "title": title,
        "mu": mu,
        "chi2_probability": chi2_probability,
        "expected_counts": expected_counts,
        "bin_centers": count_values,
    }


def _integer_count_axis_range(values: pd.Series):
    return [float(np.floor(values.min()) - 0.5), float(np.ceil(values.max()) + 0.5)]


def build_static_histogram_figure(df: Union[pd.DataFrame, Path, str], column: str, title: str, fit_result: Optional[Dict[str, Any]] = None, lang: str = "fr"):
    if not isinstance(df, pd.DataFrame):
        df = load_measurements(df)
    t = TRANSLATIONS.get(lang, TRANSLATIONS["fr"])
    if df.empty or column not in df.columns:
        return px.scatter(title=t.get("no_data", "No data yet"))

    filtered_df = _filter_poisson_measurements(df)
    if filtered_df.empty:
        return px.scatter(title=t["no_60s"].format(title))

    filtered_df = filtered_df.copy()
    filtered_df["time"] = pd.to_datetime(filtered_df["time"], errors="coerce")
    filtered_df = filtered_df.dropna(subset=["time"]).sort_values("time")
    if filtered_df.empty:
        return px.scatter(title=f"Aucun horodatage valide pour {title}")

    values = filtered_df[column].dropna().astype(float)
    if values.empty:
        return px.scatter(title=t["no_60s"].format(title))

    x_range = _integer_count_axis_range(values)
    fig = go.Figure(data=[go.Histogram(
        x=values,
        name=title,
        xbins=dict(start=x_range[0], end=x_range[1], size=1),
    )])
    fig.update_layout(title=t["histogram_title"].format(title))
    expected_counts = fit_result.get("expected_counts") if fit_result is not None else None
    if expected_counts is not None and len(expected_counts) > 0:
        fig.add_trace(go.Scatter(
            x=fit_result["bin_centers"],
            y=fit_result["expected_counts"],
            mode="lines",
            name="Poisson fit",
            line=dict(color="#d9534f", width=2),
        ))
    fig.update_layout(
        xaxis_title=title,
        yaxis_title="Nombre",
        xaxis=dict(range=x_range),
        template="plotly_white",
    )
    return fig


def build_histogram_figure(df: Union[pd.DataFrame, Path, str], column: str, title: str, fit_result: Optional[Dict[str, Any]] = None, lang: str = "fr"):
    if not isinstance(df, pd.DataFrame):
        df = load_measurements(df)
    t = TRANSLATIONS.get(lang, TRANSLATIONS["fr"])
    if df.empty or column not in df.columns:
        return px.scatter(title=t.get("no_data", "No data yet"))

    filtered_df = _filter_poisson_measurements(df)
    if filtered_df.empty:
        return px.scatter(title=t["no_60s"].format(title))

    filtered_df = filtered_df.copy()
    filtered_df["time"] = pd.to_datetime(filtered_df["time"], errors="coerce")
    filtered_df = filtered_df.dropna(subset=["time"]).sort_values("time")
    if filtered_df.empty:
        return px.scatter(title=f"No valid timestamps for {title}")

    values = filtered_df[column].dropna().astype(float)
    if values.empty:
        return px.scatter(title=t["no_60s"].format(title))

    mean_value = float(values.mean())
    std_value = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    uncertainty = std_value / np.sqrt(len(values)) if len(values) > 1 else 0.0
    x_range = _integer_count_axis_range(values)
    expected_counts = fit_result.get("expected_counts") if fit_result is not None else None
    histogram_bins = dict(start=x_range[0], end=x_range[1], size=1)

    if len(filtered_df) == 1:
        frames = [go.Frame(data=[go.Histogram(x=[filtered_df.iloc[0][column]], xbins=histogram_bins)], name=str(filtered_df.iloc[0]["time"]))]
    else:
        frames = []
        for index, row in filtered_df.iterrows():
            frame_values = filtered_df.loc[filtered_df["time"] <= row["time"], column].dropna().astype(float)
            frames.append(go.Frame(
                data=[go.Histogram(x=frame_values, xbins=histogram_bins)],
                name=row["time"].strftime("%Y-%m-%d %H:%M:%S"),
            ))

    histogram_values = [frame.data[0].x for frame in frames] if frames else []
    max_bin_count = 0
    if histogram_values:
        for values in histogram_values:
            counts, _ = np.histogram(values, bins=np.arange(x_range[0], x_range[1] + 1.0, 1))
            max_bin_count = max(max_bin_count, int(counts.max()) if counts.size else 0)
    if max_bin_count == 0:
        max_bin_count = 1

    fig = go.Figure(
        data=[go.Histogram(x=filtered_df.iloc[[0]][column].dropna().astype(float), xbins=histogram_bins)],
        frames=frames,
    )
    if expected_counts is not None and len(expected_counts) > 0:
        fig.add_trace(go.Scatter(
            x=fit_result["bin_centers"],
            y=fit_result["expected_counts"],
            mode="lines",
            name="Poisson fit",
            line=dict(color="#d9534f", width=2),
        ))
    fig.update_layout(
        title=t["histogram_over_time"].format(title),
        xaxis_title=title,
        yaxis_title="Count",
        xaxis=dict(range=x_range),
        yaxis=dict(range=[0, max_bin_count + 1]),
        template="plotly_white",
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.1,
            "y": 1.15,
            "xanchor": "left",
            "yanchor": "top",
            "buttons": [{
                "label": "Lire",
                "method": "animate",
                "args": [None, {"frame": {"duration": 800, "redraw": True}, "fromcurrent": True, "transition": {"duration": 200}}],
            }],
        }],
        sliders=[{
            "steps": [{"method": "animate", "args": [[frame.name], {"mode": "immediate", "frame": {"duration": 800, "redraw": True}, "transition": {"duration": 200}}], "label": frame.name} for frame in frames],
            "active": 0,
            "currentvalue": {"prefix": "Date : "},
        }],
    )
    fig.add_annotation(
        x=0.5,
        y=1.02,
        xref="paper",
        yref="paper",
        text=f"Moyenne = {mean_value:.2f} ± {uncertainty:.2f}",
        showarrow=False,
        align="center",
    )
    return fig


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="127.0.0.1", port=8050, use_reloader=False)
