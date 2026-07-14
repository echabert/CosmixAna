import math
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:
    from scipy.stats import chi2 as chi2_dist
except ImportError:  # pragma: no cover - optional dependency fallback
    chi2_dist = None

from dash import Dash, dcc, html, Input, Output, State, dash_table


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "detector_measurements.csv"

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
        "last_measurement_summary": "Last measurement summary",
        "run_measurement_hint": "Run a new measurement to update the summary.",
        "duration_label": "Duration (s): ",
        "person_label": "Operator name: ",
        "person_placeholder": "Enter your name",
        "angle_label": "Angle (deg): ",
        "run_button": "Run measurement",
        "latest_measurements": "Latest Measurements",
        "poisson_h2": "Poisson law check",
        "current_distributions": "Current distributions",
        "activate_fit": "Activate fit",
        "click_fit_hint": "Click the fit button to overlay a Poisson fit on each histogram.",
        "fitted_parameter": "Fitted parameter: μ = {}",
        "chi2_probability": "χ² probability: {}",
        "angle_fit_title": "A·cos(θ)^n + B fit",
        "angle_fit_function": "Model: f(θ) = A·cos(θ)^n + B",
        "angle_fit_summary": "Fit result: A = {} ± {} | n = {} ± {} | B = {} ± {}",
        "animated_history": "Animated history",
        "trend_h2": "Trend plots",
        "angle_h2": "Angle dependence",
        "no_data": "No data yet",
        "no_10s": "No 10-second runs for {}",
        "no_timestamps": "No valid timestamps for {}",
        "histogram_title": "Histogram of {}",
        "histogram_over_time": "Histogram of {} over time",
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
        "last_measurement_summary": "Résumé de la dernière mesure",
        "run_measurement_hint": "Exécutez une nouvelle mesure pour mettre à jour le résumé.",
        "duration_label": "Durée (s) : ",
        "person_label": "Nom de l'opérateur : ",
        "person_placeholder": "Entrez votre nom",
        "angle_label": "Angle (°) : ",
        "run_button": "Exécuter la mesure",
        "latest_measurements": "Dernières mesures",
        "poisson_h2": "Vérification de la loi de Poisson",
        "current_distributions": "Distributions actuelles",
        "activate_fit": "Activer l'ajustement",
        "click_fit_hint": "Cliquez sur le bouton d'ajustement pour superposer un ajustement de Poisson sur chaque histogramme.",
        "fitted_parameter": "Paramètre ajusté : μ = {}",
        "chi2_probability": "Probabilité χ² : {}",
        "angle_fit_title": "A·cos(θ)^n + B ajustement",
        "angle_fit_function": "Modèle : f(θ) = A·cos(θ)^n + B",
        "angle_fit_summary": "Résultat de l'ajustement : A = {} ± {} | n = {} ± {} | B = {} ± {}",
        "animated_history": "Historique animé",
        "trend_h2": "Graphiques d'évolution en fonction du temps",
        "angle_h2": "Dépendance angulaire",
        "no_data": "Pas encore de données",
        "no_10s": "Aucune prise de 10 secondes pour {}",
        "no_timestamps": "Aucun horodatage valide pour {}",
        "histogram_title": "Histogramme de {}",
        "histogram_over_time": "Histogramme de {} au fil du temps",
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


def build_trend_plot(csv_path: Union[Path, str] = DATA_FILE, lang: str = "fr"):
    t = TRANSLATIONS.get(lang, TRANSLATIONS["fr"])
    df = load_measurements(csv_path)
    if df.empty:
        return px.scatter(title=t.get("no_data", "Pas encore de données"))
    df = df.copy()
    df["index"] = np.arange(1, len(df) + 1)
    fig = px.line(
        df,
        x="index",
        y=["count_1", "count_2", "coincidences"],
        title=t.get("trend_h2", "Évolutions des décomptes dans le temps"),
        markers=True,
    )
    fig.update_layout(template="plotly_white")
    return fig


def build_rate_trend_plot(csv_path: Union[Path, str] = DATA_FILE, lang: str = "fr"):
    df = load_measurements(csv_path)
    t = TRANSLATIONS.get(lang, TRANSLATIONS["fr"])
    if df.empty:
        return px.scatter(title=t.get("no_data", "Pas encore de données"))

    df = df.copy()
    df["index"] = np.arange(1, len(df) + 1)
    df["count_1_rate"] = df["count_1"] / df["duration"]
    df["count_2_rate"] = df["count_2"] / df["duration"]
    df["coincidences_rate"] = df["coincidences"] / df["duration"]

    fig = go.Figure()
    for column, name in [("count_1_rate", f"{t.get('count1_title')}/s"), ("count_2_rate", f"{t.get('count2_title')}/s"), ("coincidences_rate", "Coïncidences/s")]:
        values = df[column].astype(float)
        running_means = []
        running_uncertainties = []
        for i in range(len(values)):
            subset = values.iloc[: i + 1]
            if len(subset) == 1:
                running_means.append(float(subset.iloc[0]))
                running_uncertainties.append(0.0)
            else:
                mean_value = float(subset.mean())
                std_value = float(subset.std(ddof=1)) if len(subset) > 1 else 0.0
                uncertainty = std_value / np.sqrt(len(subset)) if len(subset) > 1 else 0.0
                running_means.append(mean_value)
                running_uncertainties.append(uncertainty)

        fig.add_trace(go.Scatter(
            x=df["index"],
            y=running_means,
            mode="lines+markers",
            name=name,
            error_y=dict(type="data", array=running_uncertainties, visible=True),
        ))

    fig.update_layout(
        title=t.get("rate_trend_title", "Moyenne cumulée des décomptes par seconde"),
        xaxis_title=t.get("run_number_label", "Numéro de la prise"),
        yaxis_title=t.get("rate_ylabel", "Décomptes/s"),
        template="plotly_white",
    )
    return fig


def build_angle_fit_result(df: Union[pd.DataFrame, Path, str], lang: str = "fr") -> Dict[str, Any]:
    if not isinstance(df, pd.DataFrame):
        df = load_measurements(df)
    if df.empty:
        return {"A": 0.0, "n": 0.0, "B": 0.0, "A_unc": 0.0, "n_unc": 0.0, "B_unc": 0.0, "angle": np.array([]), "fit_rate": np.array([])}

    df = df.copy()
    df["coincidences_per_sec"] = df["coincidences"] / df["duration"]
    grouped = df.groupby("angle")["coincidences_per_sec"]
    angle_groups = pd.DataFrame({
        "angle": list(grouped.groups.keys()),
        "mean_rate": [float(values.mean()) for _, values in grouped],
    }).sort_values("angle")

    angles = angle_groups["angle"].to_numpy(dtype=float)
    rates = angle_groups["mean_rate"].to_numpy(dtype=float)
    if len(angles) == 0:
        return {"A": 0.0, "n": 0.0, "B": 0.0, "A_unc": 0.0, "n_unc": 0.0, "B_unc": 0.0, "angle": angles, "fit_rate": np.array([])}

    cos_theta = np.cos(np.deg2rad(angles))

    def fit_for_n(n_value: float) -> Optional[Dict[str, Any]]:
        x = np.sign(cos_theta) * np.power(np.abs(cos_theta), n_value)
        design = np.vstack([x, np.ones_like(x)]).T
        try:
            params, residuals, rank, s = np.linalg.lstsq(design, rates, rcond=None)
        except np.linalg.LinAlgError:
            return None
        A_val, B_val = float(params[0]), float(params[1])
        fit = A_val * x + B_val
        ssr = float(np.sum((rates - fit) ** 2))
        return {"A": A_val, "B": B_val, "n": float(n_value), "ssr": ssr, "x": x}

    best_fit = None
    for n_value in np.linspace(0.0, 5.0, 101):
        result = fit_for_n(n_value)
        if result is None:
            continue
        if best_fit is None or result["ssr"] < best_fit["ssr"]:
            best_fit = result

    if best_fit is None:
        return {"A": 0.0, "n": 0.0, "B": 0.0, "A_unc": 0.0, "n_unc": 0.0, "B_unc": 0.0, "angle": angles, "fit_rate": np.array([])}

    design = np.vstack([best_fit["x"], np.ones_like(best_fit["x"]) ]).T
    dof = len(rates) - 2
    sigma2 = best_fit["ssr"] / dof if dof > 0 else 0.0
    cov = np.zeros((2, 2), dtype=float)
    if dof > 0:
        xtx = design.T.dot(design)
        if np.linalg.cond(xtx) < 1e12:
            cov = sigma2 * np.linalg.inv(xtx)

    A_unc = float(math.sqrt(cov[0, 0])) if cov[0, 0] >= 0 else 0.0
    B_unc = float(math.sqrt(cov[1, 1])) if cov[1, 1] >= 0 else 0.0

    n_unc = 0.0
    dense_n = np.linspace(max(0.0, best_fit["n"] - 0.25), min(5.0, best_fit["n"] + 0.25), 21)
    ssr_values = []
    for n_value in dense_n:
        result = fit_for_n(n_value)
        ssr_values.append(result["ssr"] if result is not None else float("inf"))
    min_index = int(np.argmin(ssr_values))
    if 0 < min_index < len(ssr_values) - 1:
        h = dense_n[1] - dense_n[0]
        second_derivative = (ssr_values[min_index + 1] - 2.0 * ssr_values[min_index] + ssr_values[min_index - 1]) / (h * h)
        if second_derivative > 0:
            n_unc = float(math.sqrt(2.0 / second_derivative))

    fit_rate = best_fit["A"] * best_fit["x"] + best_fit["B"]
    return {
        "A": best_fit["A"],
        "n": best_fit["n"],
        "B": best_fit["B"],
        "A_unc": A_unc,
        "n_unc": n_unc,
        "B_unc": B_unc,
        "angle": angles,
        "fit_rate": fit_rate,
    }


def build_angle_plot(csv_path: Union[Path, str] = DATA_FILE, fit_result: Optional[Dict[str, Any]] = None, lang: str = "fr"):
    df = load_measurements(csv_path)
    t = TRANSLATIONS.get(lang, TRANSLATIONS["fr"])
    if df.empty:
        return px.scatter(title=t.get("no_data", "Pas encore de données"))

    df = df.copy()
    df["coincidences_per_sec"] = df["coincidences"] / df["duration"]

    grouped = df.groupby("angle")["coincidences_per_sec"]
    angle_groups = pd.DataFrame({
        "angle": list(grouped.groups.keys()),
        "mean_rate": [float(values.mean()) for _, values in grouped],
        "std_rate": [float(values.std(ddof=1)) if len(values) > 1 else 0.0 for _, values in grouped],
        "count": [int(len(values)) for _, values in grouped],
    })
    angle_groups["uncertainty"] = np.where(
        angle_groups["count"] > 1,
        angle_groups["std_rate"] / np.sqrt(angle_groups["count"]),
        0.0,
    )
    angle_summary = angle_groups.sort_values("angle")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=angle_summary["angle"],
        y=angle_summary["mean_rate"],
        mode="lines+markers",
        name=f"{t.get('coincidences_title','Coïncidences')}/s",
        error_y=dict(type="data", array=angle_summary["uncertainty"], visible=True),
    ))
    if fit_result is not None and len(fit_result.get("angle", [])) > 0:
        fig.add_trace(go.Scatter(
            x=fit_result["angle"],
            y=fit_result["fit_rate"],
            mode="lines",
            name=t.get("angle_fit_title", "A·cos(θ)^n fit"),
            line=dict(color="#d9534f"),
        ))
    fig.update_layout(
        title=t.get("angle_plot_title", "Moyenne du taux de coïncidences/s en fonction de l'angle"),
        xaxis_title=t.get("angle_label", "Angle (°)"),
        yaxis_title=t.get("angle_ylabel", "Taux de coïncidences/s"),
        template="plotly_white",
    )
    return fig


def build_measurement_controls(visible: bool = True, lang: str = "fr"):
    t = TRANSLATIONS.get(lang, TRANSLATIONS["fr"])
    return html.Div(
        style={"display": "flex" if visible else "none", "gap": "12px", "flexWrap": "wrap", "alignItems": "flex-end", "marginBottom": "16px"},
        children=[
            html.Div(
                style={"display": "flex", "flexDirection": "column", "minWidth": "180px"},
                children=[html.Label(t["person_label"]), dcc.Input(id="person", type="text", value="", placeholder=t["person_placeholder"], style={"width": "100%"})],
            ),
            html.Div(
                style={"display": "flex", "flexDirection": "column", "minWidth": "120px"},
                children=[html.Label(t["duration_label"]), dcc.Input(id="duration", type="number", value=10, min=1, step=1, style={"width": "80px"})],
            ),
            html.Div(
                style={"display": "flex", "flexDirection": "column", "minWidth": "120px"},
                children=[html.Label(t["angle_label"]), dcc.Dropdown(
                    id="angle",
                    options=[{"label": str(v), "value": v} for v in [0, 10, 20, 30, 40]],
                    value=0,
                    clearable=False,
                    style={"width": "100%"},
                )],
            ),
            html.Button(t["run_button"], id="run-button", n_clicks=0, style={"height": "40px"}),
        ],
    )


def build_home_page(status_text: str = None, df: pd.DataFrame = None, lang: str = "fr"):
    if df is None:
        df = load_measurements(DATA_FILE)
    t = TRANSLATIONS.get(lang, TRANSLATIONS["fr"])
    if status_text is None:
        status_text = t["waiting_first"]
    return html.Div(
        style={"padding": "24px", "fontFamily": "Arial, sans-serif"},
        children=[
            html.H2(t["home_h2"]),
            html.P(t["home_p"]),
            html.Div(
                style={"marginBottom": "16px"},
                children=[
                    dcc.Link(t["link_home"], href="/", style={"marginRight": "12px"}),
                    dcc.Link(t["link_poisson"], href="/poisson", style={"marginRight": "12px"}),
                    dcc.Link(t["link_trends"], href="/trends", style={"marginRight": "12px"}),
                    dcc.Link(t["link_angle"], href="/angle"),
                ],
            ),
            html.Div(id="status-output", style={"marginTop": "12px", "fontWeight": "bold"}, children=status_text),
            html.Hr(),
            html.Div(
                style={"marginBottom": "16px", "padding": "12px", "border": "1px solid #ddd", "borderRadius": "6px"},
                children=[
                    html.B(t["last_measurement_summary"]),
                    html.Div(t["run_measurement_hint"]),
                ],
            ),
            build_measurement_controls(visible=True, lang=lang),
            html.H3(t["latest_measurements"]),
            dcc.Loading(children=[
                dash_table.DataTable(
                    id="measurements-table",
                    data=df.tail(10).to_dict("records"),
                    columns=[{"name": col, "id": col} for col in df.columns],
                    page_size=10,
                    style_table={"overflowX": "auto"},
                    style_cell={"padding": "8px", "textAlign": "left"},
                    style_header={"backgroundColor": "#f2f2f2", "fontWeight": "bold"},
                )
            ]),
        ],
    )


def build_poisson_page(df: pd.DataFrame = None, fit_requested: bool = False, lang: str = "fr"):
    if df is None:
        df = load_measurements(DATA_FILE)
    t = TRANSLATIONS.get(lang, TRANSLATIONS["fr"])
    summary = build_comparison_summary(df)

    fit_results = []
    titles = [t.get("count1_title"), t.get("count2_title"), t.get("coincidences_title")]
    if fit_requested:
        for (column, _), title in zip([("count_1", ""), ("count_2", ""), ("coincidences", "")], titles):
            fit_results.append(build_poisson_fit_result(df, column, title))

    static_histograms = [
        dcc.Graph(figure=build_static_histogram_figure(df, "count_1", titles[0], fit_result=fit_results[0] if fit_requested else None)),
        dcc.Graph(figure=build_static_histogram_figure(df, "count_2", titles[1], fit_result=fit_results[1] if fit_requested else None)),
        dcc.Graph(figure=build_static_histogram_figure(df, "coincidences", titles[2], fit_result=fit_results[2] if fit_requested else None)),
    ]
    animated_histograms = [
        dcc.Graph(figure=build_histogram_figure(df, "count_1", titles[0])),
        dcc.Graph(figure=build_histogram_figure(df, "count_2", titles[1])),
        dcc.Graph(figure=build_histogram_figure(df, "coincidences", titles[2])),
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
                style={"padding": "12px", "border": "1px dashed #ccc", "borderRadius": "6px", "color": "#666"},
                children="Cliquez sur le bouton d'ajustement pour superposer un ajustement de Poisson sur chaque histogramme.",
            )
        ]

    return html.Div(
        style={"padding": "24px", "fontFamily": "Arial, sans-serif"},
        children=[
            html.H2(t["poisson_h2"]),
            html.Div(
                style={"marginBottom": "16px"},
                children=[
                    dcc.Link(t["link_home"], href="/", style={"marginRight": "12px"}),
                    dcc.Link(t["link_poisson"], href="/poisson", style={"marginRight": "12px"}),
                    dcc.Link(t["link_trends"], href="/trends", style={"marginRight": "12px"}),
                    dcc.Link(t["link_angle"], href="/angle"),
                ],
            ),
            html.Div(
                style={"marginBottom": "16px", "padding": "12px", "border": "1px solid #ddd", "borderRadius": "6px"},
                children=[
                    html.B(t["last_measurement_summary"]),
                    html.Div(summary["last_measurement"]),
                    html.Div(summary["compatibility"]),
                ],
            ),
            build_measurement_controls(visible=False, lang=lang),
            html.H3(t["current_distributions"]),
            html.Div(
                style={"marginBottom": "16px"},
                children=[
                    dcc.Link(
                        t["activate_fit"],
                        href="/poisson?fit=1",
                        style={
                            "display": "inline-block",
                            "padding": "8px 12px",
                            "border": "1px solid #2b6cb0",
                            "borderRadius": "6px",
                            "backgroundColor": "#ebf8ff",
                            "color": "#2b6cb0",
                            "textDecoration": "none",
                        },
                    )
                ],
            ),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))", "gap": "16px", "marginBottom": "24px"},
                children=static_histograms,
            ),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))", "gap": "12px", "marginBottom": "24px"},
                children=fit_summary,
            ),
            html.H3("Historique animé"),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))", "gap": "16px"},
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
        style={"padding": "24px", "fontFamily": "Arial, sans-serif"},
        children=[
            html.H2(t.get("trend_h2", "Graphiques d'évolution")),
            html.Div(
                style={"marginBottom": "16px"},
                children=[
                    dcc.Link(t["link_home"], href="/", style={"marginRight": "12px"}),
                    dcc.Link(t["link_poisson"], href="/poisson", style={"marginRight": "12px"}),
                    dcc.Link(t["link_trends"], href="/trends", style={"marginRight": "12px"}),
                    dcc.Link(t["link_angle"], href="/angle"),
                ],
            ),
            html.Div(
                style={"marginBottom": "16px", "padding": "12px", "border": "1px solid #ddd", "borderRadius": "6px"},
                children=[
                    html.B(t["last_measurement_summary"]),
                    html.Div(summary["last_measurement"]),
                    html.Div(summary["compatibility"]),
                ],
            ),
            build_measurement_controls(visible=False, lang=lang),
            dcc.Graph(figure=build_trend_plot(DATA_FILE, lang=lang)),
            html.Br(),
            dcc.Graph(figure=build_rate_trend_plot(DATA_FILE, lang=lang)),
        ],
    )


def build_angle_page(df: pd.DataFrame = None, fit_requested: bool = False, lang: str = "fr"):
    if df is None:
        df = load_measurements(DATA_FILE)
    t = TRANSLATIONS.get(lang, TRANSLATIONS["fr"])
    summary = build_comparison_summary(df)
    fit_result = build_angle_fit_result(df, lang=lang) if fit_requested else None
    return html.Div(
        style={"padding": "24px", "fontFamily": "Arial, sans-serif"},
        children=[
            html.H2(t.get("angle_h2", "Dépendance angulaire")),
            html.Div(
                style={"marginBottom": "16px"},
                children=[
                    dcc.Link(t["link_home"], href="/", style={"marginRight": "12px"}),
                    dcc.Link(t["link_poisson"], href="/poisson", style={"marginRight": "12px"}),
                    dcc.Link(t["link_trends"], href="/trends", style={"marginRight": "12px"}),
                    dcc.Link(t["link_angle"], href="/angle"),
                ],
            ),
            html.Div(
                style={"marginBottom": "16px", "padding": "12px", "border": "1px solid #ddd", "borderRadius": "6px"},
                children=[
                    html.B(t["last_measurement_summary"]),
                    html.Div(summary["last_measurement"]),
                    html.Div(summary["compatibility"]),
                ],
            ),
            html.Div(
                style={"marginBottom": "16px"},
                children=[
                    dcc.Link(
                        t["activate_fit"],
                        href="/angle?fit=1",
                        style={
                            "display": "inline-block",
                            "padding": "8px 12px",
                            "border": "1px solid #2b6cb0",
                            "borderRadius": "6px",
                            "backgroundColor": "#ebf8ff",
                            "color": "#2b6cb0",
                            "textDecoration": "none",
                        },
                    )
                ],
            ),
            html.Div(
                style={"padding": "12px", "border": "1px dashed #ccc", "borderRadius": "6px", "color": "#666", "marginBottom": "16px"},
                children=[
                    html.Div(t["click_fit_hint"]),
                    html.Div(t["angle_fit_function"]) if fit_requested else None,
                    html.Div(
                        t["angle_fit_summary"].format(
                            f"{fit_result['A']:.2f}",
                            f"{fit_result['A_unc']:.2f}",
                            f"{fit_result['n']:.2f}",
                            f"{fit_result['n_unc']:.2f}",
                            f"{fit_result['B']:.2f}",
                            f"{fit_result['B_unc']:.2f}",
                        )
                    ) if fit_requested else None,
                ],
            ),
            build_measurement_controls(visible=False, lang=lang),
            dcc.Graph(figure=build_angle_plot(DATA_FILE, fit_result=fit_result, lang=lang)),
        ],
    )


def create_app() -> Dash:
    app = Dash(__name__, title=TRANSLATIONS.get("fr")["app_title"])
    app.layout = html.Div(
        style={"padding": "24px", "fontFamily": "Arial, sans-serif"},
        children=[
            dcc.Location(id="url", refresh=False),
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "12px"},
                children=[
                    html.Label("Lang:"),
                    dcc.Dropdown(
                        id="lang-select",
                        options=[{"label": "Français", "value": "fr"}, {"label": "English", "value": "en"}],
                        value="fr",
                        clearable=False,
                        style={"width": "160px"},
                    ),
                ],
            ),
            html.Div(id="page-content", children=build_home_page(lang="fr")),
        ],
    )

    @app.callback(
        Output("page-content", "children"),
        [Input("url", "pathname"), Input("url", "search"), Input("run-button", "n_clicks"), Input("lang-select", "value")],
        [State("person", "value"), State("duration", "value"), State("angle", "value")],
    )
    def update_dashboard(pathname, search, n_clicks, lang, person, duration, angle):
        t = TRANSLATIONS.get(lang, TRANSLATIONS["fr"])
        if pathname == "/poisson" and "fit=1" in (search or ""):
            return build_poisson_page(fit_requested=True, lang=lang)
        if pathname == "/angle" and "fit=1" in (search or ""):
            return build_angle_page(fit_requested=True, lang=lang)

        if n_clicks is not None and n_clicks >= 1:
            factor = float(np.cos(np.deg2rad(float(angle or 0.0)))) ** 2
            count_1 = int(np.random.poisson(max(1.0, 10 + 5 * factor)))
            count_2 = int(np.random.poisson(max(1.0, 8 + 4 * factor)))
            coincidences = int(np.random.poisson(max(0.0, 2 + 10 * factor)))
            row = append_measurement(
                DATA_FILE,
                duration=float(duration or 10.0),
                angle=float(angle or 0.0),
                count_1=count_1,
                count_2=count_2,
                coincidences=coincidences,
                person=str(person or ""),
            )
            df = load_measurements(DATA_FILE)
            status = t.get("measurement_saved", "Measurement saved: {} | operator={} | duration={}s | angle={}°").format(row["time"], row["person"], row["duration"], row["angle"])
            return build_home_page(
                status_text=status,
                df=df,
                lang=lang,
            )

        if pathname == "/poisson":
            return build_poisson_page(fit_requested=False, lang=lang)
        if pathname == "/trends":
            return build_trend_page(lang=lang)
        if pathname == "/angle":
            return build_angle_page(fit_requested=False, lang=lang)
        return build_home_page(lang=lang)

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

    filtered_df = df[df["duration"] == 10].copy()
    if filtered_df.empty:
        return {"title": title, "mu": 0.0, "chi2_probability": 1.0, "expected_counts": [], "bin_centers": []}

    filtered_df = filtered_df.copy()
    filtered_df["time"] = pd.to_datetime(filtered_df["time"], errors="coerce")
    filtered_df = filtered_df.dropna(subset=["time"]).sort_values("time")
    values = filtered_df[column].dropna().astype(float)
    if values.empty:
        return {"title": title, "mu": 0.0, "chi2_probability": 1.0, "expected_counts": [], "bin_centers": []}

    mu = float(values.mean())
    edges = np.linspace(0, 10, 11)
    counts, _ = np.histogram(values, bins=10, range=(0, 10))
    expected_counts = []
    for index in range(len(edges) - 1):
        lower = edges[index]
        upper = edges[index + 1]
        k_values = np.arange(int(np.floor(lower)), int(np.ceil(upper)) + 1)
        k_values = k_values[(k_values >= lower) & (k_values < upper)]
        if index == len(edges) - 2:
            k_values = k_values[k_values <= int(np.ceil(upper))]
        if len(k_values) == 0:
            expected_counts.append(0.0)
            continue
        probabilities = poisson_pmf(k_values, mu)
        expected_counts.append(float(len(values) * np.sum(probabilities)))

    expected_counts = np.array(expected_counts, dtype=float)
    non_zero = expected_counts > 0
    chi2_stat = 0.0
    if np.any(non_zero):
        observed = counts[non_zero]
        expected = expected_counts[non_zero]
        chi2_stat = float(np.sum(np.square(observed - expected) / expected))

    dof = max(1, len(counts) - 2)
    chi2_probability = _chi2_survival_function(chi2_stat, dof)

    return {
        "title": title,
        "mu": mu,
        "chi2_probability": chi2_probability,
        "expected_counts": expected_counts,
        "bin_centers": (edges[:-1] + edges[1:]) / 2,
    }


def build_static_histogram_figure(df: Union[pd.DataFrame, Path, str], column: str, title: str, fit_result: Optional[Dict[str, Any]] = None):
    if not isinstance(df, pd.DataFrame):
        df = load_measurements(df)
    if df.empty or column not in df.columns:
        return px.scatter(title="No data yet")

    filtered_df = df[df["duration"] == 10].copy()
    if filtered_df.empty:
        return px.scatter(title=f"Aucune prise de 10 secondes pour {title}")

    filtered_df = filtered_df.copy()
    filtered_df["time"] = pd.to_datetime(filtered_df["time"], errors="coerce")
    filtered_df = filtered_df.dropna(subset=["time"]).sort_values("time")
    if filtered_df.empty:
        return px.scatter(title=f"Aucun horodatage valide pour {title}")

    values = filtered_df[column].dropna().astype(float)
    if values.empty:
        return px.scatter(title=f"Aucune prise de 10 secondes pour {title}")

    fig = px.histogram(filtered_df, x=column, nbins=10, title=f"Histogramme de {title}")
    expected_counts = fit_result.get("expected_counts") if fit_result is not None else None
    if expected_counts is not None and len(expected_counts) > 0:
        fig.add_trace(go.Scatter(
            x=fit_result["bin_centers"],
            y=fit_result["expected_counts"],
            mode="lines+markers",
            name="Poisson fit",
            line=dict(color="#d9534f"),
        ))
    fig.update_layout(
        xaxis_title=title,
        yaxis_title="Nombre",
        xaxis=dict(range=[0, 10]),
        template="plotly_white",
    )
    return fig


def build_histogram_figure(df: Union[pd.DataFrame, Path, str], column: str, title: str, fit_result: Optional[Dict[str, Any]] = None):
    if not isinstance(df, pd.DataFrame):
        df = load_measurements(df)
    if df.empty or column not in df.columns:
        return px.scatter(title="No data yet")

    filtered_df = df[df["duration"] == 10].copy()
    if filtered_df.empty:
        return px.scatter(title=f"No 10-second runs for {title}")

    filtered_df = filtered_df.copy()
    filtered_df["time"] = pd.to_datetime(filtered_df["time"], errors="coerce")
    filtered_df = filtered_df.dropna(subset=["time"]).sort_values("time")
    if filtered_df.empty:
        return px.scatter(title=f"No valid timestamps for {title}")

    values = filtered_df[column].dropna().astype(float)
    if values.empty:
        return px.scatter(title=f"No 10-second runs for {title}")

    mean_value = float(values.mean())
    std_value = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    uncertainty = std_value / np.sqrt(len(values)) if len(values) > 1 else 0.0

    if len(filtered_df) == 1:
        frames = [go.Frame(data=[go.Histogram(x=[filtered_df.iloc[0][column]], nbinsx=10)], name=str(filtered_df.iloc[0]["time"]))]
    else:
        frames = []
        for index, row in filtered_df.iterrows():
            frame_values = filtered_df.loc[filtered_df["time"] <= row["time"], column].dropna().astype(float)
            frames.append(go.Frame(
                data=[go.Histogram(x=frame_values, nbinsx=10)],
                name=row["time"].strftime("%Y-%m-%d %H:%M:%S"),
            ))

    histogram_values = [frame.data[0].x for frame in frames] if frames else []
    max_bin_count = 0
    if histogram_values:
        for values in histogram_values:
            counts, _ = np.histogram(values, bins=10, range=(0, 10))
            max_bin_count = max(max_bin_count, int(counts.max()) if counts.size else 0)
    if max_bin_count == 0:
        max_bin_count = 1

    fig = go.Figure(
        data=[go.Histogram(x=filtered_df.iloc[[0]][column].dropna().astype(float), nbinsx=10)],
        frames=frames,
    )
    expected_counts = fit_result.get("expected_counts") if fit_result is not None else None
    if expected_counts is not None and len(expected_counts) > 0:
        fig.add_trace(go.Scatter(
            x=fit_result["bin_centers"],
            y=fit_result["expected_counts"],
            mode="lines+markers",
            name="Poisson fit",
            line=dict(color="#d9534f"),
        ))
    fig.update_layout(
        title=f"{title} vs temps",
        xaxis_title=title,
        yaxis_title="Count",
        xaxis=dict(range=[0, 10]),
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
