import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import detector_app


def test_append_measurement_creates_csv(tmp_path):
    csv_path = tmp_path / "measurements.csv"
    row = detector_app.append_measurement(
        csv_path=csv_path,
        duration=10.0,
        angle=15.0,
        count_1=4,
        count_2=5,
        coincidences=2,
    )

    assert row["duration"] == 10.0
    assert row["angle"] == 15.0
    assert row["count_1"] == 4

    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert list(df.columns) == [
        "time",
        "person",
        "duration",
        "angle",
        "count_1",
        "count_2",
        "coincidences",
    ]


def test_poisson_summary_uses_mean_and_pmf(tmp_path):
    csv_path = tmp_path / "measurements.csv"
    detector_app.append_measurement(csv_path, 5.0, 10.0, 3, 2, 1)
    detector_app.append_measurement(csv_path, 5.0, 20.0, 4, 4, 2)
    detector_app.append_measurement(csv_path, 5.0, 30.0, 5, 3, 3)

    summary = detector_app.build_poisson_summary(csv_path)
    assert summary["mean_count"] > 0
    assert summary["pmf_series"].shape[0] >= 1
    assert summary["observed_counts"].shape[0] == 3


def test_rate_trend_plot_uses_cumulative_mean(tmp_path):
    csv_path = tmp_path / "measurements.csv"
    detector_app.append_measurement(csv_path, 10.0, 0.0, 20, 10, 5)
    detector_app.append_measurement(csv_path, 20.0, 10.0, 30, 20, 10)

    fig = detector_app.build_rate_trend_plot(csv_path)
    assert len(fig.data) == 3
    assert len(fig.data[0].y) == 2
    assert fig.data[0].y[1] != fig.data[0].y[0]


def test_comparison_summary_reports_probabilities_from_histograms(tmp_path):
    csv_path = tmp_path / "measurements.csv"
    detector_app.append_measurement(csv_path, 10.0, 0.0, 3, 2, 1)
    detector_app.append_measurement(csv_path, 10.0, 10.0, 4, 4, 2)
    detector_app.append_measurement(csv_path, 10.0, 20.0, 5, 3, 3)

    summary = detector_app.build_comparison_summary(detector_app.load_measurements(csv_path))
    assert "P(count_1" in summary["compatibility"]
    assert "%" in summary["compatibility"]


def test_home_page_initial_layout_contains_measurement_controls():
    app = detector_app.create_app()
    layout = app.layout

    def find_component(component, target_id):
        if getattr(component, "id", None) == target_id:
            return component
        children = getattr(component, "children", None)
        if children is None:
            return None
        if isinstance(children, (list, tuple)):
            for child in children:
                result = find_component(child, target_id)
                if result is not None:
                    return result
        else:
            return find_component(children, target_id)
        return None

    assert find_component(layout, "person") is not None
    assert find_component(layout, "duration") is not None
    assert find_component(layout, "angle") is not None
    assert find_component(layout, "run-button") is not None


def test_poisson_page_contains_fit_link():
    page = detector_app.build_poisson_page(detector_app.load_measurements(detector_app.DATA_FILE))
    assert "Activer l'ajustement" in str(page)


def test_poisson_fit_result_reports_nonzero_chi2_probability(tmp_path):
    csv_path = tmp_path / "measurements.csv"
    detector_app.append_measurement(csv_path, 10.0, 0.0, 2, 1, 1)
    detector_app.append_measurement(csv_path, 10.0, 10.0, 3, 2, 2)
    detector_app.append_measurement(csv_path, 10.0, 20.0, 4, 3, 3)

    fit_result = detector_app.build_poisson_fit_result(csv_path, "coincidences", "Coincidences")
    assert fit_result["chi2_probability"] > 0


def test_angle_plot_builds_with_uncertainty(tmp_path):
    csv_path = tmp_path / "measurements.csv"
    detector_app.append_measurement(csv_path, 10.0, 0.0, 3, 2, 1)
    detector_app.append_measurement(csv_path, 10.0, 10.0, 4, 3, 2)
    detector_app.append_measurement(csv_path, 10.0, 20.0, 5, 4, 3)

    fig = detector_app.build_angle_plot(csv_path)
    assert len(fig.data) == 1
    assert len(fig.data[0].y) == 3


def test_histogram_figure_is_animated(tmp_path):
    csv_path = tmp_path / "measurements.csv"
    detector_app.append_measurement(csv_path, 10.0, 0.0, 3, 2, 1)
    detector_app.append_measurement(csv_path, 10.0, 10.0, 4, 3, 2)
    detector_app.append_measurement(csv_path, 10.0, 20.0, 5, 4, 3)

    fig = detector_app.build_histogram_figure(csv_path, "coincidences", "Coincidences")
    assert len(fig.frames) >= 2
    assert fig.layout.updatemenus
    assert fig.layout.sliders


def test_histogram_figure_adapts_y_axis_to_data(tmp_path):
    csv_path = tmp_path / "measurements.csv"
    detector_app.append_measurement(csv_path, 10.0, 0.0, 3, 2, 1)
    detector_app.append_measurement(csv_path, 10.0, 10.0, 4, 3, 2)
    detector_app.append_measurement(csv_path, 10.0, 20.0, 5, 4, 3)

    fig = detector_app.build_histogram_figure(csv_path, "coincidences", "Coincidences")
    assert fig.layout.yaxis.range is not None
    assert fig.layout.yaxis.range[1] > fig.layout.yaxis.range[0]


def test_poisson_page_shows_static_and_animated_histograms():
    page = detector_app.build_poisson_page(detector_app.load_measurements(detector_app.DATA_FILE))

    graph_components = []

    def collect_graphs(component):
        if isinstance(component, detector_app.dcc.Graph):
            graph_components.append(component)
        children = getattr(component, "children", None)
        if children is None:
            return
        if isinstance(children, (list, tuple)):
            for child in children:
                collect_graphs(child)
        else:
            collect_graphs(children)

    collect_graphs(page)
    assert len(graph_components) >= 6


def test_poisson_page_fit_button_and_results():
    page = detector_app.build_poisson_page(detector_app.load_measurements(detector_app.DATA_FILE), fit_requested=True)

    def find_text(component, target_text):
        if isinstance(component, str) and target_text in component:
            return True
        children = getattr(component, "children", None)
        if children is None:
            return False
        if isinstance(children, (list, tuple)):
            return any(find_text(child, target_text) for child in children)
        return find_text(children, target_text)

    assert find_text(page, "Paramètre ajusté") is True
    assert find_text(page, "Probabilité χ²") is True
