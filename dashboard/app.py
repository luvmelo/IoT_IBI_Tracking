"""Dash app: dark instrumentation operator + minimalist subject view.

Two routes via the `?view=…` query string:
    /                  → operator (default)
    /?view=subject     → subject (full-screen HR + ping ring)

The app reads from a module-level `DashboardState` injected by
`scripts/run_dashboard.py`. UI never calls the pipeline; it only reads
the most recent `PipelineResult` snapshot.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from dash import Dash, Input, Output, State, dcc, html

from dashboard.figures import COLORS, heartbeat_figure, ibi_figure, phase_figure
from dashboard.state import DashboardState

# Set by `make_app(state)` before `app.run(...)`.
STATE: DashboardState | None = None

_GOOGLE_FONTS = (
    "https://fonts.googleapis.com/css2?"
    "family=DM+Mono:wght@300;400&family=Sora:wght@300;400;500&display=swap"
)


def _format_metric(value: float, *, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "––"
    return f"{value:.{decimals}f}"


def _quality_class(result) -> tuple[str, str]:
    if result is None or result.nn_ms.size < 4:
        return "fair", "warming up"
    coverage = float(np.mean(result.motion_mask))
    if coverage >= 0.85 and result.nn_ms.size >= 8:
        return "good", "good signal"
    if coverage >= 0.5:
        return "fair", "fair signal"
    return "poor", "high motion"


def _format_elapsed(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


# ── Layouts ────────────────────────────────────────────────────────────────


def _header(view: str, mode: str = "REPLAY") -> html.Header:
    other = "operator" if view == "subject" else "subject"
    other_url = "/" if view == "subject" else "/?view=subject"
    return html.Header(
        [
            html.H1([html.Span("mm", className="accent"), "Wave HRV"], className="logo"),
            html.Span(mode, className=f"badge {mode.lower()}"),
            html.Span(id="elapsed", className="timer", children="00:00"),
            html.Span(className="spacer"),
            html.A(f"→ {other} view", href=other_url, className="link"),
        ],
        className="header",
    )


def operator_layout() -> html.Div:
    return html.Div(
        [
            _header("operator"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(id="hr-value", children="––", className="hr-big"),
                                    html.Div("BPM", className="hr-unit"),
                                ],
                                className="card hr-card",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div("SDNN", className="metric-label"),
                                            html.Div(id="sdnn-value", children="–", className="metric-value"),
                                            html.Div("ms", className="metric-unit"),
                                        ],
                                        className="metric",
                                    ),
                                    html.Div(
                                        [
                                            html.Div("RMSSD", className="metric-label"),
                                            html.Div(id="rmssd-value", children="–", className="metric-value"),
                                            html.Div("ms", className="metric-unit"),
                                        ],
                                        className="metric",
                                    ),
                                    html.Div(
                                        [
                                            html.Div("pNN50", className="metric-label"),
                                            html.Div(id="pnn50-value", children="–", className="metric-value"),
                                            html.Div("%", className="metric-unit"),
                                        ],
                                        className="metric",
                                    ),
                                ],
                                className="card metric-row",
                            ),
                            html.Div(
                                [
                                    html.Span(id="quality-pill", className="quality fair"),
                                    html.Span(id="quality-text", children="warming up", className="quality-text"),
                                ],
                                className="card quality-row",
                            ),
                            html.Div(
                                [
                                    html.Div("CHEST RANGE", className="metric-label"),
                                    html.Div(id="chest-range", children="––", className="metric-value"),
                                    html.Div("FRAMES", className="metric-label"),
                                    html.Div(id="frame-count", children="––", className="metric-value"),
                                ],
                                className="card meta-row",
                            ),
                        ],
                        className="left-col",
                    ),
                    html.Div(
                        [
                            dcc.Graph(id="phase-graph", config={"displayModeBar": False}),
                            dcc.Graph(id="heartbeat-graph", config={"displayModeBar": False}),
                            dcc.Graph(id="ibi-graph", config={"displayModeBar": False}),
                        ],
                        className="right-col",
                    ),
                ],
                className="operator-grid",
            ),
        ],
        className="app-root",
    )


def subject_layout() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(id="pulse-ring", className="pulse-ring idle"),
                            html.Div(id="subject-hr", children="––", className="subject-hr"),
                        ],
                        className="pulse-frame",
                    ),
                    html.Div("BPM", className="subject-unit"),
                    html.Div(
                        [
                            html.Span(id="subject-quality-pill", className="quality fair"),
                            html.Span(id="subject-status-text", children="warming up", className="quality-text"),
                        ],
                        className="subject-status",
                    ),
                ],
                className="subject-stack",
            ),
            # Hidden tracker for animation triggering: stores the last beat index.
            dcc.Store(id="last-beat-index", data=0),
            html.A("→ operator view", href="/", className="link",
                   style={"position": "fixed", "top": "16px", "right": "20px"}),
        ],
        className="subject-root",
    )


# ── App factory ────────────────────────────────────────────────────────────


def make_app(state: DashboardState, *, tick_ms: int = 200) -> Dash:
    global STATE
    STATE = state

    app = Dash(__name__, external_stylesheets=[_GOOGLE_FONTS], title="mmWave HRV")
    app.layout = html.Div(
        [
            dcc.Location(id="url", refresh=False),
            dcc.Interval(id="tick", interval=tick_ms, n_intervals=0),
            html.Div(id="root"),
        ]
    )

    @app.callback(Output("root", "children"), Input("url", "search"))
    def render_view(search: str | None) -> html.Div:
        view = "subject" if (search and "view=subject" in search) else "operator"
        return subject_layout() if view == "subject" else operator_layout()

    # ── Operator-view tick ────────────────────────────────────────────────
    @app.callback(
        Output("hr-value", "children"),
        Output("sdnn-value", "children"),
        Output("rmssd-value", "children"),
        Output("pnn50-value", "children"),
        Output("quality-pill", "className"),
        Output("quality-text", "children"),
        Output("phase-graph", "figure"),
        Output("heartbeat-graph", "figure"),
        Output("ibi-graph", "figure"),
        Output("chest-range", "children"),
        Output("frame-count", "children"),
        Output("elapsed", "children"),
        Input("tick", "n_intervals"),
        prevent_initial_call=False,
    )
    def update_operator(_n: int) -> tuple[Any, ...]:
        snap = STATE.snapshot()
        result = snap.result
        quality, quality_text = _quality_class(result)
        if result is None:
            return (
                "––", "–", "–", "–",
                f"quality {quality}", quality_text,
                phase_figure(None, STATE.plot_window_s),
                heartbeat_figure(None, STATE.plot_window_s),
                ibi_figure(None),
                "––", str(snap.n_appended), _format_elapsed(snap.elapsed_s),
            )
        return (
            _format_metric(result.metrics["mean_hr_bpm"]),
            _format_metric(result.metrics["sdnn_ms"]),
            _format_metric(result.metrics["rmssd_ms"]),
            _format_metric(result.metrics["pnn50_pct"]),
            f"quality {quality}",
            quality_text,
            phase_figure(result, STATE.plot_window_s),
            heartbeat_figure(result, STATE.plot_window_s),
            ibi_figure(result),
            f"{result.chest_range_m:.2f} m",
            str(snap.n_appended),
            _format_elapsed(snap.elapsed_s),
        )

    # ── Subject-view tick ─────────────────────────────────────────────────
    @app.callback(
        Output("subject-hr", "children"),
        Output("pulse-ring", "className"),
        Output("subject-quality-pill", "className"),
        Output("subject-status-text", "children"),
        Output("last-beat-index", "data"),
        Input("tick", "n_intervals"),
        State("last-beat-index", "data"),
        prevent_initial_call=False,
    )
    def update_subject(_n: int, last_idx: int) -> tuple[Any, ...]:
        snap = STATE.snapshot()
        result = snap.result
        quality, quality_text = _quality_class(result)

        if result is None:
            return "––", "pulse-ring idle", f"quality {quality}", quality_text, 0

        beat_count = int(result.peak_times_s.size)
        # Fire the animation whenever a new beat has appeared since last tick.
        ring_class = "pulse-ring fire" if beat_count > (last_idx or 0) else "pulse-ring idle"
        return (
            _format_metric(result.metrics["mean_hr_bpm"], decimals=0),
            ring_class,
            f"quality {quality}",
            quality_text,
            beat_count,
        )

    return app
