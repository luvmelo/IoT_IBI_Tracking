"""Plotly figure factories for the dashboard.

Dark instrumentation aesthetic: ground = #08090C, traces in scope-amber,
secondary in cyan, alerts in coral. Layouts disable mode bar, set
margins tight, and use monospaced numerics (DM Mono via Google Fonts).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from radar_analysis.beat_detection import clean_ibi
from radar_analysis.pipeline import PipelineResult


def _stable_yrange(y: np.ndarray, *, padding: float = 0.15, snap: float = 0.5) -> list[float]:
    """Return a y-range quantized to multiples of `snap` so the axis doesn't twitch
    on every tick. Uses 5/95 percentiles to ignore single-sample spikes.
    """
    if y.size == 0:
        return [-1.0, 1.0]
    lo = float(np.quantile(y, 0.05))
    hi = float(np.quantile(y, 0.95))
    span = max(hi - lo, 0.5)
    pad = padding * span
    lo, hi = lo - pad, hi + pad
    # Quantize to multiples of `snap`.
    lo = np.floor(lo / snap) * snap
    hi = np.ceil(hi / snap) * snap
    return [float(lo), float(hi)]


COLORS = {
    "void": "#08090C",
    "surface": "#111318",
    "border": "#1E2130",
    "amber": "#E8A838",
    "cyan": "#4DD9C0",
    "coral": "#F05C5C",
    "white": "#F0F2F6",
    "steel": "#7A8499",
}


_LAYOUT_BASE = dict(
    paper_bgcolor=COLORS["surface"],
    plot_bgcolor=COLORS["surface"],
    font=dict(family='"DM Mono", ui-monospace, Menlo, monospace',
              size=11, color=COLORS["steel"]),
    margin=dict(l=46, r=14, t=20, b=32),
    hovermode=False,
    showlegend=False,
    xaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"],
               linecolor=COLORS["border"], color=COLORS["steel"]),
    yaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"],
               linecolor=COLORS["border"], color=COLORS["steel"]),
)


def _empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, x=0.0, xanchor="left",
                   font=dict(size=11, color=COLORS["steel"])),
        height=240,
    )
    return fig


def phase_figure(result: PipelineResult | None, plot_window_s: float) -> go.Figure:
    fig = _empty_figure("phase (rad) — last seconds")
    if result is None or result.phi_clean.size == 0:
        return fig
    n = result.phi_clean.size
    t_full = np.arange(n) / result.fs_slow_hz
    if plot_window_s > 0:
        keep = t_full >= max(0.0, t_full[-1] - plot_window_s)
    else:
        keep = np.ones(n, dtype=bool)
    t = t_full[keep]
    y = result.phi_clean[keep]

    fig.add_trace(go.Scatter(
        x=t, y=y, mode="lines",
        line=dict(color=COLORS["amber"], width=1.4),
    ))

    # Dim the heartbeat-band trace behind the phase
    if result.heartbeat.size == result.phi_clean.size:
        hb = result.heartbeat[keep]
        fig.add_trace(go.Scatter(
            x=t, y=hb, mode="lines",
            line=dict(color=COLORS["cyan"], width=1.0, dash="dot"),
            opacity=0.55,
        ))

    fig.update_xaxes(title_text="time (s)")
    fig.update_yaxes(title_text="phase (rad)", range=_stable_yrange(y, snap=0.25))
    return fig


def heartbeat_figure(result: PipelineResult | None, plot_window_s: float) -> go.Figure:
    fig = _empty_figure("heartbeat band + detected beats")
    if result is None or result.heartbeat.size == 0:
        return fig
    n = result.heartbeat.size
    t_full = np.arange(n) / result.fs_slow_hz
    if plot_window_s > 0:
        keep = t_full >= max(0.0, t_full[-1] - plot_window_s)
    else:
        keep = np.ones(n, dtype=bool)
    t = t_full[keep]
    y = result.heartbeat[keep]
    fig.add_trace(go.Scatter(
        x=t, y=y, mode="lines",
        line=dict(color=COLORS["amber"], width=1.4),
    ))

    if result.peak_times_s.size and t.size:
        in_window = result.peak_times_s >= t[0]
        peaks_t = result.peak_times_s[in_window]
        peaks_y = np.interp(peaks_t, t_full, result.heartbeat)
        fig.add_trace(go.Scatter(
            x=peaks_t, y=peaks_y, mode="markers",
            marker=dict(color=COLORS["cyan"], size=8, symbol="circle-open",
                        line=dict(width=1.5)),
        ))

    fig.update_xaxes(title_text="time (s)")
    fig.update_yaxes(title_text="filtered phase (rad)", range=_stable_yrange(y, snap=0.25))
    return fig


def ibi_figure(result: PipelineResult | None) -> go.Figure:
    fig = _empty_figure("IBI tachogram (ms)")
    if result is None or result.ibi_ms.size == 0:
        return fig
    keep = clean_ibi(result.ibi_ms)
    idx = np.arange(result.ibi_ms.size)

    if (~keep).any():
        fig.add_trace(go.Scatter(
            x=idx[~keep], y=result.ibi_ms[~keep], mode="markers",
            marker=dict(color=COLORS["coral"], size=8, symbol="x", line=dict(width=1.5)),
        ))
    fig.add_trace(go.Scatter(
        x=idx[keep], y=result.ibi_ms[keep], mode="lines+markers",
        line=dict(color=COLORS["cyan"], width=1.2),
        marker=dict(color=COLORS["cyan"], size=7),
    ))

    fig.update_xaxes(title_text="beat index")
    fig.update_yaxes(title_text="IBI (ms)", range=[400, 1400])
    return fig
