from io import BytesIO

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.analytics.flow import FlowReport
from app.analytics.levels import find_levels
from app.analytics.smc import analyze_smc
from app.analytics.volume import volume_profile, volume_summary


def render_chart(candles: pd.DataFrame, symbol: str, timeframe: str, flow: FlowReport | None = None) -> BytesIO:
    support, resistance = find_levels(candles)
    data = candles.tail(140)
    profile = volume_profile(data)  # same window as the visible candles
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=data.timestamp, open=data.open, high=data.high, low=data.low, close=data.close, name="OHLC"), row=1, col=1)
    colors = ["#32d296" if c >= o else "#ef5b6c" for o, c in zip(data.open, data.close)]
    fig.add_trace(go.Bar(x=data.timestamp, y=data.volume, marker_color=colors, name="Volume"), row=2, col=1)
    for level in support:
        fig.add_hrect(y0=level.low, y1=max(level.high, level.low + candles.close.iloc[-1] * .0015), fillcolor="#32d296", opacity=.28, line_width=0, row=1, col=1)
    for level in resistance:
        fig.add_hrect(y0=level.low, y1=max(level.high, level.low + candles.close.iloc[-1] * .0015), fillcolor="#ef5b6c", opacity=.28, line_width=0, row=1, col=1)
    smc = analyze_smc(candles)
    first = data.timestamp.iloc[0]
    last_price = float(data.close.iloc[-1])
    near = lambda zs: sorted((z for z in zs if abs((z.low + z.high) / 2 - last_price) / last_price < 0.06),
                             key=lambda z: abs((z.low + z.high) / 2 - last_price))[:3]
    for z in [*near(smc.order_blocks), *near(smc.fvgs)]:
        if z.ts < first or not (data.low.min() <= z.high and z.low <= data.high.max()):
            continue
        color = "#2dd4bf" if z.direction == "bull" else "#fb923c"
        fig.add_shape(type="rect", xref="x", yref="y", x0=z.ts, x1=data.timestamp.iloc[-1], y0=z.low, y1=z.high,
                      line={"color": color, "width": 1, "dash": "dot" if z.kind == "FVG" else "solid"},
                      fillcolor=color, opacity=.2 if z.kind == "OB" else .12)
        fig.add_annotation(x=z.ts, y=z.high, text=f"{z.kind}{'+' if z.direction == 'bull' else '-'}", showarrow=False,
                           font={"size": 9, "color": color}, xanchor="left", yanchor="bottom")
    fig.add_hrect(y0=profile["val"], y1=profile["vah"], fillcolor="#f5c451", opacity=.06, line_width=0, row=1, col=1,
                  annotation_text="Value Area*", annotation_position="top left")
    if flow and flow.all_large:
        start = data.timestamp.iloc[0]
        for side, color, symbol_shape, label in (("buy", "#32d296", "triangle-up", "Крупная покупка (агрессор)"),
                                                 ("sell", "#ef5b6c", "triangle-down", "Крупная продажа (агрессор)")):
            picks = [w for w in flow.all_large if w.side == side and pd.Timestamp(w.ts_ms, unit="ms", tz="UTC") >= start]
            if picks:
                fig.add_trace(go.Scatter(x=[pd.Timestamp(w.ts_ms, unit="ms", tz="UTC") for w in picks], y=[w.price for w in picks],
                                         mode="markers", name=label, text=[f"${w.quote:,.0f}" for w in picks],
                                         marker={"symbol": symbol_shape, "color": color, "size": 9,
                                                 "line": {"color": "white", "width": 1}}), row=1, col=1)
    fig.add_hline(y=profile["poc"], line_dash="dot", line_color="#f5c451", annotation_text="POC*", row=1, col=1)
    vwap = volume_summary(candles, float(candles.close.iloc[-1]))["vwap_24h"]
    fig.add_hline(y=vwap, line_dash="dash", line_color="#54a9ff", annotation_text="24h VWAP", row=1, col=1)
    fig.update_layout(template="plotly_dark", title=f"{symbol} · {timeframe}", height=760, xaxis_rangeslider_visible=False,
                      margin={"l": 35, "r": 25, "t": 90, "b": 35}, legend={"orientation": "h", "y": 1.07, "x": 0}, paper_bgcolor="#111827")
    pad = (data.high.max() - data.low.min()) * 0.04
    fig.update_yaxes(title_text="Price", range=[data.low.min() - pad, data.high.max() + pad], row=1, col=1)
    fig.update_yaxes(title_text="Base volume", row=2, col=1)
    image = BytesIO(fig.to_image(format="png", width=1000, height=760, scale=1))
    image.name = f"{symbol.replace('/', '_')}_{timeframe}.png"
    image.seek(0)
    return image
