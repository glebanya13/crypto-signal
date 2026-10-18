from datetime import UTC, datetime

import pandas as pd

from app.analytics.flow import FlowReport
from app.analytics.levels import (
    PivotPoints,
    PriceLevel,
    detect_breakouts,
    find_levels,
    profile_nodes,
)
from app.analytics.signal import Signal, compute_signal
from app.analytics.smc import SmcAnalysis, Zone, analyze_smc
from app.analytics.technical import analyze_technical, trend_label
from app.analytics.volume import volume_profile, volume_summary


def _price(value: float) -> str:
    return f"${value:,.4f}" if value < 1 else f"${value:,.2f}"


def _band(level: PriceLevel) -> str:
    mid = (level.low + level.high) / 2
    return f"{_price(min(level.low, mid * 0.999))}-{_price(max(level.high, mid * 1.001))}"


def _level_line(level: PriceLevel) -> str:
    mid = (level.low + level.high) / 2
    low, high = min(level.low, mid * 0.999), max(level.high, mid * 1.001)  # single-touch zones get a ±0.1% band
    grade = "сильная" if level.strength >= 65 else "средняя"
    age = ""
    if level.age_hours is not None:
        age = f", последний тест {level.age_hours / 24:.1f} дн назад" if level.age_hours >= 48 else f", последний тест {level.age_hours:.0f} ч назад"
    vol = f", объем у экстремумов {usd(level.volume * mid)}" if level.volume else ""
    return (f"• {_price(low)}-{_price(high)} - {grade}, касаний: {level.touches}, расстояние: {level.distance_pct:.2f}%"
            f"{age}{vol}")


def usd(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if x >= limit:
            return f"{sign}${x / limit:.2f}{suffix}"
    return f"{sign}${x:,.0f}"


def _hhmm(ms: int) -> str:
    return f"{datetime.fromtimestamp(ms / 1000, UTC):%H:%M:%S}"


def _side(side: str) -> str:
    return "🟢 Aggressive Buy" if side == "buy" else "🔴 Aggressive Sell"


def levels_text(support: list[PriceLevel], resistance: list[PriceLevel]) -> list[str]:
    lines = ["<b>Поддержка:</b>"]
    lines += [_level_line(x) for x in support] or ["• Нет подтвержденных уровней ниже цены"]
    lines += ["", "<b>Сопротивление:</b>"]
    lines += [_level_line(x) for x in resistance] or ["• Нет подтвержденных уровней выше цены"]
    return lines


def build_levels_report(symbol: str, candles: pd.DataFrame, ticker: dict, fetched_at: datetime, timeframe: str,
                        pivots: PivotPoints | None) -> str:
    price = float(ticker.get("last") or candles.close.iloc[-1])
    support, resistance = find_levels(candles)
    lines = [f"<b>{symbol} - уровни ({timeframe})</b>", f"Цена: <b>{_price(price)}</b>", ""] + levels_text(support, resistance)
    hvn, lvn = profile_nodes(candles)
    near = lambda xs: sorted(xs, key=lambda x: abs(x - price))[:3]
    lines += ["", "<b>Узлы объема</b> (приближенно, по свечам):",
              "HVN (много торговли, цену там часто «держат»): " + (", ".join(_price(x) for x in near(hvn)) or "н/д"),
              "LVN (мало торговли, цена проходит быстро): " + (", ".join(_price(x) for x in near(lvn)) or "н/д")]
    if pivots:
        lines += ["", "<b>Pivot Points</b> (по вчерашнему дню):",
                  (f"R2 {_price(pivots.r2)} | R1 {_price(pivots.r1)} | <b>P {_price(pivots.pivot)}</b> | "
                   f"S1 {_price(pivots.s1)} | S2 {_price(pivots.s2)}")]
    breaks = detect_breakouts(candles)
    lines += ["", "<b>Недавние пробои и ретесты:</b>"]
    for b in breaks:
        what = "пробой вверх сопротивления" if b.direction == "up" else "пробой вниз поддержки"
        vol = f", объем {b.volume_ratio:.1f}x от обычного" if pd.notna(b.volume_ratio) else ""
        lines.append(f"• {what} {_band(b.zone)}, {b.bars_ago} св. назад{vol}: {b.status}")
    if not breaks:
        lines.append("• за последние 40 свечей зон не пробивали")
    lines += ["", f"Источник: Binance Spot | получено {fetched_at:%Y-%m-%d %H:%M:%S} UTC",
              "Сила уровня - эвристика (касания, объем, давность), не вероятность отбоя."]
    return "\n".join(lines)


def tape_volume_lines(flow: FlowReport | None) -> list[str]:
    if flow is None:
        return ["Поток сделок (buy/sell, CVD): н/д - не удалось получить сделки Binance."]
    lines = []
    for minutes, label in ((5, "5 мин"), (60, "1 час")):
        w = flow.windows[minutes]
        if not w.covered:
            lines.append(f"Окно {label}: н/д (загружено только {flow.span_minutes:.0f} мин ленты)")
            continue
        share = w.buy_quote / w.quote * 100 if w.quote else float("nan")
        lines.append(f"<b>{label}:</b> {usd(w.quote)} · покупки {usd(w.buy_quote)} ({share:.0f}%) · "
                     f"продажи {usd(w.sell_quote)} · Delta <b>{usd(w.delta)}</b>")
    lines.append(f"RVOL по ленте (5 мин к среднему 5-мин окну за {flow.span_minutes:.0f} мин): "
                 + (f"{flow.rvol_5m:.2f}x" if pd.notna(flow.rvol_5m) else "н/д"))
    if flow.spike:
        sp = flow.spike
        lines.append(f"⚡ Аномальный всплеск: минута {_hhmm(sp['minute_ms'])} UTC - {usd(sp['quote'])}, "
                     f"{sp['x_median']:.1f}x от медианной минуты (z={sp['z']:.1f})")
    else:
        lines.append("Аномальных всплесков за последние минуты не обнаружено.")
    return lines


def whales_lines(flow: FlowReport | None, count: int = 10) -> list[str]:
    if flow is None:
        return ["Крупные сделки: н/д - не удалось получить сделки Binance."]
    lines = [f"Порог крупной сделки (динамический, p99.9 и ≥30× медианы): <b>{usd(flow.threshold)}</b>"]
    for w in flow.whales[:count]:
        lines.append(f"• {_hhmm(w.ts_ms)} - {_side(w.side)} <b>{usd(w.quote)}</b> по {_price(w.price)} "
                     f"({w.x_median:.0f}× медианной сделки, {w.pct_of_minute:.0f}% средней минуты)")
    if not flow.whales:
        lines.append("• За окно сделок выше порога нет")
    if flow.clusters:
        lines.append("")
        lines.append("<b>Серии крупных исполнений</b> (≥5 сделок p99 одной стороны подряд, пауза до 20с):")
        for c in sorted(flow.clusters, key=lambda c: c.quote, reverse=True)[:3]:
            lines.append(f"• {_hhmm(c.start_ms)}-{_hhmm(c.end_ms)} - {_side(c.side)} ×{c.count}, всего {usd(c.quote)}")
    return lines


def strength(score: float) -> str:
    a = abs(score)
    return "нет перевеса" if a < 25 else "слабый" if a < 50 else "средний" if a < 75 else "сильный"


def signal_lines(sig: Signal) -> list[str]:
    title = {"long": "🟢 <b>LONG</b> (перевес покупателей)", "short": "🔴 <b>SHORT</b> (перевес продавцов)",
             "neutral": "⚪ <b>Нет явного перевеса</b> - лучше наблюдать"}[sig.direction]
    lines = [title]
    if sig.direction != "neutral":
        lines.append(f"Сила сигнала: <b>{strength(sig.score)}</b> ({abs(sig.score):.0f} из 100). "
                     "Чем ближе к 100, тем больше признаков из списка ниже указывают в одну сторону "
                     "(100 значит, что все за). До 25 перевеса нет, 25-50 слабый, 50-75 средний, выше 75 сильный.")
    else:
        lines.append(f"Факторы расходятся (итог {sig.score:+.0f}, нужно хотя бы ±25), поэтому однозначного направления нет.")
    for f in sig.factors:
        mark = "▲" if f.value > 0.15 else "▼" if f.value < -0.15 else "•"
        lines.append(f"{mark} {f.name}: {f.note}")
    if sig.entry is None and sig.direction != "neutral":
        lines.append("Сценарий: направление есть, но рядом нет подходящей зоны входа (поддержки, сопротивления, Order Block или FVG) - "
                     "разумнее подождать, пока цена подойдет к такой зоне.")
    elif sig.entry is not None:
        side = "LONG" if sig.direction == "long" else "SHORT"
        low, high = sig.entry_zone
        if sig.entry_now:
            where = f"вход сейчас около {_price(sig.entry)} (цена в зоне {sig.entry_source} {_price(low)}-{_price(high)})"
        else:
            where = (f"ждать отката в зону {_price(low)}-{_price(high)} ({sig.entry_source}), "
                     f"лимитная заявка у {_price(sig.entry)}")
        lines.append(f"<b>Сценарий {side}:</b> {where}. Стоп (отмена идеи) {_price(sig.stop)}. "
                     "Цели: " + " → ".join(_price(t) for t in sig.targets)
                     + (f". Риск/доходность 1 к {sig.risk_reward:.1f}" if sig.risk_reward else ". Риск/доходность н/д"))
        if not sig.actionable:
            lines.append("⚠️ Сделка не проходит фильтр риск/доходность не хуже 1 к 1.5: на истории такие входы в среднем "
                         "убыточны. Направление есть, но входить здесь не стоит, лучше дождаться более выгодной точки.")
    lines.append("Это не вероятность роста или падения, а сумма признаков. Стоп и цели взяты из найденных зон, это не рекомендация.")
    return lines


def build_signal_report(symbol: str, candles: pd.DataFrame, ticker: dict, fetched_at: datetime,
                        timeframe: str, flow: FlowReport | None) -> str:
    price = float(ticker.get("last") or candles.close.iloc[-1])
    support, resistance = find_levels(candles)
    sig = compute_signal(candles, price, support, resistance, flow, volume_summary(candles, price, ticker)["vwap_24h"],
                         analyze_smc(candles), analyze_technical(candles))
    return "\n".join([f"<b>{symbol} - направление ({timeframe})</b>", f"Цена: <b>{_price(price)}</b>", ""] + signal_lines(sig)
                      + ["", f"Источник: Binance Spot | получено {fetched_at:%Y-%m-%d %H:%M:%S} UTC"])


def _zone_line(z: Zone, price: float) -> str:
    side = "бычья" if z.direction == "bull" else "медвежья"
    dist = (z.low - price) / price * 100 if z.low > price else (z.high - price) / price * 100 if z.high < price else 0.0
    where = "цена внутри" if dist == 0 else f"{abs(dist):.2f}% {'выше' if dist > 0 else 'ниже'}"
    return (f"• {side} {_price(z.low)}-{_price(z.high)} - {where}"
            + (", уже тестировалась" if z.tested else ", не тестировалась") + f" ({z.ts:%d.%m %H:%M})")


def build_smc_report(symbol: str, candles: pd.DataFrame, ticker: dict, fetched_at: datetime, timeframe: str) -> str:
    price = float(ticker.get("last") or candles.close.iloc[-1])
    smc: SmcAnalysis = analyze_smc(candles)
    trend = {"bull": "🟢 бычья", "bear": "🔴 медвежья", "range": "⚪ не определена"}[smc.trend]
    lines = [f"<b>{symbol} - Smart Money ({timeframe})</b>", f"Цена: <b>{_price(price)}</b>", "",
             f"<b>Структура рынка:</b> {trend} (HH/HL - бычья, LH/LL - медвежья)"]
    for e in smc.events[-3:][::-1]:
        meaning = "продолжение тренда" if e.kind == "BOS" else "смена характера, возможный разворот в эту сторону"
        lines.append(f"• {e.kind} {'вверх' if e.direction == 'bull' else 'вниз'} через {_price(e.level)} - {meaning}, {e.ts:%d.%m %H:%M}")
    if smc.position_pct is not None:
        label = {"premium": "premium - верхняя (дорогая) часть диапазона", "discount": "discount - нижняя (дешевая) часть диапазона",
                 "equilibrium": "equilibrium - около середины"}[smc.zone_label]
        lines.append(f"Диапазон {_price(smc.range_low)}-{_price(smc.range_high)}: цена на {smc.position_pct:.0f}% - {label}")
    lines += ["", "<b>Order Blocks</b> (непробитые):"]
    above = [z for z in smc.order_blocks if z.low > price][:2]
    below = [z for z in smc.order_blocks if z.high < price][:2]
    inside = [z for z in smc.order_blocks if z.low <= price <= z.high][:2]
    lines += [_zone_line(z, price) for z in inside + above + below] or ["• нет"]
    lines += ["", "<b>Fair Value Gaps</b> (не закрытые полностью):"]
    fvg = sorted(smc.fvgs, key=lambda z: abs((z.low + z.high) / 2 - price))[:4]
    lines += [_zone_line(z, price) for z in fvg] or ["• нет"]
    lines += ["", "<b>Ликвидность</b> (equal highs/lows - скопление стопов):"]
    for pool in smc.pools:
        what = "equal highs (стопы шортистов выше)" if pool.side == "buy" else "equal lows (стопы лонгистов ниже)"
        lines.append(f"• {_price(pool.price)} - {what}, ×{pool.touches}, {abs(pool.price - price) / price * 100:.2f}% от цены")
    if not smc.pools:
        lines.append("• явных пулов нет")
    if smc.sweep:
        sw = smc.sweep
        taken = "верхней ликвидности (highs)" if sw.side == "buy" else "нижней ликвидности (lows)"
        lines.append(f"⚡ Недавний снос {taken} на {_price(sw.level)} ({sw.ts:%d.%m %H:%M}) с возвратом - "
                     + ("часто читают как ловушку покупателей" if sw.side == "buy" else "часто читают как ловушку продавцов"))
    lines += ["", ("<b>Как читать:</b> BOS/CHoCH, OB, FVG и пулы ликвидности - механические правила по свечам, они описывают "
               "структуру цены, а не действия конкретных участников. Это не прогноз."),
              f"Источник: Binance Spot | свечи {timeframe} | получено {fetched_at:%Y-%m-%d %H:%M:%S} UTC"]
    return "\n".join(lines)


def build_ta_report(symbol: str, candles: pd.DataFrame, ticker: dict, fetched_at: datetime, timeframe: str,
                    mtf: dict[str, pd.DataFrame]) -> str:
    price = float(ticker.get("last") or candles.close.iloc[-1])
    ta = analyze_technical(candles)
    lines = [f"<b>{symbol} - Технический анализ ({timeframe})</b>", f"Цена: <b>{_price(price)}</b>", ""]
    for ind in ta.indicators:
        mark = "▲" if ind.score > 0.15 else "▼" if ind.score < -0.15 else "•"
        lines.append(f"{mark} <b>{ind.name}:</b> {ind.text}")
    bias = {"bull": "🟢 бычий перевес", "bear": "🔴 медвежий перевес", "neutral": "⚪ нейтрально"}[ta.bias]
    lines += ["", f"Сводка индикаторов: {bias}, сила: {strength(ta.score)} ({abs(ta.score):.0f} из 100)"]
    lines += ["", "<b>Тренд по таймфреймам</b> (EMA20/50, RSI):"]
    for tf, frame in mtf.items():
        label, rv = trend_label(frame)
        icon = "🟢" if label == "восходящий" else "🔴" if label == "нисходящий" else "⚪"
        lines.append(f"{icon} {tf}: {label}, RSI {rv:.0f}")
    lines += [""] + [f"ℹ️ {n}" for n in ta.notes]
    lines += ["Индикаторы запаздывают и описывают прошлое; сводка - не прогноз и не рекомендация.",
              f"Источник: Binance Spot | получено {fetched_at:%Y-%m-%d %H:%M:%S} UTC"]
    return "\n".join(lines)


def build_verdict(price: float, support: list[PriceLevel], resistance: list[PriceLevel],
                  flow: FlowReport | None, rvol: float) -> str:
    """Plain-language, non-predictive summary built only from the numbers above."""
    parts = []
    near = [(x, "поддержки") for x in support[:1]] + [(x, "сопротивления") for x in resistance[:1]]
    near = [(x, kind) for x, kind in near if x.distance_pct <= 1.0]
    if near:
        x, kind = min(near, key=lambda item: item[0].distance_pct)
        parts.append(f"Цена находится около {kind} {_price(x.low)}-{_price(x.high)} (до {x.distance_pct:.2f}%).")
    else:
        parts.append("Цена не вплотную к ближайшим уровням, ближайшие зоны указаны выше.")
    if pd.notna(rvol):
        parts.append("Объем свечи " + ("выше" if rvol > 1.3 else "ниже" if rvol < 0.7 else "около") + f" среднего (RVOL {rvol:.1f}x).")
    w = flow.windows[60] if flow else None
    if w and w.covered and w.quote:
        share = w.buy_quote / w.quote
        if share > 0.55:
            parts.append(f"За час преобладают агрессивные покупки ({share:.0%}), Delta {usd(w.delta)} - покупатели активнее, но это не гарантирует рост.")
        elif share < 0.45:
            parts.append(f"За час преобладают агрессивные продажи ({1 - share:.0%}), Delta {usd(w.delta)} - продавцы активнее, но это не гарантирует падение.")
        else:
            parts.append("Агрессивные покупки и продажи за час примерно сбалансированы.")
    if flow and flow.spike:
        parts.append("Есть аномальный всплеск оборота - значимые движения часто сопровождаются повышенной активностью.")
    if near and near[0][1] == "сопротивления" or (resistance and resistance[0].distance_pct <= 1.0):
        parts.append("Закрепление выше сопротивления с объемом усилит бычий сценарий; отбой при падении Delta - признак слабости покупателей.")
    elif support and support[0].distance_pct <= 1.0:
        parts.append("Удержание поддержки с ростом покупок - позитивный признак; пробой вниз с ростом агрессивных продаж усилит медвежий сценарий.")
    parts.append("Это описание рыночных данных, а не прогноз и не финансовая рекомендация.")
    return " ".join(parts)


def build_volume_report(symbol: str, candles: pd.DataFrame, ticker: dict, fetched_at: datetime,
                        timeframe: str, flow: FlowReport | None) -> str:
    price = float(ticker.get("last") or candles.close.iloc[-1])
    m = volume_summary(candles, price, ticker)
    profile = volume_profile(candles)
    fmt = lambda x: usd(x) if pd.notna(x) else "н/д"
    lines = [f"<b>{symbol} - Анализ объемов</b>", "", "<b>По ленте сделок (агрессор):</b>"] + tape_volume_lines(flow)
    lines += ["", f"<b>По свечам {timeframe}:</b>",
              f"Последняя свеча: {m['volume_base_last']:,.4g} {symbol.split('/')[0]} (~{fmt(m['volume_quote_last'])}), "
              + (f"к предыдущей {m['volume_change_pct']:+.0f}%" if pd.notna(m["volume_change_pct"]) else "изменение н/д"),
              f"RVOL свечи: {m['rvol']:.2f}x" if pd.notna(m["rvol"]) else "RVOL свечи: н/д",
              f"Объем 1ч: {fmt(m['volume_1h_quote'])} | 24ч (Binance): {fmt(m['volume_24h_quote'])}",
              f"VWAP 24ч: {_price(m['vwap_24h'])} | POC*: {_price(profile['poc'])}, VAL*: {_price(profile['val'])}, VAH*: {_price(profile['vah'])}",
              "", "Buy/Sell показывает сторону агрессора, а не приток или отток денег.",
              f"Источник: Binance Spot | получено {fetched_at:%Y-%m-%d %H:%M:%S} UTC",
              "*Профиль приближенный: объем свечи отнесен к typical price."]
    return "\n".join(lines)


def build_whales_report(symbol: str, flow: FlowReport | None, fetched_at: datetime) -> str:
    span = f"за последние {flow.span_minutes:.0f} мин" if flow else ""
    lines = [f"<b>{symbol} - крупные сделки {span}</b>", ""] + whales_lines(flow)
    lines += ["", "Сторона - агрессор исполнения. Личность участника неизвестна: это не обязательно «кит».",
              f"Источник: Binance Spot aggTrades | получено {fetched_at:%H:%M:%S} UTC"]
    return "\n".join(lines)


def build_report(symbol: str, candles: pd.DataFrame, ticker: dict, fetched_at: datetime, timeframe: str,
                 flow: FlowReport | None = None) -> str:
    price = float(ticker.get("last") or candles.close.iloc[-1])
    change = ticker.get("percentage")
    support, resistance = find_levels(candles)
    metrics = volume_summary(candles, price, ticker)
    profile = volume_profile(candles)
    trend = "восходящий" if candles.close.iloc[-1] > candles.close.iloc[-20] else "нисходящий" if candles.close.iloc[-1] < candles.close.iloc[-20] else "боковой"
    lines = [f"<b>{symbol} - Market Analysis</b>", f"Цена: <b>{_price(price)}</b> | 24ч: {f'{change:+.2f}%' if change is not None else 'н/д'}",
             f"Тренд {timeframe}: {trend} (сравнение с 20 свечами)", ""]
    lines += levels_text(support, resistance)
    sig = compute_signal(candles, price, support, resistance, flow, metrics["vwap_24h"],
                         analyze_smc(candles), analyze_technical(candles))
    lines += ["", "<b>Направление:</b>"] + signal_lines(sig)
    lines += ["", "<b>Объемы и поток:</b>"] + tape_volume_lines(flow)
    lines += [f"VWAP 24ч: {_price(metrics['vwap_24h'])} | POC*: {_price(profile['poc'])}, VAL*: {_price(profile['val'])}, VAH*: {_price(profile['vah'])}"]
    lines += ["", "<b>Крупнейшие сделки:</b>"] + whales_lines(flow, 3)
    lines += ["", "Стакан - кнопка «📚 Стакан»; фьючерсы, OI, funding и ликвидации пока не реализованы.", "",
              "<b>Вывод:</b> " + build_verdict(price, support, resistance, flow, metrics["rvol"]),
              "", f"Источник: Binance Spot | свечи {timeframe} + лента сделок | получено {fetched_at:%Y-%m-%d %H:%M:%S} UTC",
              "*Профиль приближенный: объем каждой свечи отнесен к typical price."]
    return "\n".join(lines)


def build_orderbook_report(symbol: str, book, depth_limit: int) -> str:
    """Large resting limit orders from the synchronized Binance Spot book. Orders are liquidity, not executed trades."""
    bids = sorted(book.bids, key=lambda x: x[0] * x[1], reverse=True)[:5]
    asks = sorted(book.asks, key=lambda x: x[0] * x[1], reverse=True)[:5]
    bid_total = sum(p * q for p, q in book.bids)
    ask_total = sum(p * q for p, q in book.asks)
    snapshot = datetime.fromtimestamp(book.snapshot_time_ms / 1000, UTC)
    tilt = ("перевес заявок на покупку" if book.imbalance > 0.1 else "перевес заявок на продажу" if book.imbalance < -0.1
            else "заявки примерно сбалансированы")
    lines = [f"<b>{symbol} - стакан Binance Spot</b>", f"Снимок {snapshot:%H:%M:%S} UTC, глубина {depth_limit} уровней с каждой стороны",
             f"Заявки на покупку: {usd(bid_total)} | на продажу: {usd(ask_total)}",
             f"Imbalance: {book.imbalance:+.1%} - {tilt}", "", "<b>Крупнейшие лимитные покупки (bid):</b>"]
    lines += [f"• {_price(p)} - {usd(p * q)}" for p, q in bids] or ["• н/д"]
    lines += ["", "<b>Крупнейшие лимитные продажи (ask):</b>"]
    lines += [f"• {_price(p)} - {usd(p * q)}" for p, q in asks] or ["• н/д"]
    lines += ["", ("Это лимитные заявки, а не исполненные сделки: их можно снять в любой момент. "
                   "Одного снимка недостаточно, чтобы судить о манипуляциях.")]
    return "\n".join(lines)


def build_backtest_report(result, fetched_at: datetime) -> str:
    st = result.stats()
    days = result.bars * {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800}[result.timeframe] / 86400
    lines = [f"<b>{result.symbol} - бэктест Long/Short ({result.timeframe})</b>",
             f"История: {result.bars} свечей (~{days:.0f} дн). Правила те же, что в /signal, без подсмотра в будущее.", ""]
    if st["n"] == 0:
        lines.append("За период не набралось ни одной сделки, выводов сделать нельзя.")
    else:
        longs = result.stats([t for t in result.trades if t.direction == "long"])
        shorts = result.stats([t for t in result.trades if t.direction == "short"])
        pf = "без убыточных" if st["profit_factor"] == float("inf") else f"{st['profit_factor']:.2f}"
        lines += [f"Сделок: <b>{st['n']}</b> (long {longs['n']}, short {shorts['n']}). Пропущено по фильтру риск/доходность: {result.low_rr}",
                  f"Прибыльных: <b>{st['win_rate']:.0%}</b>",
                  f"Средний результат на сделку: <b>{st['avg_r']:+.2f}R</b> (1R = размер стопа)",
                  f"Итого: {st['total_r']:+.1f}R, максимальная просадка {st['max_dd']:.1f}R, profit factor {pf}",
                  f"Long: {longs['avg_r']:+.2f}R на сделку | Short: {shorts['avg_r']:+.2f}R на сделку" if longs["n"] and shorts["n"] else ""]
        if st["n"] < 30:
            verdict = "Сделок мало, выводы статистически слабые."
        elif st["avg_r"] > 0.1:
            verdict = "На этой истории правила были прибыльны. Это не гарантия на будущее: рынок меняется."
        elif st["avg_r"] >= -0.1:
            verdict = "Результат около нуля: заметного преимущества у сигналов на этой истории нет."
        else:
            verdict = "На этой истории правила были убыточны, ориентироваться на сигнал как на готовую сделку нельзя."
        lines += ["", f"<b>Вывод:</b> {verdict}"]
    lines += ["", ("Условия: лимитный вход действует 12 свечей, позиция держится до 48 свечей, при касании стопа и цели в одной свече "
                   "засчитывается стоп, комиссия и проскальзывание 0.05R на сделку, одна позиция за раз. Поток сделок (Delta) "
                   "в бэктесте не используется, исторической ленты нет."),
              f"Источник: Binance Spot | {fetched_at:%Y-%m-%d %H:%M:%S} UTC"]
    return "\n".join(x for x in lines if x is not None)
