import re
from datetime import UTC, datetime, timedelta
from io import BytesIO

import numpy as np
import pandas as pd
import pytest
from aiogram import Bot, Dispatcher
from aiogram.types import Message, Update

from app.analytics import report
from app.analytics.flow import analyze_flow
from app.analytics.levels import pivot_points
from app.backtest import run_backtest
from app.telegram import handlers
from app.telegram.handlers import menu, split_text
from app.telegram.views import View, whats_new

ALLOWED_TAG = re.compile(r"</?(b|code)>")


def assert_valid_telegram_html(text: str) -> None:
    stripped = ALLOWED_TAG.sub("", text)
    assert "<" not in stripped and ">" not in stripped.replace("→", ""), "stray angle bracket breaks Telegram HTML"
    assert len(text) < 4096


def candles(n=400, seed=7):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.8, n))
    return pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC"),
                         "open": np.r_[close[0], close[:-1]], "high": close + 1, "low": close - 1,
                         "close": close, "volume": rng.uniform(5, 15, n)})


def tape(n=4000, seconds=4000):
    rng = np.random.default_rng(1)
    ts = np.sort(rng.integers(0, seconds * 1000, n)) + 1_700_000_000_000
    return [{"a": i, "T": int(t), "p": "100", "q": f"{q:.5f}", "m": bool(rng.integers(0, 2))}
            for i, (t, q) in enumerate(zip(ts, rng.lognormal(-3, 0.8, n), strict=True))]


@pytest.mark.parametrize("with_flow", [True, False])
def test_every_report_is_valid_telegram_html(with_flow):
    c = candles()
    ticker = {"last": float(c.close.iloc[-1]), "percentage": 1.2, "quoteVolume": 1e7}
    flow = analyze_flow(tape()) if with_flow else None
    now = datetime.now(UTC)
    mtf = {tf: c for tf in ("15m", "1h", "4h", "1d")}
    texts = [
        report.build_report("X/USDT", c, ticker, now, "1h", flow),
        report.build_signal_report("X/USDT", c, ticker, now, "1h", flow),
        report.build_smc_report("X/USDT", c, ticker, now, "1h"),
        report.build_ta_report("X/USDT", c, ticker, now, "1h", mtf),
        report.build_volume_report("X/USDT", c, ticker, now, "1h", flow),
        report.build_whales_report("X/USDT", flow, now),
        report.build_levels_report("X/USDT", c, ticker, now, "1h", pivot_points(c.tail(24))),
        report.build_backtest_report(run_backtest(c, "X/USDT", "1h"), now),
    ]
    for text in texts:
        assert_valid_telegram_html(text)


def test_split_text_keeps_lines_and_limit():
    text = "\n".join(f"line {i} " + "x" * 50 for i in range(300))
    parts = split_text(text, limit=1000)
    assert all(len(p) <= 1000 for p in parts) and "\n".join(parts) == text


def snap(**kw):
    base = {"time": datetime.now(UTC), "price": 100.0, "direction": "short", "score": -40.0, "actionable": False,
            "smc_trend": "bear", "last_event": ("BOS", "bear", "t1"), "delta_1h": -5.0, "top_whale_ms": 1, "spike": False}
    return {**base, **kw}


def test_whats_new_reports_changes_and_quiet_state():
    prev = snap(time=datetime.now(UTC) - timedelta(minutes=10))
    quiet = whats_new(prev, snap())
    assert "Существенных изменений нет" in quiet
    loud = whats_new(prev, snap(direction="long", smc_trend="bull", delta_1h=9.0, spike=True,
                                last_event=("CHoCH", "bull", "t2"), actionable=True, price=103.0))
    for needle in ("SHORT → LONG", "Структура рынка", "CHoCH вверх", "Delta за час", "всплеск", "риск/доходность"):
        assert needle in loud
    assert whats_new(None, snap()) == ""


def test_menu_callbacks_fit_telegram_limit_and_mark_current():
    kb = menu("1000SHIB/USDT", "15m", "signal")
    data = [b.callback_data for row in kb.inline_keyboard for b in row]
    assert all(len(d.encode()) <= 64 for d in data)
    labels = [b.text for row in kb.inline_keyboard for b in row]
    assert "• 🎯 Long/Short" in labels and "• 15m" in labels


class FakeTelegram:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.bot = Bot("123:abc")

        async def fake(session, bot, method, *a, **k):
            name = type(method).__name__
            self.calls.append((name, getattr(method, "text", None) or getattr(method, "caption", None) or ""))
            if name in ("SendMessage", "EditMessageText", "SendPhoto"):
                return Message.model_validate({"message_id": len(self.calls), "date": 1700000000,
                                               "chat": {"id": 1, "type": "private"}, "text": "x"}).as_(bot)
            return True
        type(self.bot.session).make_request = fake
        handlers.router._parent_router = None  # one shared module-level router, one Dispatcher per test
        self.dp = Dispatcher()
        self.dp.include_router(handlers.router)

    async def press(self, data: str):
        self.calls.clear()
        msg = {"message_id": 5, "date": 1700000000, "chat": {"id": 1, "type": "private"}, "text": "x"}
        await self.dp.feed_update(self.bot, Update.model_validate({"update_id": 1, "callback_query": {
            "id": "1", "chat_instance": "x", "from": {"id": 42, "is_bot": False, "first_name": "a"}, "data": data, "message": msg}}))
        return [c for c in self.calls if c[0] != "AnswerCallbackQuery"]

    async def send(self, text: str):
        self.calls.clear()
        entities = [{"type": "bot_command", "offset": 0, "length": len(text.split()[0])}] if text.startswith("/") else []
        await self.dp.feed_update(self.bot, Update.model_validate({"update_id": 2, "message": {
            "message_id": 1, "date": 1700000000, "chat": {"id": 1, "type": "private"},
            "from": {"id": 43, "is_bot": False, "first_name": "Глеб"}, "text": text, "entities": entities}}))
        return list(self.calls)


@pytest.fixture
def tg(monkeypatch):
    async def fake_build(action, symbol, tf):
        if action == "chart":
            image = BytesIO(b"png")
            image.name = "c.png"
            return View(photo=image, caption="cap")
        if action == "boom":
            raise RuntimeError("Стакан рассинхронизирован")
        return View(f"<b>{action}</b> {symbol} {tf}")
    monkeypatch.setattr(handlers, "build_view", fake_build)
    monkeypatch.setattr(handlers, "MIN_INTERVAL", 0.0)
    handlers._busy.clear()
    return FakeTelegram()


async def test_start_shows_welcome_with_inline_keyboard(tg):
    calls = await tg.send("/start")
    assert calls[0][0] == "SendMessage" and "Привет, Глеб" in calls[0][1]


async def test_button_edits_message_in_place(tg):
    calls = await tg.press("m:signal:SOL/USDT:1h")
    assert [c[0] for c in calls] == ["EditMessageText", "EditMessageText"]  # loading, then result
    assert "signal" in calls[-1][1]


async def test_timeframe_and_refresh_buttons_rerun_current_screen(tg):
    assert "4h" in (await tg.press("t:smc:SOL/USDT:4h"))[-1][1]
    assert "smc" in (await tg.press("r:smc:SOL/USDT:4h"))[-1][1]


async def test_chart_replaces_message_with_photo(tg):
    names = [c[0] for c in await tg.press("m:chart:SOL/USDT:1h")]
    assert names == ["EditMessageText", "DeleteMessage", "SendPhoto"]


async def test_error_is_shown_in_place_with_menu(tg):
    calls = await tg.press("m:boom:SOL/USDT:1h")
    assert calls[-1][0] == "EditMessageText" and "рассинхронизирован" in calls[-1][1]


async def test_stale_or_bad_callback_does_not_crash(tg):
    assert await tg.press("m:signal:SOL/USDT:9x") == []


async def test_command_and_plain_ticker(tg):
    assert "analyze" in (await tg.send("/analyze SOL 4h"))[-1][1]
    assert "signal" in (await tg.send("btc"))[-1][1]
    assert "Таймфрейм" in (await tg.send("/analyze SOL 7m"))[0][1]


async def test_busy_user_is_not_processed_twice(tg):
    handlers._busy.add(42)
    assert await tg.press("m:signal:SOL/USDT:1h") == []
    handlers._busy.clear()
