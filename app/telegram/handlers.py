import html
import logging
import re
import time

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from app.market.binance import TIMEFRAMES, MarketDataError
from app.telegram.services import provider, snapshots
from app.telegram.views import build_view, whats_new

logger = logging.getLogger(__name__)
router = Router()

POPULAR = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"]
TF_ROW = ["15m", "1h", "4h", "1d", "1w"]
VIEWS = [  # (button label, action)
    ("🎯 Long/Short", "signal"), ("📊 Анализ", "analyze"),
    ("🧠 Smart Money", "smc"), ("📐 Тех. анализ", "ta"),
    ("🧱 Уровни", "levels"), ("📈 Объемы", "volume"),
    ("🐋 Крупные сделки", "whales"), ("📚 Стакан", "orderbook"),
    ("📉 График", "chart"), ("🧪 Бэктест", "backtest"),
]
COMMANDS = ["analyze", "signal", "smc", "ta", "levels", "volume", "whales", "orderbook", "chart", "backtest"]
LOADING = {"backtest": "⏳ Считаю бэктест на истории, это займет до 30 секунд..."}
MIN_INTERVAL = 1.5          # seconds between heavy requests from one user
SNAPSHOT_MAX_AGE = 2 * 3600  # "what's new" is shown if the same screen was seen within this many seconds
_busy: set[int] = set()
_last_request: dict[int, float] = {}


# ---------- keyboards ----------

def menu(symbol: str = "SOL/USDT", timeframe: str = "1h", current: str | None = None) -> InlineKeyboardMarkup:
    rows = []
    for i in range(0, len(VIEWS), 2):
        rows.append([InlineKeyboardButton(text=("• " if action == current else "") + label,
                                          callback_data=f"m:{action}:{symbol}:{timeframe}")
                     for label, action in VIEWS[i:i + 2]])
    rows.append([InlineKeyboardButton(text=("• " if tf == timeframe else "") + tf,
                                      callback_data=f"t:{current or 'analyze'}:{symbol}:{tf}") for tf in TF_ROW])
    rows.append([
        InlineKeyboardButton(text="🔄 Обновить", callback_data=f"r:{current or 'analyze'}:{symbol}:{timeframe}"),
        InlineKeyboardButton(text="🔔 Уведомления", callback_data="alerts"),
        InlineKeyboardButton(text="🏠 Меню", callback_data="home"),
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def home_menu() -> InlineKeyboardMarkup:
    coins = [InlineKeyboardButton(text=c, callback_data=f"c:{c}/USDT") for c in POPULAR]
    return InlineKeyboardMarkup(inline_keyboard=[
        coins[:3], coins[3:],
        [InlineKeyboardButton(text="🔍 Ввести свой тикер", callback_data="custom")],
        [InlineKeyboardButton(text="❓ Помощь", callback_data="help")],
    ])


WELCOME = (
    "👋 Привет, {name}! Я <b>КриптоСигнал</b> - бот рыночной аналитики по Binance Spot.\n\n"
    "Что я умею:\n"
    "🎯 направление Long/Short с зоной входа, стопом и целями\n"
    "🧠 Smart Money и 📐 технический анализ\n"
    "🧱 уровни, 📈 объемы, 🐋 крупные сделки, 📚 стакан\n"
    "🧪 бэктест: как такие сигналы вели себя на истории\n\n"
    "Выберите монету кнопкой ниже или отправьте тикер (например <code>SOL</code> или "
    "<code>ETH/USDT 4h</code>). Все экраны переключаются кнопками в одном сообщении, "
    "а «🔄 Обновить» покажет, что изменилось с прошлого раза."
)
HELP = (
    "<b>Помощь</b>\n"
    "Команды: /signal SOL (long/short), /smc SOL, /ta SOL, /analyze SOL, /levels BTC, /volume ETH, "
    "/chart SOL, /whales SOL, /orderbook SOL, /backtest SOL.\n"
    "Таймфреймы: 5m, 15m, 1h, 4h, 1d, 1w (например <code>/analyze SOL 4h</code>).\n"
    "Данные публичные; бот не торгует и не дает финансовых рекомендаций."
)


# ---------- message helpers ----------

def _name(user) -> str:
    return html.escape(user.first_name) if user else "трейдер"


def split_text(text: str, limit: int = 4000) -> list[str]:
    """Split on line boundaries so every part fits a Telegram message."""
    parts, current = [], ""
    for line in text.split("\n"):
        if current and len(current) + len(line) + 1 > limit:
            parts.append(current)
            current = ""
        current += ("\n" if current else "") + line[:limit]
    return [*parts, current] if current else parts


async def show_text(msg: Message, text: str, markup: InlineKeyboardMarkup | None, edit: bool = True) -> None:
    """Show text by editing `msg` in place (or replacing it if it is a photo / cannot be edited)."""
    parts = split_text(text) or [" "]
    first_markup = markup if len(parts) == 1 else None
    if edit and not msg.photo:
        try:
            await msg.edit_text(parts[0], parse_mode="HTML", reply_markup=first_markup)
        except TelegramBadRequest as exc:
            if "not modified" not in str(exc):
                logger.warning("edit failed (%s), sending a new message", exc)
                await msg.answer(parts[0], parse_mode="HTML", reply_markup=first_markup)
    else:
        if edit and msg.photo:
            await _delete_quietly(msg)
        await msg.answer(parts[0], parse_mode="HTML", reply_markup=first_markup)
    for i, part in enumerate(parts[1:], start=1):
        await msg.answer(part, parse_mode="HTML", reply_markup=markup if i == len(parts) - 1 else None)


async def _delete_quietly(msg: Message) -> None:
    try:
        await msg.delete()
    except TelegramBadRequest:
        logger.debug("message already deleted")


def parse_request(raw: str | None) -> tuple[str, str]:
    args = (raw or "").strip().split()
    if not args:
        raise ValueError("Укажите тикер, например /analyze SOL или SOL/USDT.")
    timeframe = args[1].lower() if len(args) > 1 else "1h"
    if timeframe not in TIMEFRAMES:
        raise ValueError("Таймфрейм: 5m, 15m, 1h, 4h, 1d или 1w.")
    return provider.normalize_symbol(args[0]), timeframe


def rate_problem(user_id: int) -> str | None:
    """Per-user protection of the exchange API: one request at a time and a minimum gap between requests."""
    if user_id in _busy:
        return "Предыдущий запрос еще считается, подождите пару секунд."
    if time.monotonic() - _last_request.get(user_id, 0.0) < MIN_INTERVAL:
        return "Слишком часто, подождите секунду."
    return None


async def run_view(msg: Message, user_id: int, action: str, symbol: str, timeframe: str) -> None:
    """Compute a screen and show it in place of `msg` (which becomes the loading placeholder)."""
    problem = rate_problem(user_id)
    if problem:
        await msg.answer(problem)
        return
    _busy.add(user_id)
    _last_request[user_id] = time.monotonic()
    markup = menu(symbol, timeframe, action)
    try:
        await show_text(msg, LOADING.get(action, f"⏳ Загружаю {symbol} · {timeframe}..."), None)
        view = await build_view(action, symbol, timeframe)
        key = (msg.chat.id, symbol, timeframe, action)
        news = ""
        previous = snapshots.get(key)
        if previous and view.snapshot and (view.snapshot["time"] - previous["time"]).total_seconds() < SNAPSHOT_MAX_AGE:
            news = whats_new(previous, view.snapshot)
        if view.snapshot:
            snapshots[key] = view.snapshot
        if view.photo:
            await _delete_quietly(msg)
            await msg.answer_photo(BufferedInputFile(view.photo.getvalue(), filename=view.photo.name),
                                   caption=(news and news.split("\n\n")[0] + "\n" or "")[:600] + view.caption,
                                   parse_mode="HTML", reply_markup=markup)
        else:
            await show_text(msg, news + (view.text or ""), markup)
    except (ValueError, MarketDataError, RuntimeError) as exc:
        await show_text(msg, f"⚠️ {html.escape(str(exc))}", markup)
    except Exception:
        logger.exception("Unexpected analysis error")
        await show_text(msg, "⚠️ Не удалось получить данные. Проверьте доступность Binance и повторите позже.", markup)
    finally:
        _busy.discard(user_id)


# ---------- commands ----------

@router.message(CommandStart())
async def start(message: Message) -> None:
    await message.answer(WELCOME.format(name=_name(message.from_user)), parse_mode="HTML", reply_markup=home_menu())


@router.message(Command("help"))
async def help_command(message: Message) -> None:
    await message.answer(HELP, parse_mode="HTML", reply_markup=home_menu())


@router.message(Command(*COMMANDS))
async def command_analysis(message: Message, command: CommandObject) -> None:
    await request_from_text(message, command.args, command.command)


async def request_from_text(message: Message, raw: str | None, action: str) -> None:
    try:
        symbol, timeframe = parse_request(raw)
    except ValueError as exc:
        await message.answer(str(exc))
        return
    placeholder = await message.answer("⏳ ...")
    await run_view(placeholder, message.from_user.id if message.from_user else message.chat.id, action, symbol, timeframe)


@router.message(F.text.regexp(re.compile(r"^[A-Za-z0-9/]{2,20}(?:\s+(?:5m|15m|1h|4h|1d|1w))?$", re.IGNORECASE)))
async def ticker_message(message: Message) -> None:
    await request_from_text(message, message.text, "signal")


# ---------- inline buttons: everything is edited in place ----------

@router.callback_query(F.data == "home")
async def cb_home(callback: CallbackQuery) -> None:
    await callback.answer()
    await show_text(callback.message, WELCOME.format(name=_name(callback.from_user)), home_menu())


@router.callback_query(F.data == "help")
async def cb_help(callback: CallbackQuery) -> None:
    await callback.answer()
    await show_text(callback.message, HELP, home_menu())


@router.callback_query(F.data == "custom")
async def cb_custom(callback: CallbackQuery) -> None:
    await callback.answer()
    await show_text(callback.message, "Отправьте тикер сообщением, например <code>AVAX</code> или <code>LINK 4h</code>.",
                    home_menu())


@router.callback_query(F.data == "alerts")
async def cb_alerts(callback: CallbackQuery) -> None:
    await callback.answer("🔔 Уведомления появятся на следующем этапе: алерты по уровням, объему и сделкам.", show_alert=True)


@router.callback_query(F.data.startswith("c:"))
async def cb_coin(callback: CallbackQuery) -> None:
    await callback.answer()
    symbol = callback.data[2:]
    await show_text(callback.message, f"<b>{html.escape(symbol)}</b> - выберите, что показать:", menu(symbol, "1h"))


@router.callback_query(F.data.regexp(r"^[mtr]:"))
async def cb_view(callback: CallbackQuery) -> None:
    """m: open a screen | t: same screen on another timeframe | r: refresh the current screen."""
    try:
        _, action, symbol, timeframe = callback.data.split(":", 3)
        if timeframe not in TIMEFRAMES:
            raise ValueError
    except ValueError:
        await callback.answer("Кнопка устарела, отправьте тикер заново.", show_alert=True)
        return
    problem = rate_problem(callback.from_user.id)
    if problem:
        await callback.answer(problem)
        return
    await callback.answer()
    await run_view(callback.message, callback.from_user.id, action, symbol, timeframe)
