import asyncio
import logging

import structlog
from aiogram import Bot, Dispatcher

from app.config import get_settings
from app.telegram.handlers import provider, router


async def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    structlog.configure(processors=[structlog.processors.TimeStamper(fmt="iso"), structlog.processors.JSONRenderer()],
                        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))
    bot = Bot(settings.telegram_bot_token)
    dispatcher = Dispatcher()
    dispatcher.include_router(router)
    try:
        await dispatcher.start_polling(bot)
    finally:
        await provider.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
