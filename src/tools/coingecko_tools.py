import asyncio
from typing import Optional, List
from langchain_core.tools import tool
from goat_plugins.coingecko.service import CoinGeckoService
from config.settings import COINGECKO_API_KEY

def create_coingecko_service():
    return CoinGeckoService(api_key=COINGECKO_API_KEY)

@tool
async def get_token_price(symbol: str) -> str:
    """
    Получает текущую цену токена по его символу (например, BTC, ETH).
    """
    cg_service = create_coingecko_service()

    search_result = await cg_service.search_coins({
        "query": symbol.lower(),
        "exact_match": True
    })

    coins = search_result.get("coins", [])
    if not coins:
        return f"Не удалось найти токен с символом {symbol.upper()}"

    coin_id = coins[0]["id"]

    price_data = await cg_service.get_coin_price({
        "coin_id": coin_id,
        "vs_currency": "usd",
        "include_market_cap": False,
        "include_24hr_vol": False,
        "include_24hr_change": False,
        "include_last_updated_at": False
    })

    if not price_data or coin_id not in price_data:
        return f"Не удалось получить цену для {symbol.upper()}"

    price = price_data[coin_id]["usd"]
    return f"Текущая цена {symbol.upper()}: {price} USD"

@tool
async def get_trending_coins(limit: Optional[int] = None, include_platform: bool = False) -> str:
    """
    Получает список трендовых криптовалют на CoinGecko.

    Args:
        limit: Максимальное количество монет для отображения (по умолчанию показываются все)
        include_platform: Включать ли информацию о платформах (контрактах)
    """
    cg_service = create_coingecko_service()

    trending_data = await cg_service.get_trending_coins({
        "limit": limit,
        "include_platform": include_platform
    })

    if not trending_data or "coins" not in trending_data:
        return "Не удалось получить информацию о трендовых монетах."

    trending_coins = trending_data["coins"]
    if limit and len(trending_coins) > limit:
        trending_coins = trending_coins[:limit]

    if not trending_coins:
        return "Трендовых монет не найдено."

    result = "Трендовые криптовалюты на CoinGecko:\n\n"

    for idx, coin_data in enumerate(trending_coins, 1):
        coin = coin_data.get("item", {})
        name = coin.get("name", "Неизвестно")
        symbol = coin.get("symbol", "???").upper()
        score = coin.get("score", "N/A")
        market_cap_rank = coin.get("market_cap_rank", "N/A")

        result += f"{idx}. {name} ({symbol}) - Ранг: {market_cap_rank}, Счет: {score}\n"

        if include_platform and "platforms" in coin:
            result += "   Платформы:\n"
            for platform, address in coin["platforms"].items():
                if address:
                    result += f"   - {platform}: {address}\n"

    return result

@tool
async def search_cryptocurrencies(query: str, exact_match: bool = False) -> str:
    """
    Поиск криптовалют на CoinGecko по названию или символу.

    Args:
        query: Поисковый запрос (например, "bitcoin" или "btc")
        exact_match: Если True, возвращает только точные совпадения
    """
    cg_service = create_coingecko_service()

    search_result = await cg_service.search_coins({
        "query": query,
        "exact_match": exact_match
    })

    coins = search_result.get("coins", [])

    if not coins:
        return f"Не найдено криптовалют по запросу '{query}'."

    result = f"Результаты поиска по запросу '{query}':\n\n"

    for idx, coin in enumerate(coins[:10], 1):  # Ограничиваем до 10 результатов
        name = coin.get("name", "Неизвестно")
        symbol = coin.get("symbol", "???").upper()
        market_cap_rank = coin.get("market_cap_rank", "N/A")

        result += f"{idx}. {name} ({symbol})"
        if market_cap_rank != "N/A":
            result += f" - Ранг по капитализации: {market_cap_rank}"
        result += f" - ID: {coin.get('id', 'unknown')}\n"

    if len(coins) > 10:
        result += f"\n...и еще {len(coins) - 10} результатов."

    return result
