# Импортируем все инструменты для доступности через tools.*
from .coingecko_tools import get_token_price, get_trending_coins, search_cryptocurrencies
from .defi_protocol_tools import analyze_protocol, analyze_pools_geckoterminal
from .token_analysis_tools import get_token_historical_data
from .holder_analysis_tools import analyze_token_holders
# from .crypto_news_tools import fetch_crypto_news
from .hyperliquid_tools import (
    get_crypto_price,
    get_klines_history,
    execute_trade,
    confirm_trade,
    get_market_info,
    get_account_info
)
# Добавляем новые инструменты LlamaFeed
from .llamafeed_tools import (
    get_crypto_news,
    get_crypto_tweets,
    get_crypto_hacks,
    get_token_unlocks,
    get_project_raises,
    get_polymarket_data,
    get_market_summary
)

# Список всех доступных инструментов для импорта
__all__ = [
    'get_token_price',
    'get_trending_coins',
    'search_cryptocurrencies',
    'analyze_protocol',
    'analyze_pools_geckoterminal',
    'get_token_historical_data',
    'analyze_token_holders',
    # 'fetch_crypto_news',
    # Новые инструменты HyperLiquid
    'get_crypto_price',
    'get_klines_history',
    'execute_trade',
    'confirm_trade',
    'get_market_info',
    'get_account_info',
        
    # Инструменты LlamaFeed
    'get_crypto_news',
    'get_crypto_tweets',
    'get_crypto_hacks',
    'get_token_unlocks',
    'get_project_raises',
    'get_polymarket_data',
    'get_market_summary'
]