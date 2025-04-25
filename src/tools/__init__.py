# Импортируем все инструменты для доступности через tools.*
from .coingecko_tools import get_token_price, get_trending_coins, search_cryptocurrencies
from .defi_protocol_tools import analyze_protocol, analyze_pools_geckoterminal
from .token_analysis_tools import get_token_historical_data
from .holder_analysis_tools import analyze_token_holders
# from .crypto_news_tools import fetch_crypto_news

# Список всех доступных инструментов для импорта
__all__ = [
    'get_token_price',
    'get_trending_coins',
    'search_cryptocurrencies',
    'analyze_protocol',
    'analyze_pools_geckoterminal',
    'get_token_historical_data',
    'analyze_token_holders',
    # 'fetch_crypto_news' 
]