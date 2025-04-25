"""Схемы данных для криптовалютных инструментов агента."""

from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum
import time


class ToolType(str, Enum):
    """Типы инструментов в системе."""
    TOKEN_PRICE = "token_price"
    TRENDING_COINS = "trending_coins"
    SEARCH_CRYPTO = "search_crypto"
    PROTOCOL_ANALYSIS = "protocol_analysis"
    POOLS_ANALYSIS = "pools_analysis"
    TOKEN_HISTORY = "token_history"
    HOLDERS_ANALYSIS = "holders_analysis"
    CRYPTO_NEWS = "crypto_news"
    
    # Типы инструментов HyperLiquid
    HYPERLIQUID_PRICE = "hyperliquid_price"
    HYPERLIQUID_KLINES = "hyperliquid_klines"
    HYPERLIQUID_TRADE = "hyperliquid_trade"
    HYPERLIQUID_CONFIRM_TRADE = "hyperliquid_confirm_trade"
    HYPERLIQUID_MARKET_INFO = "hyperliquid_market_info"
    HYPERLIQUID_ACCOUNT_INFO = "hyperliquid_account_info"
    
    # Типы инструментов LlamaFeed
    LLAMAFEED_NEWS = "llamafeed_news"
    LLAMAFEED_TWEETS = "llamafeed_tweets"
    LLAMAFEED_HACKS = "llamafeed_hacks"
    LLAMAFEED_UNLOCKS = "llamafeed_unlocks"
    LLAMAFEED_RAISES = "llamafeed_raises"
    LLAMAFEED_POLYMARKET = "llamafeed_polymarket"
    LLAMAFEED_MARKET_SUMMARY = "llamafeed_market_summary"


# === Модели для get_token_price ===

class TokenPriceRequest(BaseModel):
    """Запрос для получения цены токена."""
    symbol: str = Field(..., description="Символ токена (например, BTC, ETH)")


class TokenPriceResponse(BaseModel):
    """Ответ с ценой токена."""
    symbol: str
    price: Optional[float] = None
    currency: str = "USD"
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    success: bool = True
    
    @validator('success', always=True)
    def check_success(cls, v, values):
        return values.get('error') is None and values.get('price') is not None


# === Модели для get_trending_coins ===

class TrendingCoinsRequest(BaseModel):
    """Запрос для получения трендовых монет."""
    limit: Optional[int] = Field(None, description="Максимальное количество монет для отображения")
    include_platform: bool = Field(False, description="Включать информацию о платформах")


class PlatformInfo(BaseModel):
    """Информация о платформе токена."""
    platform_name: str
    contract_address: str


class TrendingCoin(BaseModel):
    """Модель трендовой монеты."""
    name: str
    symbol: str
    market_cap_rank: Optional[int] = None
    score: Optional[float] = None
    platforms: Optional[Dict[str, str]] = None


class TrendingCoinsResponse(BaseModel):
    """Ответ со списком трендовых монет."""
    coins: List[TrendingCoin] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None
    success: bool = True
    
    @validator('success', always=True)
    def check_success(cls, v, values):
        return values.get('error') is None


# === Модели для search_cryptocurrencies ===

class SearchCryptoRequest(BaseModel):
    """Запрос для поиска криптовалюты."""
    query: str = Field(..., description="Поисковый запрос (например, 'bitcoin' или 'btc')")
    exact_match: bool = Field(False, description="Искать только точные совпадения")


class SearchCryptoResult(BaseModel):
    """Результат поиска криптовалюты."""
    id: str
    name: str
    symbol: str
    market_cap_rank: Optional[int] = None


class SearchCryptoResponse(BaseModel):
    """Ответ с результатами поиска криптовалют."""
    results: List[SearchCryptoResult] = Field(default_factory=list)
    query: str
    total_found: int
    timestamp: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None
    success: bool = True


# === Модели для analyze_protocol ===

class ProtocolAnalysisRequest(BaseModel):
    """Запрос для анализа протокола."""
    protocol_id: str = Field(..., description="ID протокола в DeFiLlama")
    protocol_label: str = Field(..., description="Читаемое название протокола")
    chains_to_show: List[str] = Field(..., description="Список сетей для анализа TVL")


class ChainTVLData(BaseModel):
    """Данные TVL по отдельной сети."""
    chain: str
    tvl: float


class ProtocolTVLData(BaseModel):
    """Данные TVL протокола."""
    current_tvl: float
    monthly_change_pct: Optional[float] = None
    chains: List[ChainTVLData] = Field(default_factory=list)


class ProtocolAnalysisResponse(BaseModel):
    """Ответ с анализом протокола."""
    protocol_id: str
    protocol_label: str
    market_cap: Optional[float] = None
    tvl_data: ProtocolTVLData
    timestamp: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None
    success: bool = True


# === Модели для analyze_pools_geckoterminal ===

class PoolsAnalysisRequest(BaseModel):
    """Запрос для анализа пулов протокола."""
    network: str = Field(..., description="Идентификатор сети (например, 'ethereum', 'arbitrum')")
    protocol_id: str = Field(..., description="Идентификатор DEX в GeckoTerminal (например, 'uniswap')")
    protocol_label: str = Field(..., description="Читаемое название протокола")


class PoolData(BaseModel):
    """Данные о пуле токенов."""
    pool_address: str
    token_pair: str
    volume_24h: float
    liquidity_usd: float
    price_change_24h: float
    transactions: int
    buys: Optional[int] = None
    sells: Optional[int] = None


class PoolsAnalysisResponse(BaseModel):
    """Ответ с анализом пулов протокола."""
    network: str
    protocol_id: str
    protocol_label: str
    normalized_network: str
    normalized_protocol: str
    total_pools: int
    top_pools: List[PoolData] = Field(default_factory=list)
    total_volume_24h: float
    average_volume_per_pool: float
    timestamp: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None
    success: bool = True


# === Модели для get_token_historical_data ===

class TokenHistoricalRequest(BaseModel):
    """Запрос для получения исторических данных токена."""
    token_id: str = Field(..., description="Идентификатор токена в CoinGecko")
    token_label: str = Field(..., description="Читаемое название токена")
    vs_currency: str = Field("usd", description="Валюта для расчетов (например, 'usd', 'eur')")
    days: str = Field("90", description="Период в днях для запроса данных")


class PricePoint(BaseModel):
    """Точка данных цены."""
    timestamp: int
    price: float
    date: datetime


class MarketCapPoint(BaseModel):
    """Точка данных рыночной капитализации."""
    timestamp: int
    market_cap: float
    date: datetime


class VolumePoint(BaseModel):
    """Точка данных объема торгов."""
    timestamp: int
    volume: float
    date: datetime


class TokenHistoricalResponse(BaseModel):
    """Ответ с историческими данными токена."""
    token_id: str
    token_label: str
    vs_currency: str
    days: str
    current_price: float
    price_change_pct: float
    min_price: float
    min_date: datetime
    max_price: float
    max_date: datetime
    current_market_cap: float
    market_cap_change_pct: float
    current_volume: float
    avg_volume: float
    prices: Optional[List[PricePoint]] = None  # Можно включать полные данные при необходимости
    timestamp: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None
    success: bool = True


# === Модели для analyze_token_holders ===

class HoldersAnalysisRequest(BaseModel):
    """Запрос для анализа держателей токена."""
    token_address: str = Field(..., description="Адрес контракта токена")
    token_label: str = Field(..., description="Читаемое название токена")
    chain: str = Field("ethereum", description="Блокчейн токена (по умолчанию 'ethereum')")


class HolderInfo(BaseModel):
    """Информация о держателе токена."""
    address: str
    balance: float
    percentage: float


class ConcentrationLevel(str, Enum):
    """Уровни концентрации токенов."""
    VERY_HIGH = "Очень высокая"
    HIGH = "Высокая"
    MEDIUM = "Средняя"
    LOW = "Низкая"


class HoldersAnalysisResponse(BaseModel):
    """Ответ с анализом держателей токена."""
    token_address: str
    token_label: str
    chain: str
    top_holders: List[HolderInfo] = Field(default_factory=list)
    top10_percentage: float
    top50_percentage: float
    concentration: ConcentrationLevel
    total_holders_analyzed: int
    timestamp: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None
    success: bool = True


# === Модели для fetch_crypto_news ===

class CryptoNewsRequest(BaseModel):
    """Запрос для получения новостей о криптовалютах."""
    query: str = Field(..., description="Поисковый запрос (например, 'Bitcoin', 'Ethereum')")
    max_pages: int = Field(1, description="Максимальное количество страниц результатов")


class NewsArticle(BaseModel):
    """Модель новостной статьи."""
    title: str
    url: str
    source: str
    published_date: Optional[datetime] = None
    summary: Optional[str] = None


class CryptoNewsResponse(BaseModel):
    """Ответ с новостями о криптовалютах."""
    query: str
    articles: List[NewsArticle] = Field(default_factory=list)
    total_found: int
    timestamp: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None
    success: bool = True


# === Общие модели для работы с инструментами ===

class ToolRequest(BaseModel):
    """Обобщенная модель запроса к инструменту."""
    tool_type: ToolType
    request_id: str = Field(default_factory=lambda: f"req_{datetime.now().timestamp()}")
    params: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class ToolResponse(BaseModel):
    """Обобщенная модель ответа от инструмента."""
    tool_type: ToolType
    request_id: str
    success: bool = True
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)

    def set_execution_time(self, start_time: float):
        """Устанавливает время выполнения инструмента."""
        self.execution_time = time.time() - start_time