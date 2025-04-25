import aiohttp
import pandas as pd
from langchain_core.tools import tool
from config.settings import COINGECKO_API_KEY

@tool
async def get_token_historical_data(token_id: str, token_label: str, vs_currency: str = 'usd', days: str = '90') -> str:
    """
    Получает исторические данные по токену с CoinGecko и анализирует их.

    Args:
        token_id: идентификатор токена в CoinGecko (например, 'bitcoin', 'ethereum')
        token_label: читаемое название токена для отображения
        vs_currency: валюта, в которой выражены значения (по умолчанию 'usd')
        days: период в днях для запроса данных (по умолчанию '90')
    """
    url = f'https://api.coingecko.com/api/v3/coins/{token_id}/market_chart'
    params = {'vs_currency': vs_currency, 'days': days}
    
    if COINGECKO_API_KEY:
        params['x_cg_demo_api_key'] = COINGECKO_API_KEY
        
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            if response.status != 200:
                response_text = await response.text()
                return f"Ошибка при получении данных для {token_label}: {response_text}"

            data = await response.json()

    # Преобразуем данные в датафреймы
    prices = pd.DataFrame(data.get('prices', []), columns=['timestamp', 'price'])
    market_caps = pd.DataFrame(data.get('market_caps', []), columns=['timestamp', 'market_cap'])
    volumes = pd.DataFrame(data.get('total_volumes', []), columns=['timestamp', 'volume'])

    for df in [prices, market_caps, volumes]:
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Анализ данных
    result = f"=== Анализ данных токена {token_label} за последние {days} дней ===\n\n"

    # Текущая цена и изменения
    current_price = prices.iloc[-1]['price']
    start_price = prices.iloc[0]['price']
    price_change = ((current_price - start_price) / start_price) * 100

    result += f"Текущая цена: ${current_price:.6f}\n"
    result += f"Изменение цены за период: {price_change:.2f}%\n"

    # Минимальная и максимальная цены
    min_price = prices['price'].min()
    max_price = prices['price'].max()
    min_date = prices.loc[prices['price'].idxmin(), 'date']
    max_date = prices.loc[prices['price'].idxmax(), 'date']

    result += f"Минимальная цена: ${min_price:.6f} ({min_date.strftime('%Y-%m-%d')})\n"
    result += f"Максимальная цена: ${max_price:.6f} ({max_date.strftime('%Y-%m-%d')})\n\n"

    # Рыночная капитализация
    current_market_cap = market_caps.iloc[-1]['market_cap']
    start_market_cap = market_caps.iloc[0]['market_cap']
    market_cap_change = ((current_market_cap - start_market_cap) / start_market_cap) * 100

    result += f"Текущая рыночная капитализация: ${current_market_cap:,.2f}\n"
    result += f"Изменение рыночной капитализации за период: {market_cap_change:.2f}%\n\n"

    # Объемы торгов
    avg_volume = volumes['volume'].mean()
    current_volume = volumes.iloc[-1]['volume']

    result += f"Текущий объем торгов: ${current_volume:,.2f}\n"
    result += f"Средний объем торгов за период: ${avg_volume:,.2f}\n"

    return result
