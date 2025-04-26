import aiohttp
import pandas as pd
from typing import List
from langchain_core.tools import tool

@tool
def analyze_protocol(protocol_id: str, protocol_label: str, chains_to_show: List[str]) -> str:
    """
    Получает данные протокола с DeFiLlama и анализирует TVL.

    Args:
        protocol_id: идентификатор протокола в DeFiLlama
        protocol_label: читаемое название протокола для отображения
        chains_to_show: список сетей для анализа TVL (например, ["Ethereum", "Arbitrum"])
    """
    import requests
    
    url = f'https://api.llama.fi/protocol/{protocol_id}'
    response = requests.get(url)
    
    if response.status_code != 200:
        return f"Ошибка запроса для {protocol_label}: {response.status_code} {response.reason}"

    try:
        protocol_data = response.json()
    except Exception as e:
        return f"Ошибка декодирования JSON для {protocol_label}: {e}"

    result = f"=== {protocol_label} Summary ===\n\n"

    # Основная информация
    if "mcap" in protocol_data:
        result += f"Market Cap: ${protocol_data['mcap']:,.2f}\n"

    if "tvl" in protocol_data and protocol_data["tvl"]:
        tvl_data = pd.DataFrame(protocol_data["tvl"])

        # Конвертируем Unix timestamp в datetime
        tvl_data['datetime'] = pd.to_datetime(tvl_data['date'], unit='s')
        tvl_data = tvl_data.sort_values('datetime')

        # Получаем текущий TVL (последняя запись)
        current_tvl = tvl_data.iloc[-1]['totalLiquidityUSD']
        result += f"Текущий TVL: ${current_tvl:,.2f}\n"

        # Рассчитываем изменение TVL за месяц
        if len(tvl_data) > 30:
            month_ago_tvl = tvl_data.iloc[-31]['totalLiquidityUSD']
            monthly_change_pct = ((current_tvl - month_ago_tvl) / month_ago_tvl) * 100
            result += f"Изменение TVL за 30 дней: {monthly_change_pct:.2f}%\n"

    # Анализ TVL по сетям
    chain_data = {}
    for chain in protocol_data.get('chainTvls', {}):
        if chain in chains_to_show:
            chain_info = protocol_data['chainTvls'][chain]
            if 'tvl' in chain_info and chain_info['tvl']:
                chain_tvl = chain_info['tvl'][-1]['totalLiquidityUSD']
                chain_data[chain] = chain_tvl

    if chain_data:
        result += "\nTVL по сетям:\n"
        for chain, tvl in sorted(chain_data.items(), key=lambda x: x[1], reverse=True):
            result += f"- {chain}: ${tvl:,.2f}\n"

    return result

@tool
async def analyze_pools_geckoterminal(network: str, protocol_id: str, protocol_label: str) -> str:
    """
    Анализирует пулы протокола с использованием данных GeckoTerminal.

    Args:
        network: Идентификатор сети (например, "ethereum", "arbitrum")
        protocol_id: Идентификатор DEX в GeckoTerminal (например, "uniswap")
        protocol_label: Читаемое название протокола
    """
    # Маппинг для коррекции идентификаторов сетей
    network_mapping = {
        "ethereum": "eth",
        "arbitrum": "arbitrum_one",
        "binance": "bsc",
        "polygon": "polygon_pos",
        "optimism": "optimism",
        "base": "base",
    }

    # Маппинг для коррекции идентификаторов протоколов
    protocol_mapping = {
        "uniswap": "uniswap_v3",
        "uniswap_v2": "uniswap_v2",
        "uniswap_v3": "uniswap_v3",
        "sushi": "sushiswap",
        "sushiswap": "sushiswap",
        "pancake": "pancakeswap_v2",
        "pancakeswap": "pancakeswap_v2",
        "curve": "curve",
        "balancer": "balancer_ethereum",
    }

    # Нормализуем входные параметры
    normalized_network = network_mapping.get(network.lower(), network.lower())
    normalized_protocol = protocol_mapping.get(protocol_id.lower(), protocol_id.lower())

    print(f"Запрос к GeckoTerminal: сеть={normalized_network}, протокол={normalized_protocol}")

    base_url = "https://api.geckoterminal.com/api/v2"

    # Сначала проверим доступные сети и протоколы, если параметры не очевидны
    if normalized_network not in network_mapping.values() or normalized_protocol not in protocol_mapping.values():
        try:
            # Получаем список доступных сетей
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base_url}/networks") as response:
                    if response.status == 200:
                        networks_data = await response.json()
                        available_networks = [net["id"] for net in networks_data.get("data", [])]

                        # Если наша нормализованная сеть не найдена, ищем ближайшую по имени
                        if normalized_network not in available_networks:
                            for net in networks_data.get("data", []):
                                if network.lower() in net["attributes"].get("name", "").lower():
                                    normalized_network = net["id"]
                                    break

            # Если сеть найдена, получаем список доступных протоколов для этой сети
            if normalized_network:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{base_url}/networks/{normalized_network}/dexes") as response:
                        if response.status == 200:
                            dexes_data = await response.json()
                            available_dexes = [dex["id"] for dex in dexes_data.get("data", [])]

                            # Если наш нормализованный протокол не найден, ищем ближайший по имени
                            if normalized_protocol not in available_dexes:
                                for dex in dexes_data.get("data", []):
                                    dex_name = dex["attributes"].get("name", "").lower()
                                    if protocol_id.lower() in dex_name or protocol_id.lower() in dex["id"]:
                                        normalized_protocol = dex["id"]
                                        break
        except Exception as e:
            print(f"Ошибка при получении доступных сетей/протоколов: {e}")

    # Формируем URL с нормализованными параметрами
    pools_url = f"{base_url}/networks/{normalized_network}/dexes/{normalized_protocol}/pools"
    headers = {"Accept": "application/json"}

    # Выполняем запрос к API
    async with aiohttp.ClientSession() as session:
        async with session.get(pools_url, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                print(f"Ошибка запроса: {response.status} - {error_text}")

                # Формируем понятное сообщение об ошибке
                if response.status == 404:
                    return (f"Ошибка запроса для {protocol_label}: ресурс не найден (404).\n"
                           f"Проверьте корректность идентификаторов:\n"
                           f"- Сеть: {normalized_network} (изначально: {network})\n"
                           f"- Протокол: {normalized_protocol} (изначально: {protocol_id})\n\n"
                           f"Популярные сети: eth, arbitrum_one, bsc, polygon_pos, optimism, base\n"
                           f"Популярные протоколы: uniswap_v3, uniswap_v2, sushiswap, pancakeswap_v2, curve")
                return f"Ошибка запроса для {protocol_label}: {response.status}"

            pools_data = await response.json()

    if 'data' not in pools_data or not pools_data['data']:
        return f"Нет данных о пулах для {protocol_label} (сеть: {normalized_network}, протокол: {normalized_protocol})"

    pools = []
    for pool in pools_data['data']:
        pool_info = pool['attributes']
        tokens = pool_info['name'].split(' / ')
        token_0 = tokens[0]
        token_1 = tokens[1].split(' ')[0] if len(tokens) > 1 else ""
        volume_24h = float(pool_info['volume_usd']['h24'])
        liquidity_usd = float(pool_info['reserve_in_usd'])
        price_change_24h = float(pool_info['price_change_percentage']['h24'])
        tx_buys = int(pool_info['transactions']['h24']['buys'])
        tx_sells = int(pool_info['transactions']['h24']['sells'])
        transactions = tx_buys + tx_sells

        pools.append({
            'pool_address': pool['id'],
            'token_pair': f"{token_0}/{token_1}",
            'volume_24h': volume_24h,
            'liquidity_usd': liquidity_usd,
            'price_change_24h': price_change_24h,
            'transactions': transactions
        })

    df_pools = pd.DataFrame(pools)

    result = f"=== Анализ пулов для {protocol_label} ===\n"
    result += f"Сеть: {normalized_network}, Протокол: {normalized_protocol}\n\n"
    result += f"Общее количество пулов (в выборке): {len(df_pools)}\n\n"

    # Находим топ-3 пула по количеству транзакций
    if not df_pools.empty:
        top3_pools = df_pools.sort_values(by="transactions", ascending=False).head(3)
        result += "Топ-3 пула по числу транзакций:\n"

        for i, (_, pool) in enumerate(top3_pools.iterrows(), 1):
            result += f"{i}. {pool['token_pair']} - {pool['transactions']} транзакций\n"
            result += f"   Объем торгов 24ч: ${pool['volume_24h']:,.2f}\n"
            result += f"   Ликвидность: ${pool['liquidity_usd']:,.2f}\n"
            result += f"   Изменение цены 24ч: {pool['price_change_24h']:.2f}%\n\n"

        total_tx_top3 = top3_pools["transactions"].sum()
        result += f"Суммарное количество транзакций в топ-3 пулах: {total_tx_top3}\n"

        # Статистика по объемам торгов
        total_volume = df_pools["volume_24h"].sum()
        avg_volume = df_pools["volume_24h"].mean()
        result += f"\nОбщий объем торгов за 24ч: ${total_volume:,.2f}\n"
        result += f"Средний объем торгов на пул: ${avg_volume:,.2f}\n"
    else:
        result += "Нет данных для анализа пулов.\n"

    return result
