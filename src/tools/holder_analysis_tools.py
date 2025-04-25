import aiohttp
import pandas as pd
from langchain_core.tools import tool
from config.settings import BITQUERY_API_KEY

@tool
async def analyze_token_holders(token_address: str, token_label: str, chain: str = "ethereum") -> str:
    """
    Анализирует распределение держателей токена используя Bitquery.

    Args:
        token_address: адрес контракта токена
        token_label: читаемое название токена
        chain: блокчейн, на котором размещен токен (по умолчанию "ethereum")
    """
    # Здесь был бы запрос к Bitquery API, но из-за необходимости API ключа,
    # предоставим тестовые данные для демонстрации

    if not BITQUERY_API_KEY or BITQUERY_API_KEY == "YOUR_BITQUERY_API_KEY":
        return (
            f"Анализ держателей токена {token_label} ({token_address}) на {chain}:\n\n"
            "Для получения реальных данных необходимо указать действительный API ключ Bitquery.\n\n"
            "Пример анализа распределения токенов:\n"
            "- Топ-1 адрес: 0x123...abc - 25.3% токенов\n"
            "- Топ-2 адрес: 0x456...def - 15.7% токенов\n"
            "- Топ-3 адрес: 0x789...ghi - 8.2% токенов\n"
            "- Другие адреса: 50.8% токенов\n\n"
            "Концентрация: средняя (топ-10 адресов владеют примерно 60% токенов)"
        )

    url = "https://streaming.bitquery.io/graphql"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {BITQUERY_API_KEY}"
    }

    query = f"""
    {{
      EVM(dataset: archive, network: {chain}) {{
        TokenHolders(
          tokenSmartContract: "{token_address}"
          limit: {{count: 1000}}
          orderBy: {{descending: Balance_Amount}}
        ) {{
          Holder {{
            Address
          }}
          Balance {{
            Amount
          }}
        }}
      }}
    }}
    """

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json={"query": query}) as response:
            if response.status != 200:
                return f"Ошибка запроса: {response.status}"

            try:
                response_data = await response.json()
                holders = response_data['data']['EVM']['TokenHolders']
            except Exception as e:
                return f"Ошибка парсинга JSON: {e}"

    data = []
    for holder in holders:
        balance_info = holder.get('Balance', {})
        amount_str = balance_info.get('Amount')
        if amount_str is None:
            continue

        try:
            balance = float(amount_str)
        except:
            balance = 0

        address = holder.get('Holder', {}).get('Address', 'Unknown')
        data.append({
            'Balance': balance,
            'Holder Address': address
        })

    if not data:
        return "Нет данных о держателях токена."

    df = pd.DataFrame(data)
    total_balance = df['Balance'].sum()
    df = df.sort_values(by='Balance', ascending=False)
    df['Percentage'] = (df['Balance'] / total_balance) * 100

    # Анализ распределения
    result = f"=== Анализ держателей токена {token_label} ===\n\n"

    # Топ-10 держателей
    result += "Топ-10 держателей:\n"
    for i, (_, holder) in enumerate(df.head(10).iterrows(), 1):
        addr = holder['Holder Address']
        pct = holder['Percentage']
        result += f"{i}. {addr[:6]}...{addr[-4:]} - {pct:.2f}%\n"

    # Статистика концентрации
    top10_pct = df.head(10)['Percentage'].sum()
    top50_pct = df.head(50)['Percentage'].sum()

    result += f"\nТоп-10 держателей владеют {top10_pct:.2f}% токенов\n"
    result += f"Топ-50 держателей владеют {top50_pct:.2f}% токенов\n"

    # Оценка концентрации
    if top10_pct > 90:
        concentration = "Очень высокая"
    elif top10_pct > 70:
        concentration = "Высокая"
    elif top10_pct > 50:
        concentration = "Средняя"
    else:
        concentration = "Низкая"

    result += f"\nКонцентрация токенов: {concentration}"

    return result
