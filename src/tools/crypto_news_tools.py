import aiohttp
from langchain_core.tools import tool

async def get_crypto_news(query: str, max_pages: int = 1) -> str:
    """Функция для отправки запроса к MCP серверу через HTTP"""
    url = "http://127.0.0.1:6274/get_crypto_news"  # Исправленный URL с правильным портом
    payload = {"query": query, "max_pages": max_pages}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    error_text = await response.text()
                    return f"Ошибка при запросе к серверу: {response.status}, {error_text}"
    except Exception as e:
        return f"Ошибка при подключении к MCP серверу: {str(e)}\n\nУбедитесь, что сервер запущен на порту 6274."

@tool
async def fetch_crypto_news(query: str, max_pages: int = 1) -> str:
    """
    Получает новости о криптовалютах по заданному запросу.
    
    Args:
        query: Поисковый запрос для фильтрации новостей (например, "Bitcoin", "Ethereum", "DeFi")
        max_pages: Максимальное количество страниц результатов для обработки (по умолчанию 1)
    """
    return await get_crypto_news(query, max_pages)