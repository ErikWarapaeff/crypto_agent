"""Инструменты для работы с LlamaFeed - получение новостей, твитов, информации о хаках и т.д."""

import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta, timezone
from langchain_core.tools import tool
from pawn.llamafeed_worflow.worflow import LlamaFeedWorkflow

# Создаем синглтон экземпляра LlamaFeedWorkflow для переиспользования
_llamafeed_instance = None

def get_llamafeed_workflow():
    """Возвращает экземпляр LlamaFeedWorkflow (синглтон)"""
    global _llamafeed_instance
    if _llamafeed_instance is None:
        _llamafeed_instance = LlamaFeedWorkflow(openai_model="gpt-4o")
    return _llamafeed_instance

@tool
async def get_crypto_news(days: int = 3) -> str:
    """
    Получает новости о криптовалютах за указанное количество дней.

    Args:
        days: Количество дней для получения новостей (по умолчанию 3)
    """
    workflow = get_llamafeed_workflow()
    since = datetime.now(timezone.utc) - timedelta(days=days)
    result = workflow.invoke(f"Give me crypto news since {since.isoformat()}")

    # Форматируем результат для более удобного чтения
    formatted_result = "🗞️ **ПОСЛЕДНИЕ КРИПТОНОВОСТИ**\n\n"

    for item in result.get('news', [])[:10]:  # Ограничиваем 10 новостями для читаемости
        title = item.get('title', 'Без заголовка')
        pub_date = item.get('pub_date', 'Неизвестно')
        sentiment = item.get('sentiment', 'нейтральный')
        link = item.get('link', '#')

        formatted_result += f"📌 **{title}**\n"
        formatted_result += f"📅 {pub_date}\n"
        formatted_result += f"🔍 Настроение: {sentiment}\n"
        formatted_result += f"🔗 {link}\n\n"

    return formatted_result

@tool
async def get_crypto_tweets(days: int = 3) -> str:
    """
    Получает твиты о криптовалютах от значимых аккаунтов за указанное количество дней.

    Args:
        days: Количество дней для получения твитов (по умолчанию 3)
    """
    workflow = get_llamafeed_workflow()
    since = datetime.now(timezone.utc) - timedelta(days=days)
    result = workflow.invoke(f"Fetch tweets since {since.isoformat()}")

    # Форматируем результат
    formatted_result = "🐦 **ВАЖНЫЕ КРИПТОТВИТЫ**\n\n"

    for item in result.get('tweets', [])[:10]:  # Ограничиваем 10 твитами
        tweet = item.get('tweet', 'Нет текста')
        created_at = item.get('tweet_created_at', 'Неизвестно')
        user_name = item.get('user_name', 'Аноним')
        sentiment = item.get('sentiment', 'нейтральный')

        formatted_result += f"👤 **@{user_name}**\n"
        formatted_result += f"💬 {tweet}\n"
        formatted_result += f"📅 {created_at}\n"
        formatted_result += f"🔍 Настроение: {sentiment}\n\n"

    return formatted_result

@tool
async def get_crypto_hacks(days: int = 30) -> str:
    """
    Получает информацию о хаках и взломах в криптовалютной сфере за указанный период.

    Args:
        days: Количество дней для получения данных о хаках (по умолчанию 30)
    """
    workflow = get_llamafeed_workflow()
    since = datetime.now(timezone.utc) - timedelta(days=days)
    result = workflow.invoke(f"What crypto hacks happened since {since.isoformat()}?")

    # Форматируем результат
    formatted_result = "⚠️ **НЕДАВНИЕ КРИПТОВАЛЮТНЫЕ ХАКИ**\n\n"

    for item in result.get('hacks', []):
        name = item.get('name', 'Неизвестный проект')
        timestamp = item.get('timestamp', 'Неизвестно')
        amount = item.get('amount', 'Неизвестная сумма')
        source_url = item.get('source_url', '#')
        technique = item.get('technique', 'Не указана')

        formatted_result += f"🔐 **{name}**\n"
        formatted_result += f"📅 {timestamp}\n"
        formatted_result += f"💰 Украдено: {amount}\n"
        formatted_result += f"🛠️ Техника: {technique}\n"
        formatted_result += f"🔗 {source_url}\n\n"

    if not result.get('hacks'):
        formatted_result += "За указанный период хаков не обнаружено.\n"

    return formatted_result

@tool
async def get_token_unlocks(days: int = 30) -> str:
    """
    Получает информацию о предстоящих разблокировках токенов.

    Args:
        days: Количество дней для получения данных о разблокировках (по умолчанию 30)
    """
    workflow = get_llamafeed_workflow()
    since = datetime.now(timezone.utc) - timedelta(days=days)
    result = workflow.invoke(f"Are there any unlocks or raises since {since.isoformat()}?")

    # Форматируем результат
    formatted_result = "🔓 **ПРЕДСТОЯЩИЕ РАЗБЛОКИРОВКИ ТОКЕНОВ**\n\n"

    for item in result.get('unlocks', []):
        project = item.get('project', 'Неизвестный проект')
        date = item.get('date', 'Неизвестная дата')
        amount = item.get('amount', 'Неизвестное количество')
        percentage = item.get('percentage', 'Неизвестный процент')

        formatted_result += f"🏢 **{project}**\n"
        formatted_result += f"📅 Дата: {date}\n"
        formatted_result += f"🔢 Количество: {amount}\n"
        if percentage:
            formatted_result += f"📊 Процент от общего предложения: {percentage}\n"
        formatted_result += "\n"

    if not result.get('unlocks'):
        formatted_result += "За указанный период разблокировок не обнаружено.\n"

    return formatted_result

@tool
async def get_project_raises(days: int = 30) -> str:
    """
    Получает информацию о привлечении средств проектами.

    Args:
        days: Количество дней для получения данных о финансировании (по умолчанию 30)
    """
    workflow = get_llamafeed_workflow()
    since = datetime.now(timezone.utc) - timedelta(days=days)
    result = workflow.invoke(f"Are there any unlocks or raises since {since.isoformat()}?")

    # Форматируем результат
    formatted_result = "💸 **НЕДАВНИЕ ПРИВЛЕЧЕНИЯ СРЕДСТВ**\n\n"

    for item in result.get('raises', []):
        project = item.get('project', 'Неизвестный проект')
        date = item.get('date', 'Неизвестная дата')
        amount = item.get('amount', 'Неизвестная сумма')
        investors = item.get('investors', 'Не указаны')

        formatted_result += f"🏢 **{project}**\n"
        formatted_result += f"📅 Дата: {date}\n"
        formatted_result += f"💰 Сумма: {amount}\n"
        formatted_result += f"👥 Инвесторы: {investors}\n\n"

    if not result.get('raises'):
        formatted_result += "За указанный период привлечений средств не обнаружено.\n"

    return formatted_result

@tool
async def get_polymarket_data(days: int = 7) -> str:
    """
    Получает данные с предиктивного рынка Polymarket.

    Args:
        days: Количество дней для получения данных (по умолчанию 7)
    """
    workflow = get_llamafeed_workflow()
    since = datetime.now(timezone.utc) - timedelta(days=days)
    result = workflow.invoke(f"Get Polymarket data since {since.isoformat()}")

    # Форматируем результат
    formatted_result = "🔮 **ДАННЫЕ ПРЕДИКТИВНОГО РЫНКА POLYMARKET**\n\n"

    for item in result.get('polymarket', []):
        question = item.get('question', 'Нет вопроса')
        end_date = item.get('end_date', 'Неизвестно')
        probability = item.get('probability', 'Неизвестно')
        volume = item.get('volume', 'Неизвестно')

        formatted_result += f"❓ **{question}**\n"
        formatted_result += f"📅 Дата окончания: {end_date}\n"
        formatted_result += f"📊 Вероятность: {probability}\n"
        formatted_result += f"💹 Объем: {volume}\n\n"

    if not result.get('polymarket'):
        formatted_result += "За указанный период данных не обнаружено.\n"

    return formatted_result

@tool
async def get_market_summary(days: int = 3) -> str:
    """
    Получает комплексный обзор рынка, включая новости, твиты и важные события.

    Args:
        days: Количество дней для анализа (по умолчанию 3)
    """
    workflow = get_llamafeed_workflow()
    since = datetime.now(timezone.utc) - timedelta(days=days)

    # Собираем данные из разных источников
    result = workflow.invoke(f"Provide a comprehensive market summary since {since.isoformat()} including news, tweets, and important events")

    # Форматируем результат
    formatted_result = "📊 **КОМПЛЕКСНЫЙ ОБЗОР КРИПТОВАЛЮТНОГО РЫНКА**\n\n"

    if 'summary' in result:
        formatted_result += f"{result['summary']}\n\n"

    # Добавляем основные новости
    if 'news' in result and result['news']:
        formatted_result += "**Ключевые новости:**\n"
        for item in result['news'][:5]:
            formatted_result += f"- {item.get('title', 'Нет заголовка')}\n"
        formatted_result += "\n"

    # Добавляем основные твиты
    if 'tweets' in result and result['tweets']:
        formatted_result += "**Важные твиты:**\n"
        for item in result['tweets'][:3]:
            user = item.get('user_name', 'Аноним')
            tweet = item.get('tweet', 'Нет текста')
            formatted_result += f"- @{user}: {tweet}\n"
        formatted_result += "\n"

    # Добавляем важные события
    if 'events' in result and result['events']:
        formatted_result += "**Важные события:**\n"
        for item in result['events']:
            formatted_result += f"- {item}\n"

    return formatted_result
