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
    """Получает и форматирует последние новости о криптовалютах с LlamaFeed.

    Возвращает до 10 последних новостей за указанный период с заголовком, датой, оценкой настроения и ссылкой.

    Args:
        days (int): Количество последних дней, за которые нужно получить новости (по умолчанию 3).

    Returns:
        str: Отформатированная строка с новостями или сообщение об ошибке.
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
    """Получает и форматирует важные твиты о криптовалютах с LlamaFeed.

    Возвращает до 10 последних твитов от значимых аккаунтов за указанный период с текстом, автором, датой и оценкой настроения.

    Args:
        days (int): Количество последних дней, за которые нужно получить твиты (по умолчанию 3).

    Returns:
        str: Отформатированная строка с твитами или сообщение об ошибке.
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
    """Получает и форматирует информацию о хаках и взломах в криптосфере с LlamaFeed.

    Возвращает список хаков за указанный период с названием проекта, датой, суммой ущерба, техникой взлома и ссылкой.

    Args:
        days (int): Количество последних дней для поиска информации о хаках (по умолчанию 30).

    Returns:
        str: Отформатированная строка со списком хаков или сообщение об отсутствии хаков/ошибке.
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
    """Получает и форматирует информацию о предстоящих разблокировках токенов с LlamaFeed.

    Возвращает список разблокировок за указанный период с названием проекта, датой, количеством и процентом от общего предложения.

    Args:
        days (int): Количество последних дней для поиска информации о разблокировках (по умолчанию 30).

    Returns:
        str: Отформатированная строка со списком разблокировок или сообщение об отсутствии разблокировок/ошибке.
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
    """Получает и форматирует информацию о недавних раундах финансирования криптопроектов с LlamaFeed.

    Возвращает список раундов финансирования за указанный период с названием проекта, датой, суммой и инвесторами.

    Args:
        days (int): Количество последних дней для поиска информации о финансировании (по умолчанию 30).

    Returns:
        str: Отформатированная строка со списком раундов финансирования или сообщение об отсутствии раундов/ошибке.
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
    """Получает и форматирует данные с предиктивного рынка Polymarket от LlamaFeed.

    Возвращает список активных рынков за указанный период с вопросом, датой окончания, текущей вероятностью и объемом.

    Args:
        days (int): Количество последних дней для поиска данных (по умолчанию 7).

    Returns:
        str: Отформатированная строка с данными рынков Polymarket или сообщение об отсутствии данных/ошибке.
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
    """Получает и форматирует комплексный обзор крипторынка от LlamaFeed.

    Включает текстовое summary, ключевые новости, важные твиты и события за указанный период.

    Args:
        days (int): Количество последних дней для анализа рынка (по умолчанию 3).

    Returns:
        str: Отформатированная строка с обзором рынка или сообщение об ошибке.
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
