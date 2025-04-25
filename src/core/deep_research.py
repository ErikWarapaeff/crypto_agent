"""Модуль для проведения глубокого исследования криптовалютных проектов."""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from rich.console import Console
from rich import print as rprint

from models.state import AgentState, Message, MessageRole
from tools import (
    # Базовые инструменты
    get_token_price,
    get_trending_coins,
    search_cryptocurrencies,
    analyze_protocol,
    analyze_pools_geckoterminal,
    get_token_historical_data,
    analyze_token_holders,

    # HyperLiquid инструменты
    get_crypto_price,
    get_klines_history,
    get_market_info,

    # LlamaFeed инструменты
    get_crypto_news,
    get_crypto_tweets,
    get_crypto_hacks,
    get_token_unlocks,
    get_project_raises,
    get_market_summary
)

class ResearchParams(BaseModel):
    """Параметры для проведения исследования."""
    token_symbol: str
    token_name: str = ""
    token_id: Optional[str] = None
    token_address: Optional[str] = None
    chain: str = "ethereum"
    days_lookback: int = 30
    risk_profile: str = "moderate"  # low, moderate, high

class ResearchResult(BaseModel):
    """Результаты исследования."""
    token_symbol: str
    token_name: str
    summary: str
    price_data: Dict[str, Any]
    technical_analysis: Dict[str, Any]
    market_data: Dict[str, Any]
    social_signals: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    recommendation: str
    full_report: str = ""  
    timestamp: datetime = datetime.now()

    class Config:
        arbitrary_types_allowed = True

class DeepResearchManager:
    """Менеджер для проведения глубокого исследования токенов."""

    def __init__(self, llm_model: str = "gpt-4"):
        """Инициализация менеджера исследований."""
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.console = Console()

    async def get_clarification_questions(self, token_symbol: str) -> List[str]:
        """Генерирует уточняющие вопросы для проведения исследования."""
        prompt = f"""
        Я хочу провести глубокое исследование криптовалюты {token_symbol}.
        Какие 3-5 уточняющих вопроса ты бы задал, чтобы лучше понять мои цели
        и интересы относительно этого токена?

        Вопросы должны помочь определить:
        1. Цель исследования (инвестиции, трейдинг, общее понимание)
        2. Временной горизонт интереса
        3. Аспекты, которые наиболее важны для меня (технология, экономика токена, команда и т.д.)
        4. Мой профиль риска

        Верни только список вопросов, каждый с новой строки, без нумерации.
        """

        response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
        questions = [q.strip() for q in response.content.split('\n') if q.strip()]
        return questions

    async def parse_user_requirements(self, conversation_history: List[Dict[str, str]]) -> ResearchParams:
        """Анализирует ответы пользователя для определения параметров исследования."""
        prompt = f"""
        На основе следующего диалога с пользователем, определи параметры для исследования
        криптовалютного токена. Извлеки следующую информацию:

        1. token_symbol: символ токена (например, BTC, ETH)
        2. token_name: полное название токена (если указано)
        3. token_id: идентификатор токена в CoinGecko (если указано)
        4. token_address: адрес смарт-контракта токена (если указано)
        5. chain: блокчейн, на котором запущен токен (по умолчанию "ethereum")
        6. days_lookback: на сколько дней назад смотреть исторические данные (по умолчанию 30)
        7. risk_profile: профиль риска пользователя ("low", "moderate", "high") (по умолчанию "moderate")

        Диалог:
        {conversation_history}

        Ответ предоставь в виде JSON без лишних комментариев.
        """

        conversation_formatted = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in conversation_history
        ])

        response = await self.llm.ainvoke([
            {"role": "user", "content": prompt.replace("{conversation_history}", conversation_formatted)}
        ])

        # Извлекаем JSON из ответа
        import json
        import re

        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            params_dict = json.loads(json_str)
            return ResearchParams(**params_dict)

        # Если не удалось извлечь JSON, используем базовые параметры
        # Извлекаем символ токена из диалога
        token_symbol = None
        for msg in conversation_history:
            if msg['role'] == 'user':
                # Ищем символы токенов в сообщении пользователя
                tokens = re.findall(r'\b[A-Z]{2,10}\b', msg['content'].upper())
                if tokens:
                    token_symbol = tokens[0]
                    break

        return ResearchParams(token_symbol=token_symbol or "BTC")

    async def gather_research_data(self, params: ResearchParams) -> Dict[str, Any]:
        """Собирает данные из всех доступных источников параллельно."""
        results = {}

        # Базовая информация
        try:
            results["basic_info"] = await search_cryptocurrencies.ainvoke(params.token_symbol)
        except Exception as e:
            rprint(f"[bold red]Ошибка при поиске базовой информации: {str(e)}[/bold red]")
            results["basic_info"] = f"Не удалось получить базовую информацию о {params.token_symbol}"

        try:
            results["price"] = await get_token_price.ainvoke(params.token_symbol)
        except Exception as e:
            rprint(f"[bold red]Ошибка при получении цены: {str(e)}[/bold red]")
            results["price"] = "Не удалось получить текущую цену"

        # Исторические данные
        try:
            results["historical_data"] = await get_token_historical_data.ainvoke({
                "token_id": params.token_id or params.token_symbol.lower(),
                "token_label": params.token_name or params.token_symbol,
                "vs_currency": "usd",
                "days": str(params.days_lookback)
            })
        except Exception as e:
            rprint(f"[bold red]Ошибка при получении исторических данных: {str(e)}[/bold red]")
            results["historical_data"] = "Не удалось получить исторические данные"

        # Новости и социальные данные - передаем словари с правильными ключами
        try:
            results["news"] = await get_crypto_news.ainvoke({"days": params.days_lookback})
        except Exception as e:
            rprint(f"[bold red]Ошибка при получении новостей: {str(e)}[/bold red]")
            results["news"] = "Не удалось получить новости"

        try:
            results["tweets"] = await get_crypto_tweets.ainvoke({"days": params.days_lookback})
        except Exception as e:
            rprint(f"[bold red]Ошибка при получении твитов: {str(e)}[/bold red]")
            results["tweets"] = "Не удалось получить твиты"

        # Трендовые монеты
        try:
            results["trending"] = await get_trending_coins.ainvoke({"limit": 10, "include_platform": True})
        except Exception as e:
            rprint(f"[bold red]Ошибка при получении трендовых монет: {str(e)}[/bold red]")
            results["trending"] = "Не удалось получить информацию о трендовых монетах"

        # Рыночный обзор
        try:
            results["market_summary"] = await get_market_summary.ainvoke({"days": params.days_lookback})
        except Exception as e:
            rprint(f"[bold red]Ошибка при получении обзора рынка: {str(e)}[/bold red]")
            results["market_summary"] = "Не удалось получить обзор рынка"

        # Дополнительные данные
        try:
            results["hacks"] = await get_crypto_hacks.ainvoke({"days": params.days_lookback})
        except Exception as e:
            rprint(f"[bold red]Ошибка при получении данных о хаках: {str(e)}[/bold red]")
            results["hacks"] = "Не удалось получить информацию о хаках"

        try:
            results["unlocks"] = await get_token_unlocks.ainvoke({"days": params.days_lookback})
        except Exception as e:
            rprint(f"[bold red]Ошибка при получении данных о разблокировках: {str(e)}[/bold red]")
            results["unlocks"] = "Не удалось получить информацию о разблокировках токенов"

        try:
            results["raises"] = await get_project_raises.ainvoke({"days": params.days_lookback})
        except Exception as e:
            rprint(f"[bold red]Ошибка при получении данных о финансировании: {str(e)}[/bold red]")
            results["raises"] = "Не удалось получить информацию о привлечении средств"

        # Если есть token_address, анализируем держателей
        if params.token_address:
            try:
                results["holders"] = await analyze_token_holders.ainvoke({
                    "token_address": params.token_address,
                    "token_label": params.token_name or params.token_symbol,
                    "chain": params.chain
                })
            except Exception as e:
                rprint(f"[bold red]Ошибка при анализе держателей: {str(e)}[/bold red]")
                results["holders"] = "Не удалось получить информацию о держателях токена"

        return results

    async def _execute_tool(self, tool_func, *args, result_key=None):
        """Выполняет инструмент и возвращает результат с ключом."""
        try:
            # Получаем имя инструмента для логирования
            tool_name = getattr(tool_func, "name", str(tool_func))

            # Правильный способ вызова LangChain инструментов
            if hasattr(tool_func, "ainvoke"):
                # Для LangChain инструментов мы должны передать аргументы по-другому
                # ainvoke ожидает один аргумент (строку) или словарь параметров
                if len(args) == 1:  # Если только один аргумент - передаем его напрямую
                    result = await tool_func.ainvoke(args[0])
                else:
                    # Создаем словарь из имен параметров функции и переданных значений
                    # Получаем имена параметров из сигнатуры функции
                    import inspect
                    sig = inspect.signature(tool_func.func)
                    params = list(sig.parameters.keys())

                    # Создаем словарь параметров
                    kwargs = {}
                    for i, arg in enumerate(args):
                        if i < len(params):
                            kwargs[params[i]] = arg

                    result = await tool_func.ainvoke(kwargs)
            elif asyncio.iscoroutinefunction(tool_func):
                # Для обычных асинхронных функций
                result = await tool_func(*args)
            else:
                # Для обычных синхронных функций
                result = tool_func(*args)

            return (result_key, result) if result_key else result
        except Exception as e:
            import traceback
            error_msg = f"Ошибка при выполнении {tool_name}: {str(e)}"
            rprint(f"[bold red]{error_msg}[/bold red]")
            # rprint(traceback.format_exc())  # Раскомментировать для отладки
            return (result_key, f"Ошибка: {str(e)}") if result_key else str(e)

    async def analyze_research_data(self, params: ResearchParams, data: Dict[str, Any]) -> ResearchResult:
        """Анализирует собранные данные и формирует результаты исследования."""
        # Подготавливаем данные для анализа LLM
        data_summary = ""

        for key, value in data.items():
            data_summary += f"=== {key.upper()} ===\n"
            # Проверяем, строка ли это
            if isinstance(value, str):
                data_summary += value + "\n\n"
            else:
                # Для других типов данных преобразуем в строку
                data_summary += str(value) + "\n\n"

        prompt = f"""
        Ты - опытный криптоаналитик. Проанализируй следующие данные о токене {params.token_symbol} и создай комплексный отчет.

        ПАРАМЕТРЫ ИССЛЕДОВАНИЯ:
        - Токен: {params.token_name or params.token_symbol} ({params.token_symbol})
        - Временной горизонт: {params.days_lookback} дней
        - Профиль риска: {params.risk_profile}

        СОБРАННЫЕ ДАННЫЕ:
        {data_summary}

        Создай структурированный отчет, включающий:

        1. РЕЗЮМЕ: Краткое резюме о токене и его текущем положении (3-4 предложения)

        2. ЦЕНОВОЙ АНАЛИЗ:
           - Текущая цена и изменение за анализируемый период
           - Ключевые уровни поддержки и сопротивления
           - Волатильность и сравнение с рынком в целом

        3. ТЕХНИЧЕСКИЙ АНАЛИЗ:
           - Тренды (краткосрочные, среднесрочные)
           - Объемы торгов и их динамика
           - Корреляция с другими активами (если данные доступны)

        4. РЫНОЧНЫЙ АНАЛИЗ:
           - Рыночная капитализация и позиция среди конкурентов
           - Ликвидность и глубина рынка
           - Присутствие на основных биржах и объемы

        5. АНАЛИЗ СОЦИАЛЬНЫХ СИГНАЛОВ:
           - Активность в социальных сетях
           - Настроения сообщества
           - Последние важные новости и их влияние

        6. ОЦЕНКА РИСКОВ:
           - Технические риски (безопасность, централизация)
           - Рыночные риски (конкуренция, ликвидность)
           - Регуляторные риски

        7. РЕКОМЕНДАЦИЯ:
           - Четкая рекомендация: ПОКУПАТЬ / ДЕРЖАТЬ / ПРОДАВАТЬ
           - Обоснование рекомендации с учетом профиля риска: {params.risk_profile}
           - Возможные сценарии развития (оптимистичный, нейтральный, пессимистичный)

        Формат: Предоставь ответ в виде хорошо структурированного Markdown с заголовками и подзаголовками.
        Включи эмодзи для улучшения читаемости.
        """

        response = await self.llm.ainvoke([{"role": "user", "content": prompt}])

        # Создаем объект результата
        result = ResearchResult(
            token_symbol=params.token_symbol,
            token_name=params.token_name or params.token_symbol,
            summary="",  # Будет заполнено из отчета
            price_data={},
            technical_analysis={},
            market_data={},
            social_signals={},
            risk_assessment={},
            recommendation="",
            timestamp=datetime.now()
        )

        # Устанавливаем полный отчет
        result.full_report = response.content

        # Извлекаем рекомендацию
        if "ПОКУПАТЬ" in response.content:
            result.recommendation = "ПОКУПАТЬ"
        elif "ПРОДАВАТЬ" in response.content:
            result.recommendation = "ПРОДАВАТЬ"
        else:
            result.recommendation = "ДЕРЖАТЬ"

        # Извлекаем резюме (первый параграф после заголовка РЕЗЮМЕ)
        import re
        summary_match = re.search(r'РЕЗЮМЕ[:\s]*\n+(.*?)(?=\n*#)', response.content, re.DOTALL)
        if summary_match:
            result.summary = summary_match.group(1).strip()

        return result

    async def conduct_research(self, user_state: AgentState, token_symbol: str) -> str:
        """
        Проводит полное исследование токена с уточняющими вопросами и анализом.

        Args:
            user_state: Текущее состояние диалога с пользователем
            token_symbol: Символ токена для исследования

        Returns:
            Строка с отчетом о результатах исследования
        """
        # Шаг 1: Генерация уточняющих вопросов
        questions = await self.get_clarification_questions(token_symbol)

        # Формируем сообщение с вопросами
        clarification_message = f"""
        Для проведения глубокого исследования токена {token_symbol.upper()}, мне нужна дополнительная информация:

        {chr(10).join(['• ' + q for q in questions])}

        Пожалуйста, ответьте на эти вопросы для более точного анализа.
        """

        # Добавляем сообщение в историю диалога
        user_state.add_assistant_message(clarification_message)

        # На данном этапе пользователь должен ответить на вопросы в интерфейсе
        # После получения ответа, мы извлекаем параметры для исследования

        # Получаем историю диалога
        conversation_history = user_state.get_conversation_history()

        # Шаг 2: Парсинг требований пользователя
        params = await self.parse_user_requirements(conversation_history)
        params.token_symbol = token_symbol  # Уточняем символ токена

        # Шаг 3: Сбор данных из всех источников
        with self.console.status(f"[bold green]Собираю данные о {token_symbol}...", spinner="dots"):
            research_data = await self.gather_research_data(params)

        # Шаг 4: Анализ данных и формирование отчета
        with self.console.status(f"[bold green]Анализирую данные и формирую отчет...", spinner="dots"):
            research_result = await self.analyze_research_data(params, research_data)

        # Возвращаем отчет
        return research_result.full_report
