"""Инструменты для работы с HyperLiquid"""

import asyncio
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
from pawn.hyperliquid_trader_worflow import HyperliquidWorkflow

# Создаем синглтон экземпляра HyperliquidWorkflow для переиспользования
_workflow_instance = None

def get_hyperliquid_workflow():
    """Возвращает экземпляр HyperliquidWorkflow (синглтон)"""
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = HyperliquidWorkflow()
    return _workflow_instance

@tool
async def get_crypto_price(symbol: str) -> str:
    """Получает текущую цену указанного актива с биржи HyperLiquid.
    
    Args:
        symbol (str): Символ актива (например, 'BTC', 'ETH', 'HYPE').
    
    Returns:
        str: Строка с текущей ценой актива или результат вызова workflow.
    """
    workflow = get_hyperliquid_workflow()
    result = workflow.invoke(f"What is {symbol} price now?")
    return str(result)

@tool
async def get_klines_history(symbol: str, days: int = 7) -> str:
    """Получает историю свечей (OHLCV данные) для указанного актива с HyperLiquid.

    Возвращает данные за указанное количество последних дней.
    
    Args:
        symbol (str): Символ актива (например, 'BTC', 'ETH', 'HYPE').
        days (int): Количество последних дней, за которые нужна история (по умолчанию 7).

    Returns:
        str: Строка с историей свечей или результат вызова workflow.
    """
    workflow = get_hyperliquid_workflow()
    result = workflow.invoke(f"Send me {symbol} klines history for last {days} days?")
    return str(result)

@tool
async def execute_trade(symbol: str, amount: float, side: str = "buy") -> str:
    """Инициирует запрос на выполнение торговой операции на HyperLiquid.
    
    ВАЖНО: Этот инструмент НЕ выполняет сделку сразу. Он возвращает
    сообщение с деталями запроса и требованием явного подтверждения
    с помощью инструмента `confirm_trade`.
    
    Args:
        symbol (str): Символ актива для торговли (например, 'BTC', 'ETH', 'HYPE').
        amount (float): Количество актива для покупки или продажи. Должно быть положительным.
        side (str): Сторона сделки: 'buy' (купить) или 'sell' (продать). По умолчанию 'buy'.

    Returns:
        str: Строка с запросом на подтверждение операции или сообщение об ошибке валидации.
    """
    # Добавляем проверки безопасности
    if side.lower() not in ["buy", "sell"]:
        return f"Ошибка: недопустимая сторона сделки '{side}'. Допустимые значения: 'buy' или 'sell'."
    
    if amount <= 0:
        return f"Ошибка: количество должно быть положительным числом."
    
    # Запрашиваем подтверждение через ответ агента
    return (
        f"⚠️ ЗАПРОС НА ТОРГОВУЮ ОПЕРАЦИЮ ⚠️\n\n"
        f"Получен запрос на {side} {amount} {symbol}.\n\n"
        f"Это действие требует дополнительного подтверждения. "
        f"Пожалуйста, подтвердите операцию, "
        f"явно указав 'Подтверждаю торговую операцию {side} {amount} {symbol}'.\n\n"
        f"⚠️ Торговые операции связаны с финансовыми рисками. ⚠️"
    )

@tool
async def confirm_trade(symbol: str, amount: float, side: str = "buy") -> str:
    """Подтверждает и выполняет ранее запрошенную торговую операцию на HyperLiquid.

    Этот инструмент следует вызывать ПОСЛЕ того, как `execute_trade` вернул запрос
    на подтверждение, и пользователь явно подтвердил намерение.
    
    Args:
        symbol (str): Символ актива (например, 'BTC', 'ETH', 'HYPE'). Должен совпадать с запросом `execute_trade`.
        amount (float): Количество актива. Должно совпадать с запросом `execute_trade`.
        side (str): Сторона сделки ('buy' или 'sell'). Должна совпадать с запросом `execute_trade`.

    Returns:
        str: Строка с результатом выполнения торговой операции от HyperLiquid.
    """
    workflow = get_hyperliquid_workflow()
    request = f"Make a trade for {amount} {symbol} {side}"
    result = workflow.invoke(request)
    return f"Операция выполнена: {side} {amount} {symbol}\nРезультат: {result}"

@tool
async def get_market_info(symbol: str) -> str:
    """Получает общую рыночную информацию для указанного актива с HyperLiquid.

    Может включать данные о текущей цене, объеме, ставке финансирования и т.д.
    
    Args:
        symbol (str): Символ актива (например, 'BTC', 'ETH', 'HYPE').

    Returns:
        str: Строка с рыночной информацией или результат вызова workflow.
    """
    workflow = get_hyperliquid_workflow()
    result = workflow.invoke(f"Get market info for {symbol}")
    return str(result)

@tool
async def get_account_info() -> str:
    """Получает информацию о текущем состоянии торгового аккаунта на HyperLiquid.

    Включает данные о балансах, позициях, марже и т.д.

    Returns:
        str: Строка с информацией об аккаунте или результат вызова workflow.
    """
    workflow = get_hyperliquid_workflow()
    result = workflow.invoke("Get my account information")
    return str(result)