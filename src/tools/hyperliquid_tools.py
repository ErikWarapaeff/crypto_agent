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
    """
    Получает текущую цену криптовалюты на HyperLiquid.
    
    Args:
        symbol: Символ актива (например, BTC, ETH, HYPE)
    """
    workflow = get_hyperliquid_workflow()
    result = workflow.invoke(f"What is {symbol} price now?")
    return str(result)

@tool
async def get_klines_history(symbol: str, days: int = 7) -> str:
    """
    Получает историю свечей (klines) для указанного актива.
    
    Args:
        symbol: Символ актива (например, BTC, ETH, HYPE)
        days: Количество дней истории (по умолчанию 7)
    """
    workflow = get_hyperliquid_workflow()
    result = workflow.invoke(f"Send me {symbol} klines history for last {days} days?")
    return str(result)

@tool
async def execute_trade(symbol: str, amount: float, side: str = "buy") -> str:
    """
    Выполняет торговую операцию на HyperLiquid.
    
    Args:
        symbol: Символ актива (например, BTC, ETH, HYPE)
        amount: Количество для торговли
        side: Сторона сделки ('buy' или 'sell', по умолчанию 'buy')
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
    """
    Подтверждает и выполняет торговую операцию на HyperLiquid.
    
    Args:
        symbol: Символ актива (например, BTC, ETH, HYPE)
        amount: Количество для торговли
        side: Сторона сделки ('buy' или 'sell', по умолчанию 'buy')
    """
    workflow = get_hyperliquid_workflow()
    request = f"Make a trade for {amount} {symbol} {side}"
    result = workflow.invoke(request)
    return f"Операция выполнена: {side} {amount} {symbol}\nРезультат: {result}"

@tool
async def get_market_info(symbol: str) -> str:
    """
    Получает информацию о рынке для указанного актива на HyperLiquid.
    
    Args:
        symbol: Символ актива (например, BTC, ETH, HYPE)
    """
    workflow = get_hyperliquid_workflow()
    result = workflow.invoke(f"Get market info for {symbol}")
    return str(result)

@tool
async def get_account_info() -> str:
    """
    Получает информацию о текущем аккаунте на HyperLiquid.
    """
    workflow = get_hyperliquid_workflow()
    result = workflow.invoke("Get my account information")
    return str(result)