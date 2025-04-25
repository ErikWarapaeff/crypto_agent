"""Основной файл приложения Crypto Analysis Assistant."""

import asyncio
import sys
import time

from config.settings import setup_environment
from ui.interface import (
    display_welcome,
    display_response,
    get_multiline_input,
    display_thinking,
    display_exit_message,
    display_separator
)
from core.agent import create_agent
from rich.console import Console
from rich import print as rprint
from config.settings import setup_environment, OPENAI_API_KEY

# Инициализация консоли
console = Console()

# Настройка окружения
setup_environment()

async def main():
    """Основная функция приложения."""
    
    # Проверка наличия необходимых ключей API
    if not OPENAI_API_KEY:
        console.print("[bold red]ОШИБКА: OPENAI_API_KEY не установлен![/bold red]")
        console.print("Добавьте ваш ключ API в .env файл или переменные окружения.")
        return
    # Инициализация
    display_welcome()
    agent = create_agent()
    
    # Основной цикл
    while True:
        # Получаем ввод пользователя
        user_input = get_multiline_input()
        
        # Проверяем выход
        if user_input.lower() in ["exit", "quit", "q"]:
            display_exit_message()
            break
        
        # Засекаем время обработки запроса
        start_time = time.time()
        
        # Вызываем модель
        with display_thinking():
            response = await agent.process_user_input(user_input)
        
        # Вычисляем время обработки
        processing_time = time.time() - start_time
        
        # Отображаем ответ
        display_response(response)
        
        # Показываем информацию о времени обработки
        rprint(f"[dim italic]Запрос обработан за {processing_time:.2f} сек[/dim italic]")
        
        display_separator()
        
        # Получаем состояние для отладки
        state = agent.get_state()
        rprint(f"[dim]Диалог содержит {len(state.messages)} сообщений, {len(state.tool_calls)} вызовов инструментов[/dim]")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold red]Программа прервана пользователем[/bold red]")
        sys.exit(0)