"""Основной файл мультиагентной системы криптоанализа."""

import asyncio
import sys
import time
from typing import List, Dict, Any  

from rich.console import Console
from rich import print as rprint

from config.settings import setup_environment, OPENAI_API_KEY
from core.multi_flow import MultiAgentSystem, create_multi_agent_system
from ui.interface import (
    display_welcome,
    display_response,
    get_multiline_input,
    display_thinking,
    display_exit_message,   
    display_separator,
    display_research_result,
    display_task_status,
    display_agents_list
)

# Инициализация консоли
console = Console()

async def main():
    """Основная функция приложения."""
    # Настройка окружения
    setup_environment()

    # Проверка API ключа
    if not OPENAI_API_KEY:
        console.print("[bold red]ОШИБКА: OPENAI_API_KEY не установлен![/bold red]")
        return

    # Инициализация системы
    display_welcome()
    system = create_multi_agent_system()
    
    # Основной цикл
    while True:
        try:
            user_input = get_multiline_input()
            
            # Выход из программы
            if user_input.lower() in ["exit", "quit", "q"]:
                display_exit_message()
                break
                
            # Специальные команды
            if user_input.startswith('/'):
                await handle_special_commands(user_input, system)
                continue
                
            # Обработка обычного запроса
            await process_user_request(user_input, system)
            
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Операция прервана пользователем[/bold yellow]")
        except Exception as e:
            console.print(f"[bold red]Ошибка: {str(e)}[/bold red]")

async def handle_special_commands(command: str, system: Any) -> None:
    """Обрабатывает специальные команды системы."""
    if command.startswith("/research"):
        parts = command.split(maxsplit=1)
        if len(parts) < 2:
            rprint("[red]Укажите символ токена: /research BTC[/red]")
            return
            
        token = parts[1].strip().upper()
        await perform_deep_research(token, system)
        
    elif command == "/tasks":
        display_all_tasks(system)
        
    elif command == "/agents":
        display_agents_list(system.agents)
        
    elif command.startswith("/task"):
        parts = command.split()
        if len(parts) < 2:
            rprint("[red]Укажите ID задачи: /task TASK_ID[/red]")
            return
        display_task_status(system.get_task_status(parts[1]))
        
    else:
        rprint(f"[red]Неизвестная команда: {command}[/red]")

async def process_user_request(input_text: str, system: Any) -> None:
    """Обрабатывает пользовательский запрос через мультиагентную систему."""
    start_time = time.time()
    
    # Отправляем запрос супервизору
    
    with display_thinking("Супервизор планирует задачи..."):
        supervisor_initial_response = await system.process_user_input(input_text)
    
    # Параллельно выполняем все созданные задачи
    with display_thinking("Выполнение запланированных задач..."):
        tasks_result_list = await system.execute_all_pending_tasks()
    
    # Отображаем результаты выполнения задач
    successful_results = []
    failed_tasks = []
    if tasks_result_list: # Если были задачи для выполнения
        for task_result_info in tasks_result_list:
            task_id = task_result_info.get("task_id")
            status = task_result_info.get("status")
            result_content = task_result_info.get("result")

            if status == "completed" or (isinstance(result_content, str) and not result_content.startswith("[Произошла ошибка")) :
                 # Считаем успешным, если статус completed ИЛИ если result это строка без ошибки
                 # (execute_all_pending_tasks может вернуть просто строку из агента)
                 if isinstance(result_content, dict) and "error" in result_content:
                      # Если статус completed, но результат - словарь с ошибкой
                      failed_tasks.append(f"Задача {task_id}: {result_content.get('error', 'Неизвестная ошибка')}")
                 else:
                      successful_results.append(str(result_content)) # Собираем успешные результаты
            elif status and "failed" in status.lower():
                 error_detail = str(result_content.get("error", result_content) if isinstance(result_content, dict) else result_content)
                 failed_tasks.append(f"Задача {task_id}: {error_detail}")
            elif isinstance(result_content, Exception): # Если gather вернул исключение
                 failed_tasks.append(f"Задача {task_id}: Исключение - {result_content}")


        if successful_results:
            # Если есть успешные результаты, объединяем их
            final_response_to_display = "\n\n".join(successful_results)
            if failed_tasks:
                final_response_to_display += "\n\n[dim]Некоторые подзадачи не удалось выполнить:[/dim]\n" + "\n".join(f"  - {err}" for err in failed_tasks)
        elif failed_tasks:
            # Если только ошибки
            final_response_to_display = "[bold red]Не удалось выполнить запрос:[/bold red]\n" + "\n".join(f"  - {err}" for err in failed_tasks)
        else:
            # Если задачи были, но нет ни успехов, ни явных ошибок (странная ситуация)
             final_response_to_display = supervisor_initial_response # Возвращаемся к ответу супервизора
             final_response_to_display += "\n[dim](Задачи были выполнены, но результат не получен)[/dim]"

    else:
        # Если супервизор не создал ни одной задачи (например, ответил сам)
        final_response_to_display = supervisor_initial_response

    # 4. Отображаем финальный ответ
    display_response(final_response_to_display)

    # 5. Отображаем статус выполнения задач (опционально, можно убрать если мешает)
    # display_task_execution_results(tasks_result_list)

    # 6. Показываем общее время и статистику (как и было)
    total_time = time.time() - start_time
    rprint(f"[dim]Общее время обработки: {total_time:.2f} сек[/dim]")
    display_separator()
    display_system_stats(system)

async def perform_deep_research(token: str, system: Any) -> None:
    """Выполняет глубокое исследование токена через мультиагентную систему."""
    start_time = time.time()
    
    # Создаем задачи исследования
    research_tasks = [
        ("market_analyst", f"Проанализировать рыночные данные для {token}"),
        ("technical_analyst", f"Сделать технический анализ {token}"),
        ("news_researcher", f"Найти последние новости о {token}"),
        ("protocol_analyst", f"Проанализировать протоколы для {token}")
    ]
    
    # Создаем и выполняем задачи
    task_ids = []
    for agent_id, description in research_tasks:
        task_id = system.agents["supervisor"].process_user_input(
            f"Добавить задачу для {agent_id}: {description}"
        )
        task_ids.append(task_id)
    
    # Запускаем выполнение
    with display_thinking(f"Выполнение исследования {token}..."):
        results = await system.execute_all_pending_tasks()
    
    # Объединяем результаты
    merged = system.agents["supervisor"].process_user_input(
        f"Объединить результаты задач {', '.join(task_ids)}"
    )
    
    # Отображаем результаты
    display_research_result(merged, token)
    rprint(f"[dim]Исследование выполнено за {time.time()-start_time:.2f} сек[/dim]")
    display_separator()

def display_system_stats(system: Any) -> None:
    """Отображает статистику работы системы."""
    stats = {
        "Агентов": len(system.agents),
        "Всего задач": len(system.tasks),
        "Выполнено": sum(1 for t in system.tasks.values() if t.status == "completed"),
        "В процессе": sum(1 for t in system.tasks.values() if t.status == "in_progress"),
        "Ошибок": sum(1 for t in system.tasks.values() if t.status == "failed")
    }
    
    rprint("[bold cyan]Статистика системы:[/bold cyan]")
    for k, v in stats.items():
        rprint(f"  [bold]{k}:[/bold] {v}")
    display_separator()

def display_all_tasks(system: Any) -> None:
    """Отображает список всех задач системы."""
    rprint("[bold cyan]Список задач:[/bold cyan]")
    for task_id, task in system.tasks.items():
        status_color = {
            "completed": "green",
            "in_progress": "yellow",
            "failed": "red"
        }.get(task.status, "white")
        
        rprint(f"  [bold]{task_id}[/bold] - {task.title}")
        rprint(f"  Статус: [{status_color}]{task.status}[/{status_color}]")
        rprint(f"  Агент: [blue]{task.assigned_agent_id}[/blue]")
        display_separator()

def display_task_execution_results(results: List[Dict[str, Any]]) -> None:
    """Отображает результаты выполнения задач."""
    rprint("[bold cyan]Результаты выполнения задач:[/bold cyan]")
    for result in results:
        if isinstance(result.get('result'), Exception):
            rprint(f"  [red]Ошибка в задаче {result['task_id']}: {result['result']}[/red]")
        else:
            rprint(f"  [green]Задача {result['task_id']} выполнена успешно[/green]")
    display_separator()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold red]Программа завершена[/bold red]")
        sys.exit(0)
