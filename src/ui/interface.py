"""Компоненты пользовательского интерфейса."""

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text
from rich import box
from typing import List, Dict, Any  
from config.settings import APP_NAME, APP_COLOR

# Инициализация Rich консоли
console = Console()

def display_welcome():
    """Отображает приветственный экран приложения."""
    console.clear()
    title = Text(APP_NAME, style=f"bold {APP_COLOR}")
    console.print(Panel(
        title, 
        box=box.DOUBLE,
        border_style=APP_COLOR,
        padding=(1, 15)
    ))
    
    console.print("\n[bold yellow]Доступные возможности:[/bold yellow]")
    
    # Аналитические возможности
    console.print("[bold cyan]Анализ криптовалют:[/bold cyan]")
    console.print(" • Получение цен токенов")
    console.print(" • Анализ трендовых монет")
    console.print(" • Поиск информации о криптовалютах")
    console.print(" • Анализ DeFi протоколов и пулов")
    console.print(" • Исторический анализ токенов")
    console.print(" • Анализ держателей токенов")
    
    # Торговые возможности
    console.print("\n[bold cyan]Торговля (HyperLiquid):[/bold cyan]")
    console.print(" • Получение цен активов")
    console.print(" • Графики и история свечей")
    console.print(" • Выполнение торговых операций")
    console.print(" • Информация о рынках и аккаунте")
    
    # Новостные возможности
    console.print("\n[bold cyan]Новости и информация (LlamaFeed):[/bold cyan]")
    console.print(" • Последние криптоновости")
    console.print(" • Значимые твиты из криптомира")
    console.print(" • Информация о хаках и уязвимостях")
    console.print(" • Данные о разблокировках токенов")
    console.print(" • Информация о финансировании проектов")
    console.print(" • Данные Polymarket")
    console.print(" • Комплексный обзор рынка")
    
    console.print("\n[bold yellow]Специальные команды:[/bold yellow]")
    console.print(" • /research SYMBOL - Запустить глубокое исследование токена (например: /research BTC)")
    console.print(" • exit, quit, q - Выход из приложения")
    
    console.print("\n[dim](Введите команду или запрос)[/dim]\n")

def display_response(response_text):
    """Отображает ответ ассистента в красивом формате."""
    console.print(Panel(
        Markdown(response_text),
        title="🤖 [bold blue]Ответ ассистента[/bold blue]",
        title_align="left",
        border_style="blue",
        box=box.ROUNDED,
        padding=1
    ))

def get_multiline_input():
    """Обрабатывает многострочный ввод пользователя."""
    lines = []
    console.print("[bold green]Введите запрос (для завершения введите пустую строку):[/bold green]")
    
    while True:
        line = input("│ " if lines else "╭ ")
        if not line and lines:  # Пустая строка завершает ввод
            break
        lines.append(line)
    
    return "\n".join(lines)

def display_thinking(message: str = "Модель думает...") -> Console.status:
    """Показывает анимацию с кастомным сообщением."""
    return console.status(f"[bold green]{message}[/bold green]", spinner="dots")

def display_exit_message():
    """Отображает сообщение при выходе из приложения."""
    console.print("\n[bold cyan]До свидания! Спасибо за использование Crypto Analysis Assistant![/bold cyan]")

def display_separator():
    """Отображает разделитель между взаимодействиями."""
    console.print("\n" + "-" * 80 + "\n")
    
def display_research_result(result: str, token_symbol: str):
    """Отображает результаты глубокого исследования токена."""
    console.print(Panel(
        Markdown(result),
        title=f"🔬 [bold blue]Глубокое исследование {token_symbol}[/bold blue]",
        title_align="left",
        border_style="blue",
        box=box.ROUNDED,
        padding=1,
        width=100  # Фиксированная ширина для лучшего форматирования Markdown
    ))
    
    console.print("\n[dim italic]Исследование завершено. Используйте эти данные на свой страх и риск.[/dim italic]")
    display_separator()
    
    
def display_task_status(task_info: Dict[str, Any]) -> None:
    """Отображает статус конкретной задачи."""
    table = Table(box=box.ROUNDED, title="Статус задачи")
    table.add_column("Параметр", style="cyan")
    table.add_column("Значение", style="magenta")
    
    status_colors = {
        "completed": "green",
        "in_progress": "yellow",
        "failed": "red"
    }
    
    for key, value in task_info.items():
        if key == "status":
            color = status_colors.get(value, "white")
            value = f"[{color}]{value}[/{color}]"
        table.add_row(key, str(value))
    
    console.print(table)

def display_agents_list(agents: Dict[str, Any]) -> None:
    """Отображает список доступных агентов."""
    table = Table(box=box.ROUNDED, title="Список агентов")
    table.add_column("ID", style="cyan")
    table.add_column("Роль", style="magenta")
    table.add_column("Статус", style="green")
    
    for agent_id, agent in agents.items():
        table.add_row(
            agent_id,
            agent.role.value,
            "[green]Активен[/green]" if agent.state else "[red]Неактивен[/red]"
        )
    
    console.print(table)

def display_task_execution_results(results: List[Dict[str, Any]]) -> None:
    """Отображает результаты выполнения задач."""
    table = Table(box=box.ROUNDED, title="Результаты задач")
    table.add_column("ID задачи", style="cyan")
    table.add_column("Статус", style="magenta")
    table.add_column("Результат")
    
    for result in results:
        status = "[green]Успех[/green]" if not isinstance(result.get('result'), Exception) else "[red]Ошибка[/red]"
        table.add_row(
            result['task_id'],
            status,
            str(result['result'])[:100] + "..." if len(str(result['result'])) > 100 else str(result['result'])
        )
    
    console.print(table)

def display_system_stats(stats: Dict[str, Any]) -> None:
    """Отображает системную статистику."""
    table = Table(box=box.ROUNDED, title="Системная статистика")
    table.add_column("Метрика", style="cyan")
    table.add_column("Значение", style="magenta")
    
    for metric, value in stats.items():
        table.add_row(metric, str(value))
    
    console.print(table)