"""Компоненты пользовательского интерфейса."""

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text
from rich import box

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
    console.print(" • Получение цен криптовалют")
    console.print(" • Анализ трендовых монет")
    console.print(" • Поиск информации о криптовалютах")
    console.print(" • Анализ DeFi протоколов и пулов")
    console.print(" • Исторический анализ токенов")
    console.print(" • Анализ держателей токенов")
    console.print("\n[dim](Введите 'exit' или 'quit' для выхода)[/dim]\n")

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

def display_thinking():
    """Показывает анимацию пока модель думает."""
    return console.status("[bold green]Модель думает...", spinner="dots")

def display_exit_message():
    """Отображает сообщение при выходе из приложения."""
    console.print("\n[bold cyan]До свидания! Спасибо за использование Crypto Analysis Assistant![/bold cyan]")

def display_separator():
    """Отображает разделитель между взаимодействиями."""
    console.print("\n" + "-" * 80 + "\n")