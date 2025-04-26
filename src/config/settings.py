"""Конфигурация приложения и настройки."""

import os
from dotenv import load_dotenv

# Загрузка переменных из .env файла
load_dotenv()

# API ключи
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")
BITQUERY_API_KEY = os.getenv("BITQUERY_API_KEY", "")

# Настройки LLM
LLM_MODEL = "gpt-4"
LLM_TEMPERATURE = 0

# Настройки приложения
APP_NAME = "🚀 CRYPTO AI ASSISTANT 🚀"
APP_COLOR = "cyan"

# Устанавливаем переменные окружения
def setup_environment():
    """Устанавливает необходимые переменные окружения."""
    # Если ключей нет в переменных окружения, устанавливаем их
    if not os.environ.get("OPENAI_API_KEY") and OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    
    # Проверка наличия ключей
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  ВНИМАНИЕ: OPENAI_API_KEY не установлен!")
        
        
import logging
from logging.handlers import QueueHandler, QueueListener
from logging import StreamHandler, FileHandler
from queue import Queue

# Настройка асинхронного логирования
# log_queue = Queue()
# queue_listener = QueueListener(
#     log_queue,
#     StreamHandler(),
#     FileHandler('agent_system.log', encoding='utf-8')
# )
# queue_listener.start()

# def configure_logging():
#     root_logger = logging.getLogger()
#     root_logger.setLevel(logging.INFO)
#     root_logger.addHandler(QueueHandler(log_queue))
    
#     # Форматтер с цветами для консоли
#     console_formatter = logging.Formatter(
#         '[%(asctime)s] %(levelname)-8s %(agent_id)s %(task_id)s %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S'
#     )
    
#     # Форматтер для файла
#     file_formatter = logging.Formatter(
#         '[%(asctime)s] %(levelname)-8s %(agent_id)s %(task_id)s %(message)s'
#     )
    
#     for handler in queue_listener.handlers:
#         if isinstance(handler, StreamHandler):
#             handler.setFormatter(console_formatter)
#         elif isinstance(handler, FileHandler):
#             handler.setFormatter(file_formatter)

# configure_logging()
