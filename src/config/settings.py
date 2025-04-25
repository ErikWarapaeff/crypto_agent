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
APP_NAME = "🚀 CRYPTO ANALYSIS AI ASSISTANT 🚀"
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