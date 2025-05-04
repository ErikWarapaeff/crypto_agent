import os
from pprint import pprint
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

load_dotenv()

from pawn.llamafeed_worflow.worflow import LlamaFeedWorkflow

if __name__ == "__main__":
    # Инициализация workflow
    workflow = LlamaFeedWorkflow(openai_model="gpt-4o")

    # Пример: запросить данные за последние 3 дня
    since = datetime.now(timezone.utc) - timedelta(days=3)

    queries = [
        f"Give me crypto news since {since.isoformat()}",
        f"Fetch tweets since {since.isoformat()}",
        f"What crypto hacks happened since {since.isoformat()}?",
        f"Are there any unlocks or raises since {since.isoformat()}?"
    ]

    for query in queries:
        print(f"\n Запрос: {query}")
        result = workflow.invoke(query)
        pprint(result)
