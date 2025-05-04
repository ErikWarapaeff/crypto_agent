"""Мультиагентная система для анализа криптовалют."""

import json
import time
import uuid
import logging
from typing import Literal, Dict, Any, List, Optional, Tuple, Union
import asyncio
from enum import Enum
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain.tools import tool

from config.settings import LLM_MODEL, LLM_TEMPERATURE
from models.state import AgentState, MessageRole, Message, ToolCall, ToolResult
from models.tool_schemas import ToolType
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

# Импорт инструментов
from tools import (
    get_token_price,
    get_trending_coins,
    search_cryptocurrencies,
    analyze_protocol,
    analyze_pools_geckoterminal,
    get_token_historical_data,
    analyze_token_holders,
    get_crypto_price,
    get_klines_history,
    execute_trade,
    confirm_trade,
    get_market_info,
    get_account_info,
    get_crypto_news,
    get_crypto_tweets,
    get_crypto_hacks,
    get_token_unlocks,
    get_project_raises,
    get_polymarket_data,
    get_market_summary
)


class AgentRole(str, Enum):
    """Роли агентов в системе."""
    SUPERVISOR = "supervisor"
    MARKET_ANALYST = "market_analyst"
    TECHNICAL_ANALYST = "technical_analyst"
    NEWS_RESEARCHER = "news_researcher"
    TRADER = "trader"
    PROTOCOL_ANALYST = "protocol_analyst"
    CONSULTANT = "consultant"
    CUSTOM = "custom"


class Task(BaseModel):
    """Модель задачи для агентов."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    assigned_agent_id: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    priority: int = 1
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    result: Optional[Any] = None
    parent_task_id: Optional[str] = None
    sub_tasks: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MultiAgentSystem:
    """Система управления множеством агентов для криптоанализа."""

    def __init__(self):
        """Инициализация мультиагентной системы."""
        self.agents = {}
        self.tasks = {}
        self.global_state = {}

        # Создаем все агенты
        self.initialize_all_agents()

    def initialize_all_agents(self):
        """Инициализирует всех агентов в системе."""
        # Создаем супервизорного агента
        self.create_supervisor_agent()

        # Создаем набор специализированных агентов
        self.initialize_specialized_agents()
        
        # Создаем агента-консультанта
        self.create_consultant_agent()

    def create_supervisor_agent(self):
        """Создает супервизорного агента, координирующего работу других агентов."""
        supervisor_tools = [
            self._create_delegate_task_tool(),
            self._create_check_task_status_tool(),
            self._create_merge_results_tool()
        ]

        supervisor_system_prompt = """
        Ты - супервизорный агент, координирующий работу команды специализированных агентов.
        ОБЯЗАТЕЛЬНО делегируй задачи следующим агентам в зависимости от запроса:

        1. MARKET_ANALYST - текущие цены и тренды
        2. TECHNICAL_ANALYST - исторические данные, графики, изменения цен и капитализации за период
        3. NEWS_RESEARCHER - новости и социальные сигналы
        4. PROTOCOL_ANALYST - анализ протоколов и холдеров

        КРИТИЧЕСКИ ВАЖНО: Когда запрос касается "исторических данных", "изменений за период"
        или "анализа капитализации" - ВСЕГДА назначай задачу агенту TECHNICAL_ANALYST с чёткими
        параметрами: название токена (Bitcoin/Ethereum/др.), период в днях, и что именно
        анализировать (цену/капитализацию/объем).
        """

        supervisor = CryptoAgent(
            agent_id="supervisor",
            role=AgentRole.SUPERVISOR,
            system_prompt=supervisor_system_prompt,
            tools=supervisor_tools
        )

        self.agents["supervisor"] = supervisor

    def initialize_specialized_agents(self):
        """Инициализирует набор специализированных агентов."""
        # Агент по анализу рынка
        market_analyst_tools = [
            get_token_price,
            get_trending_coins,
            search_cryptocurrencies,
            get_crypto_price
        ]

        market_analyst_prompt = """
        Ты - агент-аналитик рынка. Твоя задача - анализировать текущие цены,
        тренды и рыночные показатели криптовалют. Используй доступные инструменты
        для получения и анализа данных о ценах и трендах.
        """

        self.agents["market_analyst"] = CryptoAgent(
            agent_id="market_analyst",
            role=AgentRole.MARKET_ANALYST,
            system_prompt=market_analyst_prompt,
            tools=market_analyst_tools
        )

        # Агент технического анализа
        tech_analyst_tools = [
            get_token_historical_data,
            get_klines_history,
            get_market_info
        ]

        tech_analyst_prompt = """
        ы - агент технического анализа. Твоя задача - анализировать исторические данные,
        графики и технические индикаторы для криптовалют.

        ВАЖНО: Для получения исторических данных о токенах используй инструмент get_token_historical_data
        с правильными параметрами:
        - Для Ethereum: token_id="ethereum", token_label="Ethereum"
        - Для Bitcoin: token_id="bitcoin", token_label="Bitcoin"
        - Для других токенов: соответствующие идентификаторы

        Когда запрос касается изменения капитализации или цен за определенный период,
        всегда указывай точный период в днях в параметре days.

        Анализируй полученные данные, выделяя тренды, уровни поддержки и сопротивления,
        и предоставляй обоснованные прогнозы на основе технических индикаторов.
        """

        self.agents["technical_analyst"] = CryptoAgent(
            agent_id="technical_analyst",
            role=AgentRole.TECHNICAL_ANALYST,
            system_prompt=tech_analyst_prompt,
            tools=tech_analyst_tools
        )

        # Новостной исследователь
        news_researcher_tools = [
            get_crypto_news,
            get_crypto_tweets,
            get_crypto_hacks,
            get_token_unlocks,
            get_project_raises,
            get_polymarket_data,
            get_market_summary
        ]

        news_researcher_prompt = """
        Ты - агент-исследователь новостей. Твоя задача - собирать и анализировать
        новости, твиты и события, связанные с криптовалютами. Выделяй ключевые события,
        которые могут влиять на рынок, и оценивай их потенциальное воздействие.
        """

        self.agents["news_researcher"] = CryptoAgent(
            agent_id="news_researcher",
            role=AgentRole.NEWS_RESEARCHER,
            system_prompt=news_researcher_prompt,
            tools=news_researcher_tools
        )

        # Трейдер
        trader_tools = [
            execute_trade,
            confirm_trade,
            get_account_info
        ]

        trader_prompt = """
        Ты - агент-трейдер. Твоя задача - выполнять торговые операции на основе
        аналитических данных, предоставленных другими агентами. Учитывай риски,
        оценивай потенциальную прибыль и контролируй исполнение сделок.
        """

        self.agents["trader"] = CryptoAgent(
            agent_id="trader",
            role=AgentRole.TRADER,
            system_prompt=trader_prompt,
            tools=trader_tools
        )

        # Аналитик протоколов
        protocol_analyst_tools = [
            analyze_protocol,
            analyze_pools_geckoterminal,
            analyze_token_holders
        ]

        protocol_analyst_prompt = """
        Ты - агент-аналитик протоколов. Твоя задача - анализировать блокчейн-протоколы,
        пулы ликвидности и данные о холдерах. Выявляй риски, оценивай ликвидность
        и анализируй показатели здоровья протоколов.
        """

        self.agents["protocol_analyst"] = CryptoAgent(
            agent_id="protocol_analyst",
            role=AgentRole.PROTOCOL_ANALYST,
            system_prompt=protocol_analyst_prompt,
            tools=protocol_analyst_tools
        )

    def _create_delegate_task_tool(self):
        """Создает инструмент для делегирования задач другим агентам."""
        def delegate_task(agent_id: str, task_title: str, task_description: str, priority: int = 1) -> str:
            """
            Делегирует задачу указанному агенту.

            Args:
                agent_id: ID агента, которому делегируется задача
                task_title: Заголовок задачи
                task_description: Описание задачи
                priority: Приоритет задачи (1-5, где 5 - наивысший)

            Returns:
                ID созданной задачи
            """
            agent_id = agent_id.lower()
            if agent_id not in self.agents:
                return f"Ошибка: агент с ID {agent_id} не найден"

            task = Task(
                title=task_title,
                description=task_description,
                assigned_agent_id=agent_id,
                priority=priority
            )

            self.tasks[task.task_id] = task
            return f"Задача успешно делегирована агенту {agent_id}. ID задачи: {task.task_id}"

        return delegate_task

    def _create_check_task_status_tool(self):
        """Создает инструмент для проверки статуса задачи."""
        def check_task_status(task_id: str) -> Dict[str, Any]:
            """
            Проверяет статус указанной задачи.

            Args:
                task_id: ID задачи

            Returns:
                Словарь с информацией о задаче
            """
            if task_id not in self.tasks:
                return {"error": f"Задача с ID {task_id} не найдена"}

            task = self.tasks[task_id]
            return {
                "task_id": task.task_id,
                "title": task.title,
                "status": task.status,
                "assigned_agent_id": task.assigned_agent_id,
                "result": task.result if task.status == "completed" else None
            }

        return check_task_status

    def _create_merge_results_tool(self):
        """Создает инструмент для объединения результатов нескольких задач в структурированный отчет."""

        # Создаем LLM для формирования отчета
        report_formatter_llm = ChatOpenAI(model=LLM_MODEL, temperature=0.0, base_url="https://api.vsegpt.ru/v1")

        async def merge_results(task_ids: List[str], summary_title: str) -> Dict[str, Any]:
            """
            Объединяет результаты указанных задач в структурированный отчет.

            Args:
                task_ids: Список ID задач, результаты которых нужно объединить
                summary_title: Заголовок итогового отчета

            Returns:
                Структурированный отчет на основе результатов задач
            """
            task_results = {}
            missing_tasks = []
            incomplete_tasks = []
            tasks_info = []

            # Собираем результаты всех задач
            for task_id in task_ids:
                if task_id not in self.tasks:
                    missing_tasks.append(task_id)
                    continue

                task = self.tasks[task_id]
                tasks_info.append({
                    "task_id": task_id,
                    "title": task.title,
                    "agent_id": task.assigned_agent_id,
                    "status": task.status
                })

                if task.status != "completed":
                    incomplete_tasks.append(task_id)
                    continue

                task_results[task.title] = task.result

            # Формируем промпт для создания структурированного отчета
            report_prompt = f"""
            # Инструкция по форматированию комплексного криптоаналитического отчета

            Твоя задача - создать хорошо структурированный, комплексный аналитический отчет на основе результатов нескольких исследовательских задач.

            ## Основная структура отчета:

            Твой отчет должен ВСЕГДА содержать следующие разделы:

            1. **📋 СВОДНОЕ РЕЗЮМЕ (EXECUTIVE SUMMARY)** - 3-5 предложений с ключевыми выводами
            2. **📊 РЫНОЧНЫЙ АНАЛИЗ** - цены, объемы, капитализация, тренды
            3. **📈 ТЕХНИЧЕСКИЙ АНАЛИЗ** - паттерны, индикаторы, уровни поддержки/сопротивления
            4. **📰 НОВОСТИ И НАСТРОЕНИЯ** - ключевые новости, социальные сигналы
            5. **🔍 ФУНДАМЕНТАЛЬНЫЙ АНАЛИЗ** - технология, команда, развитие проекта
            6. **⚠️ РИСКИ И ВОЗМОЖНОСТИ** - обзор потенциальных рисков и возможностей
            7. **🔮 ПРОГНОЗ И РЕКОМЕНДАЦИИ** - обоснованное мнение о перспективах
            8. **📚 ИСТОЧНИКИ ДАННЫХ** - перечисление использованных источников

            ## Принципы форматирования:

            - Используй **жирный шрифт** для выделения важных моментов
            - Структурируй информацию с использованием заголовков ## и подзаголовков ###
            - Применяй эмодзи в начале разделов для лучшей визуальной навигации
            - Используй маркированные списки для перечисления пунктов
            - Выделяй предупреждения и важные замечания в отдельные блоки
            - Включай таблицы для сравнительного анализа, где уместно
            - Каждый вывод должен быть подкреплен данными

            ## Правила обработки данных:

            1. Объедини похожую информацию из разных источников
            2. При противоречивых данных указывай на расхождения и приводи все версии
            3. Все числовые данные должны сопровождаться единицами измерения и временными метками
            4. Все сложные термины должны быть кратко объяснены
            5. Для всех прогнозов указывай степень уверенности и временной горизонт

            ## Результаты исследовательских задач:

            {json.dumps(task_results, indent=2, ensure_ascii=False)}

            ## Задачи, которые не удалось выполнить (учти это в отчете):

            Невыполненные задачи: {incomplete_tasks}
            Отсутствующие задачи: {missing_tasks}

            ## Заголовок отчета:

            {summary_title}

            Сформируй ПОЛНЫЙ, комплексный отчет, максимально интегрируя и структурируя всю предоставленную информацию в соответствии с указанными принципами. Каждый раздел отчета должен содержать конкретную и релевантную информацию, даже если для этого нужно сделать обоснованные выводы на основе имеющихся данных.
            """

            # Генерируем структурированный отчет с помощью LLM
            try:
                report_response = await report_formatter_llm.ainvoke([{"role": "user", "content": report_prompt}])
                structured_report = report_response.content
            except Exception as e:
                # В случае ошибки просто соединяем результаты с минимальным форматированием
                structured_report = f"# {summary_title}\n\n"
                structured_report += "## Результаты выполненных задач\n\n"
                for title, result in task_results.items():
                    structured_report += f"### {title}\n\n{result}\n\n---\n\n"
                structured_report += f"\n\n⚠️ Примечание: При формировании отчета произошла ошибка: {str(e)}"

            # Возвращаем структурированный отчет и мета-информацию
            return {
                "summary_title": summary_title,
                "structured_report": structured_report,
                "raw_results": task_results,
                "tasks_info": tasks_info,
                "missing_tasks": missing_tasks,
                "incomplete_tasks": incomplete_tasks,
                "timestamp": datetime.now().isoformat()
            }

        # Создаем синхронную обертку для асинхронной функции
        def sync_merge_results(task_ids: List[str], summary_title: str) -> Dict[str, Any]:
            """Синхронная обертка для асинхронной функции merge_results."""
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Если event loop уже запущен, создаем новый
                new_loop = asyncio.new_event_loop()
                result = new_loop.run_until_complete(merge_results(task_ids, summary_title))
                new_loop.close()
            else:
                # Иначе используем существующий loop
                result = loop.run_until_complete(merge_results(task_ids, summary_title))
            return result

        return sync_merge_results

    async def create_custom_agent(self, agent_id: str, system_prompt: str, tools: List[Any]) -> str:
        """
        Создает пользовательского агента с указанными параметрами.

        Args:
            agent_id: Уникальный идентификатор агента
            system_prompt: Системный промпт для агента
            tools: Список инструментов для агента

        Returns:
            ID созданного агента
        """
        if agent_id in self.agents:
            return f"Ошибка: агент с ID {agent_id} уже существует"

        custom_agent = CryptoAgent(
            agent_id=agent_id,
            role=AgentRole.CUSTOM,
            system_prompt=system_prompt,
            tools=tools
        )

        self.agents[agent_id] = custom_agent
        return f"Агент {agent_id} успешно создан"

    async def process_user_input(self, user_input: str, agent_id: str = "supervisor") -> str:
        """
        Обрабатывает запрос пользователя с помощью указанного агента.

        Args:
            user_input: Запрос пользователя
            agent_id: ID агента для обработки запроса (по умолчанию "supervisor")

        Returns:
            Ответ агента
        """
        if agent_id not in self.agents:
            logging.error(f"Агент с ID {agent_id} не найден", 
                        extra={"agent_id": "system", "task_id": "process_input"})
            return f"Ошибка: агент с ID {agent_id} не найден"
            
        agent = self.agents[agent_id]
        
        logging.info(f"Передача запроса агенту {agent_id}", 
                    extra={"agent_id": agent_id, "task_id": "process_input"})
                    
        start_time = time.time()
        result = await agent.process_user_input(user_input)
        processing_time = time.time() - start_time
        
        logging.info(f"Запрос обработан агентом {agent_id} за {processing_time:.2f} сек", 
                    extra={"agent_id": agent_id, "task_id": "process_input", "processing_time": processing_time})
                    
        return result

    async def execute_task(self, task_id: str) -> Dict[str, Any]:
        """
        Запускает выполнение задачи указанным агентом.

        Args:
            task_id: ID задачи для выполнения

        Returns:
            Результат выполнения задачи
        """
        if task_id not in self.tasks:
            error_msg = f"Задача с ID {task_id} не найдена"
            logging.error(error_msg, extra={"agent_id": "system", "task_id": task_id})
            return {"error": error_msg}

        task = self.tasks[task_id]
        logging.info(f"Начало выполнения задачи: {task.title} (ID: {task.task_id})", 
                    extra={"agent_id": "system", "task_id": task_id})
        
        agent_id = task.assigned_agent_id

        logging.debug(f"Назначенный агент: {agent_id}", 
                     extra={"agent_id": "system", "task_id": task_id, "assigned_agent": agent_id})

        if agent_id not in self.agents:
            error_msg = f"Агент с ID {agent_id} не найден"
            logging.error(error_msg, extra={"agent_id": "system", "task_id": task_id})
            task.status = "failed"
            task.result = {"error": error_msg}
            return task.result

        agent = self.agents[agent_id]
        task.status = "in_progress"

        logging.info(f"Статус задачи изменен на: {task.status}", 
                    extra={"agent_id": agent_id, "task_id": task_id})

        try:
            start_time = time.time()
            result = await agent.process_user_input(task.description)
            execution_time = time.time() - start_time
            
            task.result = result
            task.status = "completed"
            logging.info(f"Задача выполнена успешно за {execution_time:.2f} сек: {task.title} (ID: {task.task_id})", 
                        extra={"agent_id": agent_id, "task_id": task_id, "execution_time": execution_time})
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Ошибка при выполнении задачи: {error_msg}", 
                         extra={"agent_id": agent_id, "task_id": task_id, "error": error_msg})
            task.status = "failed"
            task.result = {"error": error_msg}

        task.updated_at = datetime.now()
        return task.result

    async def execute_all_pending_tasks(self) -> List[Dict[str, Any]]:
        """
        Выполняет все ожидающие задачи параллельно.

        Returns:
            Список результатов выполнения задач
        """
        pending_tasks = [task_id for task_id, task in self.tasks.items() if task.status == "pending"]
        if not pending_tasks:
            logging.info("Нет ожидающих задач для выполнения", 
                         extra={"agent_id": "system", "task_id": "execute_tasks"})
            return []

        logging.info(f"Запуск выполнения {len(pending_tasks)} ожидающих задач", 
                    extra={"agent_id": "system", "task_id": "execute_tasks"})

        # Создаем и запускаем задачи асинхронно
        start_time = time.time()
        coroutines = [self.execute_task(task_id) for task_id in pending_tasks]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        total_time = time.time() - start_time

        # Собираем результаты выполнения
        execution_results = []
        for i, (task_id, result) in enumerate(zip(pending_tasks, results)):
            task_info = self.tasks.get(task_id)
            success = not isinstance(result, Exception) and task_info and task_info.status == "completed"
            
            if success:
                logging.info(f"Задача {task_id} выполнена успешно", 
                           extra={"agent_id": "system", "task_id": task_id})
            else:
                error_msg = str(result) if isinstance(result, Exception) else "Неизвестная ошибка"
                logging.error(f"Ошибка при выполнении задачи {task_id}: {error_msg}", 
                             extra={"agent_id": "system", "task_id": task_id, "error": error_msg})
            
            execution_results.append({
                "task_id": task_id,
                "result": result if not isinstance(result, Exception) else str(result),
                "success": success
            })

        logging.info(f"Все задачи выполнены за {total_time:.2f} сек, успешно: " 
                    f"{sum(1 for r in execution_results if r['success'])}/{len(execution_results)}", 
                    extra={"agent_id": "system", "task_id": "execute_tasks", "execution_time": total_time})

        return execution_results

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Возвращает статус и результат указанной задачи.

        Args:
            task_id: ID задачи

        Returns:
            Информация о задаче
        """
        if task_id not in self.tasks:
            return {"error": f"Задача с ID {task_id} не найдена"}

        task = self.tasks[task_id]
        return {
            "task_id": task.task_id,
            "title": task.title,
            "description": task.description,
            "status": task.status,
            "assigned_agent_id": task.assigned_agent_id,
            "priority": task.priority,
            "created_at": task.created_at.isoformat(),
            "updated_at": task.updated_at.isoformat(),
            "result": task.result
        }

    def reset_system(self) -> None:
        """Сбрасывает состояние системы, очищая задачи и состояния агентов."""
        self.tasks = {}
        self.global_state = {}

        for agent in self.agents.values():
            agent.reset_state()

    def _create_get_agent_tool_mapping_tool(self):
        """Вспомогательный генератор инструмента: get_agent_tool_mapping."""
        def get_agent_tool_mapping() -> Dict[str, List[Dict[str, str]]]:
            """
            Возвращает карту всех агентов в системе и их инструментов:
            {
              "market_analyst": [
                  {"name": "get_token_price", "description": "Получает текущую цену токена..."},
                  ...
              ],
              ...
            }
            """
            mapping = {}
            for ag_id, agent in self.agents.items():
                tools_info = []
                for fn in agent.tools:
                    # Используем только fn.name, если его нет - 'unknown_tool'
                    tool_name = getattr(fn, 'name', 'unknown_tool') 
                    # Описание берем из fn.description или fn.__doc__
                    tool_description = getattr(fn, 'description', getattr(fn, '__doc__', ''))
                    
                    tools_info.append({
                        "name": tool_name,
                        "description": (tool_description or "").strip().split("\n")[0]
                    })
                mapping[ag_id] = tools_info
            return mapping

        return tool(get_agent_tool_mapping)
    
    def _register_get_agent_tool_mapping(self):
        """Регистрирует get_agent_tool_mapping в системном списке инструментов."""
        mapper_tool = self._create_get_agent_tool_mapping_tool()
        # Сохраняем его в атрибуте, чтобы потом передать в create_consultant_agent
        self._agent_mapping_tool = mapper_tool

    @tool
    def clarify_user_goal(
        self,
        user_query: str,
        agent_mapping: Dict[str, List[Dict[str, str]]]
    ) -> List[str]:
        """
        Генерирует список уточняющих вопросов, основанных
        на том, какую информацию недостаёт для построения стратегии
        и какие инструменты/агенты есть в системе.
        
        Вход:
        - user_query: исходный запрос пользователя
        - agent_mapping: карта всех агентов и их инструментов, например
            {
            "market_analyst": [
                {"name":"get_token_price", "description":"..."},
                ...
            ],
            ...
            }
            
        Выход:
        Список вопросов (List[str]). Если всё ясно - пустой список.
        """
        # 1) Подготовим LLM
        llm = ChatOpenAI(
            model="openai/gpt-4o-mini", 
            temperature=0.0,
            base_url="https://api.vsegpt.ru/v1"
        )
        
        # 2) Системный промпт для генерации вопросов
        system_prompt = """
        Ты - ассистент, генерирующий уточняющие вопросы для составления 
        пошагового плана анализа. У тебя есть:
        1) user_query - то, что написал пользователь
        2) agent_mapping - JSON-карта всех агентов и их инструментов:
        { agent_id: [ {name, description}, ... ], ... }
        
        Твоя задача - проанализировать запрос и список доступных инструментов,
        выявить, какие параметры или детали пользователю нужно уточнить 
        (например: символ актива, горизонт анализа, тип анализа, профиль пользователя 
        и пр.), и сформулировать эти уточнения в виде массива вопросов.
        
        Требования:
        - Не хардкодь имена параметров; извлекай из названий и описаний инструментов.
        - Выводи **ТОЛЬКО** JSON-массив строк. НИКАКОГО другого текста или пояснений.
        - Если дополнительных вопросов не требуется - верни пустой JSON-массив `[]`. Не возвращай пустую строку или другой текст.
        """
        
        # 3) Соберём сообщение пользователю с картой инструментов
        user_prompt = f"""
        Задача пользователя:
        {json.dumps(user_query, ensure_ascii=False)}
        Карта агентов и инструментария:
        {json.dumps(agent_mapping, indent=2, ensure_ascii=False)}
        """
        # 4) Вызов LLM
        # Используем invoke вместо прямого вызова
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ])
        
        # 5) Парсим результат из JSON - теперь берем content из объекта ответа
        try:
            # response теперь - объект сообщения (например, AIMessage), получаем его content
            content = response.content 
            # ЛОГИРОВАНИЕ: Выводим сырой ответ перед парсингом
            print(f"DEBUG: Raw LLM response in clarify_user_goal: {content!r}") 
            questions = json.loads(content)
            if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
                return questions
        except Exception as e:
            # Логируем ошибку для отладки
            print(f"Error parsing LLM response in clarify_user_goal: {e}") 
            pass
        
        # В случае ошибки парсинга - возвращаем пустой список
        return []
    
    @tool
    def build_analysis_strategy(
        self,
        user_query: str,
        agent_mapping: Dict[str, Any],
        user_profile: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Генерирует пошаговый план анализа на основе:
        - user_query: что хочет пользователь
        - agent_mapping: карта агентов и их инструментов
        - user_profile: (опционально) профиль пользователя
        Возвращает JSON-объект:
        {
        "steps": [
            {
            "step": 1,
            "agent": "market_analyst",
            "tool": "get_token_price",
            "args": {"symbol":"ETH"},
            "reason": "оценить текущую цену актива"
            },
            ...
        ],
        "need_clarification": false
        }
        Если информации недостаточно для построения стратегии, возвращает
        {"steps": [], "need_clarification": true}
        """

        # 1) Настраиваем LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            base_url="https://api.vsegpt.ru/v1"
        )

        # 2) Системный промпт
        system_prompt = """
        Ты - эксперт по планированию анализа крипторынка. 
        У тебя есть:
        • user_query - задача пользователя;
        • agent_mapping - JSON-карта всех агентов и их инструментов;
        • user_profile - профиль пользователя (новичок/эксперт, горизонт анализа и т.п.).

        Твоя задача:
        1. Оценить, хватает ли данных (если нет - верни need_clarification = true).
        2. Если данных достаточно, составить подробный workflow:
        - each step: {step, agent, tool, args, reason}.
        3. Вернуть **чистый** JSON-объект со списком steps и полем need_clarification.
        """

        # 3) Пользовательский ввод
        user_prompt = f"""
        Задача пользователя:
        {json.dumps(user_query, ensure_ascii=False)}

        Карта агентов и инструментария:
        {json.dumps(agent_mapping, indent=2, ensure_ascii=False)}

        Профиль пользователя:
        {json.dumps(user_profile or {}, indent=2, ensure_ascii=False)}

        Построй план или укажи, каких данных не хватает.
        """

        # 4) Вызов LLM
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ])

        # 5) Парсинг результата
        try:
            content = response.content
            # ЛОГИРОВАНИЕ
            print(f"DEBUG: Raw LLM response in build_analysis_strategy: {content!r}")
            plan = json.loads(content)
            # Проверяем наличие обязательных полей
            if isinstance(plan, dict) and "steps" in plan and "need_clarification" in plan:
                return plan
        except Exception as e:
            print(f"Error parsing LLM response in build_analysis_strategy: {e}")
            pass

        # 6) Фоллбэк
        return {
            "steps": [],
            "need_clarification": True
        }
        
    @tool
    def validate_results(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Оценивает полученные результаты от агентов:
        - проверяет полноту и корректность структуры данных,
        - выделяет обнаруженные риски и пробелы,
        - формирует краткий summary и рекомендации.

        Возвращает словарь:
        {
        "status": "ok" | "incomplete" | "error",
        "summary": str,
        "issues": List[str],
        "recommendations": List[str],
        "checked_at": ISO8601 timestamp
        }
        """
        # 1) Базовые проверки
        if not isinstance(results, dict):
            return {
                "status": "error",
                "summary": "Результаты имеют неверный формат: ожидается dict.",
                "issues": ["Передан объект не является словарём."],
                "recommendations": ["Убедитесь, что результаты агентов сериализованы в dict."],
                "checked_at": datetime.utcnow().isoformat()
            }

        if not results:
            return {
                "status": "incomplete",
                "summary": "Результаты пусты.",
                "issues": ["Нет данных для анализа."],
                "recommendations": ["Проверьте, что все шаги стратегии были выполнены."],
                "checked_at": datetime.utcnow().isoformat()
            }

        # 2) Подготовка LLM для глубокой оценки
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            base_url="https://api.vsegpt.ru/v1"
        )

        system_prompt = """
        Ты - помощник по валидации результатов анализа крипторынка.
        Твоя задача:
        1. Оценить полноту и корректность данных в переданном JSON.
        2. Выявить возможные риски, пробелы или анормальные значения.
        3. Сформулировать короткое summary и дать рекомендации по доработке.
        Отвечай чистым JSON с полями:
        {
        "status": "ok" | "incomplete" | "error",
        "summary": str,
        "issues": [str, ...],
        "recommendations": [str, ...]
        }
        """

        user_prompt = f"""
        Результаты анализа (raw JSON):
        {json.dumps(results, ensure_ascii=False, indent=2)}
        """

        try:
            # 3) Вызов LLM
            response = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            content = response.content.strip()
            # ЛОГИРОВАНИЕ
            print(f"DEBUG: Raw LLM response in validate_results: {content!r}")

            # 4) Парсинг JSON-ответа LLM
            report = json.loads(content)
            # Добавляем timestamp
            report["checked_at"] = datetime.utcnow().isoformat()
            # Проверка на обязательные поля
            for field in ("status", "summary", "issues", "recommendations"):
                if field not in report:
                    raise ValueError(f"Поле {field!r} отсутствует в ответе LLM.")
            return report

        except Exception as e:
            print(f"Error parsing LLM response in validate_results: {e}")
            # 5) Фоллбэк при ошибках
            # Простая проверка на пропущенные ключи или null-значения
            issues = []
            for k, v in results.items():
                if v is None or (isinstance(v, (list, dict)) and not v):
                    issues.append(f"Пустой или null результат в ключе '{k}'.")
            status = "incomplete" if issues else "ok"
            return {
                "status": status,
                "summary": "Автоматическая проверка результатов.",
                "issues": issues,
                "recommendations": (
                    ["Проверьте логи агентов, устраните пустые значения."] 
                    if issues else []
                ),
                "checked_at": datetime.utcnow().isoformat(),
                "error": str(e)
            }

    @tool
    def fallback_strategy(
        self,
        failed_steps: List[Dict[str, Any]],
        original_strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Если какие-то шаги не удалось выполнить, предлагает альтернативный план
        или корректировку на основе оригинальной стратегии и списка упавших шагов.

        Вход:
        failed_steps: [
            {"step": int, "agent": str, "tool": str, "args": {...}, "reason": str, "error": str},
            ...
        ]
        original_strategy: {
            "steps": [...],           # полный список шагов из build_analysis_strategy
            "need_clarification": bool
        }

        Выход:
        {
            "alternative_steps": [
            {"step": int, "agent": str, "tool": str, "args": {...}, "reason": str},
            ...
            ],
            "note": str,               # краткое пояснение, что изменилось
            "generated_at": ISO8601 timestamp
        }
        """
        # Если нет упавших шагов - fallback не нужен
        if not failed_steps:
            return {
                "alternative_steps": original_strategy.get("steps", []),
                "note": "Не обнаружено сбойных шагов - возвращён оригинальный план.",
                "generated_at": datetime.utcnow().isoformat()
            }

        # Подготовка LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            base_url="https://api.vsegpt.ru/v1"
        )

        system_prompt = """
        Ты - ассистент по корректировке плана анализа крипторынка.
        У тебя на входе:
        1) original_strategy - исходный план: {steps: [...], need_clarification: bool}
        2) failed_steps - список шагов, которые не получилось выполнить, 
        с описанием ошибки в поле "error".
        Твоя задача - предложить альтернативный план:
        - для каждого упавшего шага подбери другой инструмент или другого агента,
        или измени аргументы, чтобы обойти проблему.
        - не удаляй нужные пользователю этапы анализа, а только перенастрой их.
        - сохрани формат каждого шага: {step, agent, tool, args, reason}.
        Отвечай **чистым** JSON-объектом:
        {
        "alternative_steps": [ ... ],
        "note": "<что изменилось и почему>",
        }
        """

        user_prompt = f"""
        original_strategy:
        {json.dumps(original_strategy, ensure_ascii=False, indent=2)}

        failed_steps:
        {json.dumps(failed_steps, ensure_ascii=False, indent=2)}
        """

        try:
            resp = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ])
            content = resp.content.strip()
            # ЛОГИРОВАНИЕ
            print(f"DEBUG: Raw LLM response in fallback_strategy: {content!r}")
            plan = json.loads(content)

            # Проверка структуры
            if not isinstance(plan, dict) or "alternative_steps" not in plan:
                raise ValueError("В ответе нет ключа 'alternative_steps'")
            # Добавляем timestamp
            plan["generated_at"] = datetime.utcnow().isoformat()
            return plan

        except Exception as e:
            print(f"Error parsing LLM response in fallback_strategy: {e}")
            # Фоллбэк: простой вариант - пропускаем упавшие шаги и смещаем номера
            orig_steps = original_strategy.get("steps", [])
            ok_steps = [s for s in orig_steps if s.get("step") not in {f["step"] for f in failed_steps}]
            # перенумеруем
            for idx, s in enumerate(ok_steps, start=1):
                s["step"] = idx

            return {
                "alternative_steps": ok_steps,
                "note": (
                    "Не удалось сгенерировать расширенный fallback через LLM. "
                    f"Удалены упавшие шаги: {[f['step'] for f in failed_steps]}. "
                    "Оставшиеся шаги перенумерованы."
                ),
                "generated_at": datetime.utcnow().isoformat(),
                "error": str(e)
            }

    def create_consultant_agent(self):
        """
        Создаёт агента-консультанта, который:
        1) берет у системы карту агентов и их инструментов,
        2) задает уточняющие вопросы (если нужно),
        3) строит пошаговую стратегию анализа,
        4) при необходимости валидирует результаты и
        5) предлагает запасной план при сбоях.
        """
        # Регистрируем инструмент для получения метаданных по всем агентам
        self._register_get_agent_tool_mapping()

        # Собираем все инструменты, которые будет использовать консультант
        consultant_tools = [
            self._agent_mapping_tool,
            self.clarify_user_goal,
            self.build_analysis_strategy,
            self.validate_results,
            self.fallback_strategy,
            self.execute_strategy
        ]

        # Очень подробный system_prompt - диктует логику работы консультанта
        consultant_system_prompt = """
        Ты - AI-консультант по криптоанализу. Твоя задача - максимально автоматизировать
        и структурировать работу с пользователем:

        1) Получи карту агентов и их инструментов:
            mapping = get_agent_tool_mapping()
        2) Оцени, хватает ли входных данных:
            questions = clarify_user_goal(user_query, mapping)
        • Если questions не пуст, верни их пользователю и дождись ответов.
        3) Когда все уточнения получены, вызови:
            strategy = build_analysis_strategy(user_query, mapping, user_profile)
        4) Отправь план в текстовом виде пользователю для проверки.
        5) После одобрения пользователем (сообщения типа "выполни план", "запусти", "согласен"), 
            переходи к выполнению стратегии через супервизора:
            results = execute_strategy(strategy)
        6) Если какой-то шаг упал, вызови:
            alternative = fallback_strategy(results["failed_steps"], strategy)
            и спроси у пользователя, выполнить ли этот альтернативный план.
        7) При необходимости валидируй полученные результаты через:
            validation = validate_results(results["results"])

        При любом выводе:
        - Чётко подписывай, какой инструмент ты вызываешь и с какими аргументами.
        - Вся промежуточная логика должна жить в system_prompt, а не в теле кода.
        - Возвращай пользователю план в формате markdown без лишних пояснений.
        - Понимай русский язык и всегда отвечай на русском языке.
        """

        # Регистрируем агента
        self.agents["consultant"] = CryptoAgent(
            agent_id="consultant",
            role=AgentRole.CONSULTANT,
            system_prompt=consultant_system_prompt,
            tools=consultant_tools
        )
        
        logging.info("Агент-консультант успешно создан и добавлен в систему", 
                    extra={"agent_id": "system", "task_id": "init"})
        
    @tool
    def execute_strategy(
        self,
        strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Выполняет стратегию анализа через супервизорного агента.
        
        Вход:
        strategy: {
            "steps": [
                {"step": int, "agent": str, "tool": str, "args": {...}, "reason": str},
                ...
            ],
            "need_clarification": bool
        }
        
        Выход:
        {
            "success": bool,
            "results": Dict[str, Any],  # результаты выполнения задач
            "failed_steps": List[Dict[str, Any]],  # шаги, которые не удалось выполнить
            "completed_at": ISO8601 timestamp
        }
        """
        logging.info("Начало выполнения стратегии", 
                    extra={"agent_id": "consultant", "task_id": "execute_strategy", "steps_count": len(strategy.get("steps", []))})
        
        if not strategy or "steps" not in strategy or not strategy["steps"]:
            logging.warning("Попытка выполнить пустую стратегию", 
                          extra={"agent_id": "consultant", "task_id": "execute_strategy"})
            return {
                "success": False,
                "results": {},
                "failed_steps": [],
                "completed_at": datetime.utcnow().isoformat(),
                "error": "Стратегия пуста или не содержит шагов."
            }
            
        # Преобразуем стратегию в структурированный запрос для супервизора
        steps = strategy["steps"]
        
        # Создаем задачи для каждого шага
        created_tasks = []
        for step in steps:
            agent_id = step["agent"].lower()
            tool_name = step["tool"]
            args = step.get("args", {})
            reason = step.get("reason", "")
            
            logging.info(f"Создание задачи для шага {step['step']}: {reason}", 
                        extra={"agent_id": "consultant", "task_id": "execute_strategy", 
                              "step": step['step'], "target_agent": agent_id, "tool": tool_name})
            
            # Формируем описание задачи
            task_description = f"Выполни инструмент {tool_name} с параметрами {json.dumps(args, ensure_ascii=False)}. Цель: {reason}"
            task_title = f"Шаг {step['step']}: {reason}"
            
            # Создаем задачу через метод супервизора
            task_id = self._create_task(agent_id, task_title, task_description)
            if isinstance(task_id, str) and task_id.startswith("Задача успешно"):
                task_id = task_id.split("ID задачи: ")[-1].strip()
                created_tasks.append({"step": step, "task_id": task_id})
                logging.info(f"Задача для шага {step['step']} успешно создана с ID: {task_id}", 
                            extra={"agent_id": "consultant", "task_id": "execute_strategy", 
                                  "step": step['step'], "created_task_id": task_id})
            else:
                # Если не удалось создать задачу
                error_msg = f"Не удалось создать задачу для шага {step['step']}: {task_id}"
                logging.error(error_msg, 
                            extra={"agent_id": "consultant", "task_id": "execute_strategy", 
                                  "step": step['step'], "target_agent": agent_id, "error": task_id})
                return {
                    "success": False,
                    "results": {},
                    "failed_steps": [step],
                    "completed_at": datetime.utcnow().isoformat(),
                    "error": error_msg
                }
        
        # Запускаем выполнение всех задач асинхронно
        logging.info(f"Запуск выполнения {len(created_tasks)} задач", 
                    extra={"agent_id": "consultant", "task_id": "execute_strategy"})
        
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Если event loop уже запущен, создаем новый
            new_loop = asyncio.new_event_loop()
            tasks_results = new_loop.run_until_complete(self.execute_all_pending_tasks())
            new_loop.close()
        else:
            # Иначе используем существующий loop
            tasks_results = loop.run_until_complete(self.execute_all_pending_tasks())
        
        logging.info(f"Все задачи выполнены. Обработка результатов", 
                    extra={"agent_id": "consultant", "task_id": "execute_strategy"})
            
        # Собираем результаты и неудачные шаги
        results = {}
        failed_steps = []
        
        for task_info in created_tasks:
            step = task_info["step"]
            task_id = task_info["task_id"]
            task_status = self.get_task_status(task_id)
            
            if task_status.get("status") == "completed":
                results[f"step_{step['step']}"] = {
                    "step": step["step"],
                    "agent": step["agent"],
                    "tool": step["tool"], 
                    "result": task_status.get("result")
                }
                logging.info(f"Шаг {step['step']} выполнен успешно", 
                            extra={"agent_id": "consultant", "task_id": "execute_strategy", 
                                  "step": step['step'], "task_id": task_id})
            else:
                step_with_error = step.copy()
                error_msg = f"Задача не выполнена: {task_status.get('status')}"
                step_with_error["error"] = error_msg
                failed_steps.append(step_with_error)
                logging.warning(f"Шаг {step['step']} не выполнен: {task_status.get('status')}", 
                              extra={"agent_id": "consultant", "task_id": "execute_strategy", 
                                   "step": step['step'], "task_id": task_id, "task_status": task_status.get('status')})
                
        result = {
            "success": len(failed_steps) == 0,
            "results": results,
            "failed_steps": failed_steps,
            "completed_at": datetime.utcnow().isoformat()
        }
        
        logging.info(f"Стратегия выполнена с результатом: success={result['success']}, " 
                    f"completed_steps={len(results)}, failed_steps={len(failed_steps)}", 
                    extra={"agent_id": "consultant", "task_id": "execute_strategy"})
                    
        return result
        
    def _create_task(self, agent_id: str, task_title: str, task_description: str, priority: int = 1) -> str:
        """
        Вспомогательный метод для создания задачи через функцию delegate_task.
        
        Возвращает ID созданной задачи или сообщение об ошибке.
        """
        logging.debug(f"Создание задачи для агента {agent_id}: {task_title}", 
                     extra={"agent_id": "system", "task_id": "create_task", 
                           "target_agent": agent_id, "title": task_title})
                     
        delegate_task = self._create_delegate_task_tool()
        result = delegate_task(agent_id, task_title, task_description, priority)
        
        if isinstance(result, str) and result.startswith("Задача успешно"):
            task_id = result.split("ID задачи: ")[-1].strip()
            logging.info(f"Задача успешно создана с ID: {task_id}", 
                        extra={"agent_id": "system", "task_id": "create_task", 
                              "target_agent": agent_id, "created_task_id": task_id})
        else:
            logging.error(f"Ошибка при создании задачи: {result}", 
                         extra={"agent_id": "system", "task_id": "create_task", 
                               "target_agent": agent_id, "error": result})
                               
        return result


class CryptoAgent:
    """Класс агента для анализа криптовалют с использованием LLM и инструментов."""

    def __init__(self, agent_id: str, role: AgentRole, system_prompt: str, tools: List[Any]):
        """
        Инициализация агента и его компонентов.

        Args:
            agent_id: Уникальный идентификатор агента
            role: Роль агента в системе
            system_prompt: Системный промпт для агента
            tools: Список инструментов агента
        """
        self.agent_id = agent_id
        self.role = role
        self.system_prompt = system_prompt
        self.state = AgentState()
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, base_url="https://api.vsegpt.ru/v1")
        self.tools = tools

        # Добавляем системное сообщение с промптом
        self.state.add_system_message(system_prompt)

        # Привязка инструментов к модели
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Создание графа агента
        self.agent = self._create_agent_graph()
        
        logging.info(f"Инициализирован агент {self.agent_id} с ролью {self.role} и {len(tools)} инструментами",
                    extra={"agent_id": self.agent_id, "role": self.role.value})

    def _create_agent_graph(self):
        """Создает и возвращает граф агента с инструментами."""
        # Создание ToolNode с инструментами
        tool_node = ToolNode(self.tools)

        # Создание графа состояния
        workflow = StateGraph(MessagesState)

        # Добавление узлов и ребер
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", tool_node)

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"tools": "tools", "end": END}
        )
        workflow.add_edge("tools", "agent")

        # Компиляция графа
        return workflow.compile()

    async def _call_model(self, state: MessagesState):
        """Вызывает модель с текущими сообщениями."""
        messages = state["messages"]

        # Засекаем время выполнения
        start_time = time.time()

        # Вызываем модель
        response = await self.llm_with_tools.ainvoke(messages)

        # Записываем время выполнения
        execution_time = time.time() - start_time
        
        logging.info(f"Вызов LLM агентом {self.agent_id} выполнен за {execution_time:.2f} сек",
                    extra={"agent_id": self.agent_id, "execution_time": execution_time})

        # Обновляем состояние агента
        if hasattr(response, "content") and response.content:
            self.state.add_assistant_message(response.content)
            logging.debug(f"Агент {self.agent_id} добавил сообщение в историю",
                        extra={"agent_id": self.agent_id, "message_len": len(response.content)})

        # Обрабатываем вызовы инструментов, если они есть
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name", "unknown_tool")
                tool_args = tool_call.get("args", {})

                logging.info(f"Агент {self.agent_id} вызывает инструмент {tool_name}",
                            extra={"agent_id": self.agent_id, "tool": tool_name, "args": json.dumps(tool_args)})

                # Регистрируем вызов инструмента
                self.state.add_tool_call(tool_name, tool_args)

        return {"messages": [response]}

    def _should_continue(self, state: MessagesState) -> Literal["tools", "end"]:
        """Определяет, нужно ли вызывать инструменты или завершить обработку."""
        messages = state["messages"]
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            logging.debug(f"Агент {self.agent_id} продолжает выполнение с инструментами",
                        extra={"agent_id": self.agent_id})
            return "tools"
        logging.debug(f"Агент {self.agent_id} завершает выполнение",
                    extra={"agent_id": self.agent_id})
        return "end"

    async def process_user_input(self, user_input: str) -> str:
        """
        Обрабатывает ввод пользователя и возвращает ответ.

        Args:
            user_input: Запрос пользователя

        Returns:
            Ответ агента
        """
        # Добавляем сообщение пользователя в состояние
        self.state.add_user_message(user_input)
        logging.info(f"Агент {self.agent_id} получил запрос пользователя",
                    extra={"agent_id": self.agent_id, "input_len": len(user_input)})

        # Преобразуем историю в формат для LangChain
        langchain_messages = self.state.get_conversation_history()

        # Вызываем агента
        start_time = time.time()
        result = await self.agent.ainvoke({"messages": langchain_messages})
        processing_time = time.time() - start_time
        
        logging.info(f"Агент {self.agent_id} обработал запрос за {processing_time:.2f} сек",
                    extra={"agent_id": self.agent_id, "processing_time": processing_time})

        # Получаем последний ответ
        last_message = result["messages"][-1]
        response_content = last_message.content if hasattr(last_message, "content") else str(last_message)
        
        logging.info(f"Агент {self.agent_id} сформировал ответ",
                    extra={"agent_id": self.agent_id, "response_len": len(response_content)})

        return response_content

    def get_state(self) -> AgentState:
        """Возвращает текущее состояние агента."""
        return self.state

    def reset_state(self) -> None:
        """Сбрасывает состояние агента."""
        self.state = AgentState()
        # Добавляем системное сообщение с промптом
        self.state.add_system_message(self.system_prompt)


# Функция для создания мультиагентной системы
def create_multi_agent_system() -> MultiAgentSystem:
    """Создает и возвращает мультиагентную систему для анализа криптовалют."""
    return MultiAgentSystem()


# Пример использования с консультантом
async def example_usage_with_consultant():
    """Пример использования мультиагентной системы с агентом-консультантом."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)-8s [%(agent_id)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    system = create_multi_agent_system()

    # Обработка запроса пользователя агентом-консультантом
    user_query = "Проанализируй как изменилась стоимость Ethereum за последний месяц и какие новости могли повлиять на его цену"
    logging.info("Запрос пользователя: %s", user_query, 
                extra={"agent_id": "system", "task_id": "example"})
    
    # Используем агента-консультанта вместо супервизора
    response = await system.process_user_input(user_query, agent_id="consultant")

    logging.info("Ответ консультанта получен", 
                extra={"agent_id": "system", "task_id": "example"})
    
    print("\nОтвет консультанта:")
    print("-" * 50)
    print(response)
    print("-" * 50)

    return response


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage_with_consultant())
