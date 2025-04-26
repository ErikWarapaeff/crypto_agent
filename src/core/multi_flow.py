"""Мультиагентная система для анализа криптовалют."""

import time
import uuid
from typing import Literal, Dict, Any, List, Optional, Tuple, Union
import asyncio
from enum import Enum
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

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
        
        # Создаем супервизорного агента
        self.create_supervisor_agent()
        
        # Создаем набор специализированных агентов
        self.initialize_specialized_agents()
    
    def create_supervisor_agent(self):
        """Создает супервизорного агента, координирующего работу других агентов."""
        supervisor_tools = [
            self._create_delegate_task_tool(),
            self._create_check_task_status_tool(),
            self._create_merge_results_tool()
        ]
        
        supervisor_system_prompt = """
        Ты - супервизорный агент, координирующий работу команды специализированных агентов.
        Твоя задача - анализировать запросы пользователя, разбивать их на подзадачи 
        и делегировать их соответствующим агентам:
        
        1. MARKET_ANALYST - анализирует рыночные данные, цены и тренды
        2. TECHNICAL_ANALYST - проводит технический анализ графиков и исторических данных
        3. NEWS_RESEARCHER - исследует новости, твиты и события
        4. TRADER - выполняет торговые операции и анализирует их
        5. PROTOCOL_ANALYST - анализирует протоколы, пулы ликвидности и холдеры
        
        Используй инструменты для делегирования задач, проверки их статуса и объединения результатов.
        Используй инструменты для делегирования задач, проверки их статуса и объединения результатов.

        **ВАЖНО:** Прежде чем использовать инструмент `delegate_task`, убедись, что ты не делегировал **точно такую же** задачу этому же агенту **в рамках текущего ответа**. 
        Если задача уже была успешно делегирована (инструмент `delegate_task` вернул ID задачи), не вызывай `delegate_task` для неё повторно. 
        Ты можешь использовать `check_task_status`, чтобы узнать статус уже делегированной задачи, если это необходимо.
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
        Ты - агент технического анализа. Твоя задача - анализировать исторические данные,
        графики и технические индикаторы. На основе этих данных делай прогнозы
        и выявляй паттерны в движении цен.
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
        """Создает инструмент для объединения результатов нескольких задач."""
        def merge_results(task_ids: List[str], summary_title: str) -> Dict[str, Any]:
            """
            Объединяет результаты указанных задач.
            
            Args:
                task_ids: Список ID задач, результаты которых нужно объединить
                summary_title: Заголовок итогового отчета
            
            Returns:
                Объединенные результаты задач
            """
            results = {}
            missing_tasks = []
            incomplete_tasks = []
            
            for task_id in task_ids:
                if task_id not in self.tasks:
                    missing_tasks.append(task_id)
                    continue
                    
                task = self.tasks[task_id]
                if task.status != "completed":
                    incomplete_tasks.append(task_id)
                    continue
                    
                results[task.title] = task.result
            
            return {
                "summary_title": summary_title,
                "results": results,
                "missing_tasks": missing_tasks,
                "incomplete_tasks": incomplete_tasks,
                "timestamp": datetime.now().isoformat()
            }
        
        return merge_results
    
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
    
    async def process_user_input(self, user_input: str) -> str:
        """
        Обрабатывает запрос пользователя с помощью супервизорного агента.
        
        Args:
            user_input: Запрос пользователя
            
        Returns:
            Ответ супервизорного агента
        """
        supervisor = self.agents["supervisor"]
        return await supervisor.process_user_input(user_input)
    
    async def execute_task(self, task_id: str) -> Dict[str, Any]:
        """
        Запускает выполнение задачи указанным агентом.
        
        Args:
            task_id: ID задачи для выполнения
            
        Returns:
            Результат выполнения задачи
        """
        if task_id not in self.tasks:
            return {"error": f"Задача с ID {task_id} не найдена"}
        
        task = self.tasks[task_id]
        print(f"Executing task: {task.title} (ID: {task.task_id})")
        agent_id = task.assigned_agent_id
        
        print(f"Assigned agent ID: {agent_id}")
        
        if agent_id not in self.agents:
            task.status = "failed"
            task.result = {"error": f"Агент с ID {agent_id} не найден"}
            return task.result
        
        agent = self.agents[agent_id]
        task.status = "in_progress"
        
        print(f"in_progress: {task.status}")
        
        try:
            result = await agent.process_user_input(task.description)
            task.result = result
            task.status = "completed"
            print(f"Task completed: {task.title} (ID: {task.task_id})")
        except Exception as e:
            task.status = "failed"
            task.result = {"error": str(e)}
        
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
            return []
        
        # Создаем и запускаем задачи асинхронно
        coroutines = [self.execute_task(task_id) for task_id in pending_tasks]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        return [
            {
                "task_id": task_id,
                "result": result if not isinstance(result, Exception) else str(result)
            }
            for task_id, result in zip(pending_tasks, results)
        ]
    
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
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        self.tools = tools
        
        # Добавляем системное сообщение с промптом
        self.state.add_system_message(system_prompt)
        
        # Привязка инструментов к модели
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Создание графа агента
        self.agent = self._create_agent_graph()
    
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
        
        # Обновляем состояние агента
        if hasattr(response, "content") and response.content:
            self.state.add_assistant_message(response.content)
        
        # Обрабатываем вызовы инструментов, если они есть
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name", "unknown_tool")
                tool_args = tool_call.get("args", {})
                
                # Регистрируем вызов инструмента
                self.state.add_tool_call(tool_name, tool_args)
        
        return {"messages": [response]}
    
    def _should_continue(self, state: MessagesState) -> Literal["tools", "end"]:
        """Определяет, нужно ли вызывать инструменты или завершить обработку."""
        messages = state["messages"]
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
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
        
        # Преобразуем историю в формат для LangChain
        langchain_messages = self.state.get_conversation_history()
        
        # Вызываем агента
        result = await self.agent.ainvoke({"messages": langchain_messages})
        
        # Получаем последний ответ
        last_message = result["messages"][-1]
        response_content = last_message.content if hasattr(last_message, "content") else str(last_message)
        
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


# Пример использования
# async def example_usage():
#     """Пример использования мультиагентной системы."""
#     system = create_multi_agent_system()
    
#     # Обработка запроса пользователя супервизорным агентом
#     user_query = "Проанализируй текущую цену Bitcoin, последние новости и технические индикаторы"
#     response = await system.process_user_input(user_query)
    
#     print("Ответ супервизорного агента:")
#     print(response)
    
#     # Выполнение всех задач, созданных супервизором
#     results = await system.execute_all_pending_tasks()
    
#     print("\nРезультаты выполнения задач:")
#     for result in results:
#         print(f"Задача {result['task_id']}: {result['result']}")
    
#     return response


# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(example_usage())
