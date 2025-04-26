"""Мультиагентная система для анализа криптовалют с глубоким исследованием."""

import time
import uuid
from typing import Literal, Dict, Any, List, Optional, Union
import asyncio
from enum import Enum
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from datetime import datetime

# Конфигурация
from config.settings import LLM_MODEL, LLM_TEMPERATURE

# Модели и инструменты
from models.state import AgentState, MessageRole, Message, ToolCall, ToolResult
from models.tool_schemas import ToolType
from tools import (
    get_token_price, get_trending_coins, search_cryptocurrencies,
    analyze_protocol, analyze_pools_geckoterminal, get_token_historical_data,
    analyze_token_holders, get_crypto_price, get_klines_history, execute_trade,
    confirm_trade, get_market_info, get_account_info, get_crypto_news,
    get_crypto_tweets, get_crypto_hacks, get_token_unlocks, get_project_raises,
    get_polymarket_data, get_market_summary
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
    DEEP_RESEARCHER = "deep_researcher"

class Task(BaseModel):
    """Модель задачи для агентов."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    assigned_agent_id: Optional[str] = None
    status: str = Field(default="pending")
    priority: int = Field(default=1)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    result: Optional[Any] = None
    parent_task_id: Optional[str] = None
    sub_tasks: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ResearchPlanner:
    """Планировщик исследований с уточнением запросов."""
    
    def __init__(self):
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1)
        
    async def refine_query(self, user_input: str) -> Union[str, Dict[str, str]]:
        """Генерирует уточненный запрос или просит пояснений."""
        prompt = f"""
        Анализируй запрос пользователя и генерируй:
        1. Уточненный технический запрос для системы
        2. Список требуемых инструментов
        3. Или запрос на уточнение

        Исходный запрос: {user_input}

        Формат ответа:
        {{"query": "уточненный_запрос", "tools": ["tool1", "tool2"]}}
        или
        {{"clarify": "текст_уточнения"}}
        """
        response = await self.llm.ainvoke(prompt)
        return eval(response.content)

class MultiAgentSystem:
    """Управление мультиагентной системой с поддержкой глубокого исследования."""
    
    def __init__(self):
        self.agents = {}
        self.tasks = {}
        self.research_planner = ResearchPlanner()
        self.initialize_agents()

    def initialize_agents(self):
        """Инициализация всех агентов системы."""
        self.create_supervisor()
        self.create_specialized_agents()

    def create_supervisor(self):
        """Создание супервизорного агента."""
        supervisor_tools = [
            self._create_delegate_task_tool(),
            self._create_check_task_status_tool(),
            self._create_merge_results_tool()
        ]
        
        supervisor = CryptoAgent(
            agent_id="supervisor",
            role=AgentRole.SUPERVISOR,
            system_prompt="Ты - центральный координатор системы. Анализируй запросы и делегируй задачи.",
            tools=supervisor_tools
        )
        self.agents["supervisor"] = supervisor

    def create_specialized_agents(self):
        """Создание специализированных агентов."""
        agents_config = [
            {
                "id": "market_analyst",
                "role": AgentRole.MARKET_ANALYST,
                "tools": [get_token_price, get_trending_coins],
                "prompt": "Анализ рыночных данных и цен"
            },
            {
                "id": "technical_analyst",
                "role": AgentRole.TECHNICAL_ANALYST,
                "tools": [get_token_historical_data],
                "prompt": "Технический анализ и прогнозирование"
            }
        ]
        
        for config in agents_config:
            agent = CryptoAgent(
                agent_id=config["id"],
                role=config["role"],
                system_prompt=config["prompt"],
                tools=config["tools"]
            )
            self.agents[config["id"]] = agent

    async def process_user_request(self, user_input: str) -> str:
        """Обработка запроса с использованием глубокого исследования."""
        # Этап 1: Уточнение запроса
        research_result = await self.research_planner.refine_query(user_input)
        
        if "clarify" in research_result:
            return research_result["clarify"]
        
        # Этап 2: Выполнение уточненного запроса
        if "query" in research_result:
            return await self.agents["supervisor"].process_user_input(research_result["query"])
        
        return "Не удалось обработать запрос"

    # Реализация инструментов супервизора
    def _create_delegate_task_tool(self):
        def delegate_task(agent_id: str, **kwargs):
            agent_id = agent_id.lower()
            if agent_id not in self.agents:
                return f"Агент {agent_id} не найден"
            
            task = Task(**kwargs)
            self.tasks[task.task_id] = task
            return f"Создана задача {task.task_id}"
        return delegate_task

    def _create_check_task_status_tool(self):
        def check_status(task_id: str):
            return self.tasks.get(task_id, {"error": "Задача не найдена"})
        return check_status

    def _create_merge_results_tool(self):
        def merge_results(task_ids: List[str]):
            return [self.tasks[tid] for tid in task_ids if tid in self.tasks]
        return merge_results

class CryptoAgent:
    """Агент с расширенными возможностями обработки запросов."""
    
    def __init__(self, agent_id: str, role: AgentRole, system_prompt: str, tools: list):
        self.agent_id = agent_id
        self.role = role
        self.state = AgentState()
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        self.tools = tools
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        """Создание графа обработки сообщений."""
        workflow = StateGraph(MessagesState)
        workflow.add_node("process", self.process_message)
        workflow.add_node("tools", ToolNode(self.tools))
        
        workflow.set_entry_point("process")
        workflow.add_conditional_edges(
            "process",
            lambda s: "tools" if s["messages"][-1].tool_calls else "end",
            {"tools": "tools", "end": END}
        )
        workflow.add_edge("tools", "process")
        return workflow.compile()

    async def process_message(self, state: MessagesState):
        """Обработка входящих сообщений."""
        response = await self.llm.ainvoke(state["messages"])
        return {"messages": [response]}

    async def process_user_input(self, user_input: str) -> str:
        """Основной метод обработки пользовательского ввода."""
        self.state.add_user_message(user_input)
        result = await self.workflow.ainvoke({"messages": self.state.get_conversation_history()})
        return result["messages"][-1].content

# Создание и запуск системы
async def main():
    system = MultiAgentSystem()
    response = await system.process_user_request("Проанализировать Bitcoin")
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
