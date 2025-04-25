"""Логика работы с LLM и инструментами."""

import time
from typing import Literal, Dict, Any, List, Tuple
import asyncio

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

from .deep_research import DeepResearchManager

from config.settings import LLM_MODEL, LLM_TEMPERATURE
from tools import (
    get_token_price,
    get_trending_coins,
    search_cryptocurrencies,
    analyze_protocol,
    analyze_pools_geckoterminal,
    get_token_historical_data,
    analyze_token_holders,
    # fetch_crypto_news,
    # Добавляем новые инструменты HyperLiquid
    get_crypto_price,
    get_klines_history,
    execute_trade,
    confirm_trade,
    get_market_info,
    get_account_info,
        
    # Инструменты LlamaFeed
    get_crypto_news,
    get_crypto_tweets,
    get_crypto_hacks,
    get_token_unlocks,
    get_project_raises,
    get_polymarket_data,
    get_market_summary
)
from models.state import AgentState, ToolCall, ToolResult
from models.tool_schemas import ToolType


class CryptoAgent:
    """Класс агента для анализа криптовалют с использованием LLM и инструментов."""
    
    def __init__(self):
        """Инициализация агента и его компонентов."""
        self.state = AgentState()
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        
        # Список всех инструментов
        self.tools = [
            get_token_price,
            get_trending_coins,
            search_cryptocurrencies,
            analyze_protocol,
            analyze_pools_geckoterminal,
            get_token_historical_data,
            analyze_token_holders,
            # fetch_crypto_news,
            
            # Инструменты HyperLiquid
            get_crypto_price,
            get_klines_history,
            execute_trade,
            confirm_trade,
            get_market_info,
            get_account_info,
            
            # Инструменты LlamaFeed
            get_crypto_news,
            get_crypto_tweets,
            get_crypto_hacks,
            get_token_unlocks,
            get_project_raises,
            get_polymarket_data,
            get_market_summary
        ]
        
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
        """Обрабатывает ввод пользователя и возвращает ответ."""
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
        
    async def perform_deep_research(self, token_symbol: str) -> str:
        """
        Выполняет глубокое исследование токена.
        
        Args:
            token_symbol: Символ токена для исследования
            
        Returns:
            Строка с отчетом о результатах исследования
        """
        research_manager = DeepResearchManager(llm_model=LLM_MODEL)
        result = await research_manager.conduct_research(self.state, token_symbol)
        
        # Добавляем результат исследования в историю сообщений
        self.state.add_assistant_message(result)
        
        return result


# Функция для создания экземпляра агента
def create_agent() -> CryptoAgent:
    """Создает и возвращает агента для анализа криптовалют."""
    return CryptoAgent()