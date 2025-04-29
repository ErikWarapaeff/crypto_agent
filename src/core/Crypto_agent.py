import time
import uuid
from typing import Literal, Dict, Any, List, Optional, Tuple, Union
import asyncio
from enum import Enum
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
import json
from config.settings import LLM_MODEL, LLM_TEMPERATURE
from models.state import AgentState, MessageRole, Message, ToolCall, ToolResult
from models.tool_schemas import ToolType
from datetime import datetime 
from typing import Any, Dict, List, Optional
from uuid import uuid4
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

class AgentRole(str, Enum):
    """Роли агентов в системе."""
    SUPERVISOR = "supervisor"
    MARKET_ANALYST = "market_analyst"
    TECHNICAL_ANALYST = "technical_analyst"
    NEWS_RESEARCHER = "news_researcher"
    TRADER = "trader"
    PROTOCOL_ANALYST = "protocol_analyst"
    CONSULTANT = 'concultant'
    CUSTOM = "custom"
    


# 1) Определяем кастомный State для подтверждения
class AskConfirmationState:
    async def run(self, context, *args, **kwargs):
        await context.send_message("План норм? (да/нет)")
        msg = await context.receive_message()
        txt = msg.content.strip().lower()
        # При "да" - идём в узел validate, при "нет" - обратно в build
        return "validate" if txt.startswith(("д", "y")) else "build"
    
    
class CryptoAgent:
    """Класс агента для анализа криптовалют с использованием LLM и инструментов."""
    
    def __init__(self, agent_id: str, role: AgentRole, system_prompt: str, tools: List[Any], state: Optional[AgentState] = None):
        self.agent_id = agent_id
        self.role = role
        self.system_prompt = system_prompt
        self.state = AgentState() or state  # Инициализация состояния агента
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, base_url='https://api.vsegpt.ru/v1')
        
        # инструменты: [ get_agent_tool_mapping, dynamic_planner ] для консультанта
        self.tools = tools
        
        # системное сообщение
        self.state.add_system_message(system_prompt)
        
        # приклеиваем инструменты к LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # строим граф в зависимости от роли
        if self.agent_id == "consultant":
            self.agent = self._create_consultant_graph()
        else:
            self.agent = self._create_default_graph()
    
    # --------------------
    # Консультант: mapping → planner → END
    # --------------------



    def _create_consultant_graph(self):
        # 2) Инструменты
        mapping_node   = ToolNode([self.tools[0]])
        clarify_node   = ToolNode([self.tools[1]])
        build_node     = ToolNode([self.tools[2]])
        visualize_node = ToolNode([self.tools[3]])
        validate_node  = ToolNode([self.tools[4]])
        fallback_node  = ToolNode([self.tools[5]])

        # 3) Собираем граф
        g = StateGraph(MessagesState)

        g.add_node("mapping",   mapping_node)
        g.add_node("clarify",   clarify_node)
        g.add_node("build",     build_node)
        g.add_node("visualize", visualize_node)

        # 4) Наш interrupt-State
        g.add_node("ask_confirmation", AskConfirmationState)

        g.add_node("validate",  validate_node)
        g.add_node("fallback",  fallback_node)

        g.set_entry_point("mapping")

        # 5) Линейные переходы
        g.add_edge("mapping",   "clarify")
        g.add_edge("clarify",   "build")
        g.add_edge("build",     "visualize")

        # 6) После visualize - сразу в ask_confirmation
        g.add_edge("visualize", "ask_confirmation")

        # 7) Рёбра без параметров:
        #    если run() вернул "validate" - Graph найдёт ребро visualize→ask_confirmation→validate    
        g.add_edge("ask_confirmation", "validate")
        #    если run() вернул "build" - Graph сам вернёт вас к build
        g.add_edge("ask_confirmation", "build")

        # 8) Дальше fallback и конец
        g.add_edge("validate",  "fallback")
        g.add_edge("fallback",  END)

        return g.compile()

        
    # --------------------
    # Все остальные агенты – универсальный graph: LLM → tools → LLM → ...
    # --------------------
    def _create_default_graph(self):
        tool_node = ToolNode(self.tools)
        g = StateGraph(MessagesState)
        g.add_node("agent", self._call_model)
        g.add_node("tools", tool_node)
        g.set_entry_point("agent")
        g.add_conditional_edges(
            "agent",
            self._should_continue,
            {"tools": "tools", "end": END}
        )
        g.add_edge("tools", "agent")
        return g.compile()
    
    # вызываем LLM
    async def _call_model(self, state: MessagesState):
        messages = state["messages"]
        start = time.time()
        response = await self.llm_with_tools.ainvoke(messages)
        exec_time = time.time() - start
        
        if hasattr(response, "content") and response.content:
            self.state.add_assistant_message(response.content)
        
        if hasattr(response, "tool_calls") and response.tool_calls:
            for call in response.tool_calls:
                self.state.add_tool_call(call["name"], call.get("args", {}))
        
        return {"messages": [response]}
    
    # логика, нужна ли итерация
    def _should_continue(self, state: MessagesState) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"
    
    # процессим вход
    async def process_user_input(self, user_input: str) -> str:
        # для консультанта – LLM не участвует, всё инструменты
        if self.agent_id == "consultant":
            # 1) получаем карту агентов
            mapping = await self.tools[0]()  
            # 2) строим план
            plan = await self.tools[1](user_query=user_input, agent_mapping=mapping)
            # просто отдаем JSON
            return plan
        
        # для всех остальных – стандартный flow
        self.state.add_user_message(user_input)
        msgs = self.state.get_conversation_history()
        out = await self.agent.ainvoke({"messages": msgs})
        last = out["messages"][-1]
        return getattr(last, "content", str(last))
    
    # сброс состояния
    def reset_state(self) -> None:
        self.state = AgentState()
        self.state.add_system_message(self.system_prompt)