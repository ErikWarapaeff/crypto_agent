import time
import uuid
from typing import Literal, Dict, Any, List, Optional, Tuple, Union
import asyncio
from enum import Enum
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
import json
from config.settings import LLM_MODEL, LLM_TEMPERATURE
from models.state import AgentState, MessageRole, Message, ToolCall, ToolResult
from models.tool_schemas import ToolType
from datetime import datetime 
from typing import Any, Dict, List, Optional
from uuid import uuid4
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.tools import tool as langchain_tool_decorator # Alias
import operator

# Helper function for logging state
# def log_state(state: MessagesState):
#     print("-" * 80)
#     print(f"[Graph Logger] Entering node. Current State Messages:")
#     messages = state.get('messages', [])
#     if messages:
#         for i, msg in enumerate(messages):
#             print(f"  [{i}] Type: {type(msg).__name__}, Content: {getattr(msg, 'content', 'N/A')[:100]}...")
#             if hasattr(msg, 'tool_calls') and msg.tool_calls:
#                 print(f"      Tool Calls: {msg.tool_calls}")
#     else:
#         print("  (No messages in state)")
#     print("-" * 80)
#     # This node doesn't modify the state, just passes it through
#     return state

class AgentRole(str, Enum):
    """Роли агентов в системе."""
    SUPERVISOR = "supervisor"
    MARKET_ANALYST = "market_analyst"
    TECHNICAL_ANALYST = "technical_analyst"
    NEWS_RESEARCHER = "news_researcher"
    TRADER = "trader"
    PROTOCOL_ANALYST = "protocol_analyst"
    CONSULTANT = 'consultant'
    CUSTOM = "custom"
    


# 1) Определяем кастомный State для подтверждения
# class AskConfirmationState:
#     async def run(self, context, *args, **kwargs):
#         print(f"[AskConfirmationState] Asking for confirmation...") # Log
#         # await context.send_message("План норм? (да/нет)")
#         # msg = await context.receive_message()
#         # Временно хардкодим "да" для отладки графа без реального ввода
#         print(f"[AskConfirmationState] Simulating user input: 'yes'") # Log
#         txt = "да"
#         # txt = msg.content.strip().lower()
#         
#         result = "validate" if txt.startswith(("д", "y")) else "build"
#         print(f"[AskConfirmationState] Returning next node: {result}") # Log
#         # При "да" - идём в узел validate, при "нет" - обратно в build
#         return result
    
    
class CryptoAgent:
    """Класс агента для анализа криптовалют с использованием LLM и инструментов."""
    
    def __init__(self, agent_id: str, role: AgentRole, system_prompt: str, tools: List[Any], state: Optional[Any] = None):
        print(f"[CryptoAgent.__init__] Creating agent ID: {agent_id}, Role: {role.value}") # Log
        self.agent_id = agent_id
        self.role = role
        self.system_prompt = system_prompt
        
        # Правильная инициализация состояния агента с поддержкой переданного state
        if state is not None:
            self.state = state
            # Проверяем наличие метода add_system_message
            if hasattr(state, 'add_system_message') and callable(getattr(state, 'add_system_message')):
                state_type = type(state).__name__
                print(f"[CryptoAgent.__init__] Using provided state of type {state_type} with add_system_message method")
                # Для ConsultationState сообщения уже могут быть добавлены
                has_system_msg = hasattr(state, 'metadata') and state.metadata.get('system_messages')
                if not has_system_msg:
                    self.state.add_system_message(system_prompt)
            else:
                print(f"[CryptoAgent.__init__] Warning: Provided state of type {type(state).__name__} does not have add_system_message method")
        else:
            # Создаем новое состояние AgentState
            self.state = AgentState()
            print(f"[CryptoAgent.__init__] Created new AgentState")
            self.state.add_system_message(system_prompt)
        
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, base_url='https://api.vsegpt.ru/v1')
        
        # Подготовка инструментов: разворачиваем обёртки LangChain по наличию атрибута func
        prepared_tools: List[Any] = []
        for tool_item in tools:
            # LangChain декорированные инструменты могут иметь .func
            underlying = getattr(tool_item, 'func', None)
            if underlying and callable(underlying):
                func = underlying
                # Копируем метаданные из wrapper-а
                name = getattr(tool_item, 'name', None)
                description = getattr(tool_item, 'description', None)
                if name:
                    func.__name__ = name
                if description:
                    func.__doc__ = description
                prepared_tools.append(func)
            else:
                prepared_tools.append(tool_item)
        self.tools = prepared_tools
        
        # приклеиваем инструменты к LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # строим граф в зависимости от роли
        # if self.agent_id == "consultant":
        #     print(f"[CryptoAgent.__init__] Agent {self.agent_id}: Creating CONSULTANT graph.") # Log
        #     self.agent = self._create_consultant_graph()
        # else:
        #     print(f"[CryptoAgent.__init__] Agent {self.agent_id}: Creating DEFAULT graph.") # Log
        #     self.agent = self._create_default_graph()
        print(f"[CryptoAgent.__init__] Agent {self.agent_id}: Using DEFAULT graph.") # Log
        self.agent = self._create_default_graph() # Все агенты используют дефолтный граф
    
    # --------------------
    # Консультант: больше не использует кастомный граф - УДАЛЕНО
    # --------------------
    # def _create_consultant_graph(self):
    #     print(f"[CryptoAgent._create_consultant_graph] Building graph for consultant...") # Log
    #     # ... (весь код графа удален) ...
    #     print("[CryptoAgent._create_consultant_graph] Consultant graph compiled.") # Log
    #     return compiled_graph

    # # Вспомогательная функция для подготовки промпта интро - УДАЛЕНО
    # def _prepare_intro_message(self, state: MessagesState) -> dict:
    #     # ... (весь код функции удален) ...
    #     return {"messages": [intro_message]}
        
    # --------------------
    # Все остальные агенты – универсальный graph: LLM → tools → LLM → ...
    # --------------------
    def _create_default_graph(self):
        g = StateGraph(MessagesState)
        print(f"[DEBUG default_graph] Agent: {self.agent_id} | Building graph")
        g.add_node("agent", self._call_model)

        # Create the ToolNode instance
        tool_node_instance = ToolNode(self.tools)

        # Define a valid node function that wraps the ToolNode call with prints
        def debug_tool_node_wrapper(state: MessagesState) -> dict:
            print(f"[DEBUG ToolNode ENTRY] Agent: {self.agent_id} | State messages count: {len(state.get('messages', []))}")
            # Call the ToolNode instance
            try:
                result_state = tool_node_instance.invoke(state)
            except Exception as e:
                print(f"[DEBUG ToolNode ERROR] Agent: {self.agent_id} | Exception during tool invocation: {e}")
                # В случае ошибки возвращаем пустой результат, чтобы граф не падал
                # Можно добавить сюда ToolMessage с ошибкой, если нужно передать ее LLM
                from langchain_core.messages import ToolMessage
                # Пытаемся получить ID последнего tool_call для связи
                last_ai_msg = next((m for m in reversed(state.get('messages', [])) if hasattr(m, 'tool_calls') and m.tool_calls), None)
                tool_call_id = last_ai_msg.tool_calls[0]['id'] if last_ai_msg and last_ai_msg.tool_calls else None
                tool_name = last_ai_msg.tool_calls[0]['name'] if last_ai_msg and last_ai_msg.tool_calls else 'unknown_tool'
                error_content = f"Error executing tool {tool_name}: {e}"
                # Возвращаем исходное состояние + сообщение об ошибке
                error_message = ToolMessage(content=error_content, tool_call_id=tool_call_id)
                # Важно: возвращаем словарь с ключом 'messages'
                current_messages = state.get('messages', [])
                return {"messages": current_messages + [error_message]}
            # Ensure result_state is a dictionary before accessing 'messages'
            if isinstance(result_state, dict):
                messages_after = result_state.get('messages', [])
            else:
                # Handle unexpected return type from ToolNode if necessary
                messages_after = []
                print(f"[DEBUG ToolNode WARNING] Agent: {self.agent_id} | ToolNode returned non-dict: {type(result_state)}")
            print(f"[DEBUG ToolNode EXIT] Agent: {self.agent_id} | Messages count after tools: {len(messages_after)}")
            return result_state # Return the result from ToolNode

        g.add_node("tools", debug_tool_node_wrapper) # Pass the wrapper function

        g.set_entry_point("agent")
        g.add_conditional_edges(
            "agent",
            tools_condition,
            {"tools": "tools", END: END}
        )
        g.add_edge("tools", "agent")
        print(f"[DEBUG default_graph] Agent: {self.agent_id} | Graph compiled")
        return g.compile()
    
    # вызываем LLM
    async def _call_model(self, state: MessagesState):
        # DEBUG: вход в _call_model
        print(f"[DEBUG _call_model ENTRY] Agent: {self.agent_id} | messages count: {len(state.get('messages', []))}")
        
        # Для агента-консультанта добавляем специальную проверку вызовов инструментов
        if self.agent_id == "consultant":
            print(f"[DEBUG _call_model] Agent: {self.agent_id} | Performing consultant-specific checks")
            # Перед вызовом модели проверяем, нет ли вызовов недоступных инструментов
            last_message = state.get("messages", [])[-1] if state.get("messages", []) else None
            if last_message and hasattr(last_message, "content") and last_message.content:
                # Проверяем, есть ли в запросе упоминание инструментов других агентов
                content = last_message.content.lower()
                print(f"[DEBUG _call_model] Agent: {self.agent_id} | Checking message content for tool references")
                if any(agent_name in content for agent_name in ["market_analyst.", "technical_analyst.", "news_researcher.", "trader.", "protocol_analyst."]):
                    print(f"[DEBUG _call_model] Agent: {self.agent_id} | Detected attempt to call other agent tools in user message")
                    # Добавляем явное предупреждение перед вызовом LLM
                    from langchain_core.messages import SystemMessage
                    warning_message = SystemMessage(content="""
                    ВНИМАНИЕ: Ты не можешь вызывать инструменты других агентов напрямую!
                    Твоя задача - составить ПЛАН для супервизора, а не выполнять инструменты.
                    
                    Допустимые инструменты для тебя:
                    - get_agent_tool_mapping
                    - clarify_user_goal
                    - save_text_plan
                    - visualize_plan
                    - send_plan_to_supervisor
                    
                    Создай текстовый план шагов, сохрани его с save_text_plan, визуализируй и отправь супервизору!
                    """)
                    state["messages"] = state.get("messages", []) + [warning_message]
                    print(f"[DEBUG _call_model] Agent: {self.agent_id} | Added warning message about incorrect tool usage")
        
        print(f"[DEBUG _call_model] Agent: {self.agent_id} | Calling LLM with {len(state.get('messages', []))} messages")
        response = await self.llm_with_tools.ainvoke(state.get("messages", []))
        content = getattr(response, "content", "") or ""
        print(f"[DEBUG _call_model] Agent: {self.agent_id} | LLM returned content length: {len(content)}")
        if content:
            self.state.add_assistant_message(content)
        from langchain_core.messages import AIMessage
        tool_calls = getattr(response, "tool_calls", []) or []
        print(f"[DEBUG _call_model] Agent: {self.agent_id} | tool_calls: {tool_calls}")
        
        # Для консультанта: проверяем, что вызываются только разрешенные инструменты
        if self.agent_id == "consultant" and tool_calls:
            print(f"[DEBUG _call_model] Agent: {self.agent_id} | Checking {len(tool_calls)} tool calls for consultant")
            allowed_tools = ["get_agent_tool_mapping", "clarify_user_goal", "save_text_plan", "visualize_plan", "send_plan_to_supervisor"]
            invalid_calls = []
            
            for idx, call in enumerate(tool_calls):
                tool_name = call.get("name", "")
                print(f"[DEBUG _call_model] Agent: {self.agent_id} | Checking tool call: {tool_name}")
                # Проверяем на недопустимые инструменты
                if "." in tool_name or tool_name not in allowed_tools:
                    invalid_calls.append(idx)
                    print(f"[DEBUG _call_model] Agent: {self.agent_id} | Detected invalid tool call: {tool_name}")
            
            # Если найдены недопустимые вызовы, заменяем ответ на предупреждение
            if invalid_calls:
                print(f"[DEBUG _call_model] Agent: {self.agent_id} | Blocking {len(invalid_calls)} invalid tool calls. Invalid tools: {[tool_calls[idx].get('name', '') for idx in invalid_calls]}")
                
                # Создаем новое сообщение без tool_calls, но с пояснением
                warning_content = """
                Я заметил, что вы пытаетесь использовать инструменты других агентов напрямую. Это не допускается.
                
                Как консультант, я должен создать ПЛАН анализа, а не выполнять инструменты напрямую. 
                Давайте я создам для вас текстовый план анализа BTC, который затем будет передан супервизору для выполнения.
                
                Вот примерный план для анализа BTC с учетом ваших требований (краткосрочное инвестирование, средний риск, сумма 1,000,000):
                """
                
                # Генерируем план на основе запроса пользователя
                last_user_msg = None
                for msg in reversed(state.get("messages", [])):
                    if (hasattr(msg, "type") and msg.type == "human") or \
                       (isinstance(msg, HumanMessage)):
                        last_user_msg = getattr(msg, "content", "")
                        break
                
                if last_user_msg is None:
                    last_user_msg = ""
                    print(f"[DEBUG _call_model] Agent: {self.agent_id} | Could not find last user message")
                
                plan = """
                План анализа BTC для краткосрочного инвестирования:
                1. Получить текущую цену BTC через market_analyst.get_token_price
                2. Получить исторические данные за 30 дней через technical_analyst.get_token_historical_data
                3. Проанализировать новости за последнюю неделю через news_researcher.get_crypto_news
                4. Определить уровни поддержки и сопротивления на основе исторических данных
                5. Рассчитать оптимальные точки входа и выхода с учетом среднего риска
                6. Сформировать план инвестирования 1,000,000 для краткосрочной позиции
                """
                
                if "btc" in last_user_msg.lower() or "bitcoin" in last_user_msg.lower():
                    warning_content += plan
                    warning_content += """
                
                Для сохранения этого плана я могу использовать инструмент save_text_plan:
                
                Пример вызова:
                ```
                save_text_plan(plan="План анализа BTC для краткосрочного инвестирования:
                1. Получить текущую цену BTC через market_analyst.get_token_price
                2. Получить исторические данные за 30 дней через technical_analyst.get_token_historical_data
                ...")
                ```
                
                Хотите, чтобы я сохранил этот план и визуализировал его для дальнейшей отправки супервизору?
                """
                
                # Создаем новое сообщение без tool_calls
                ai_msg = AIMessage(content=warning_content)
                print(f"[DEBUG _call_model] Agent: {self.agent_id} | Replaced invalid tool calls with guidance message")
                return {"messages": state.get("messages", []) + [ai_msg]}
            else:
                print(f"[DEBUG _call_model] Agent: {self.agent_id} | All tool calls are valid for consultant")
                
        ai_msg = AIMessage(content=content, tool_calls=tool_calls)
        print(f"[DEBUG _call_model EXIT] Agent: {self.agent_id} | returning new message")
        return {"messages": state.get("messages", []) + [ai_msg]}
    
    # логика, нужна ли итерация (для default graph)
    # def _should_continue(self, state: MessagesState) -> Literal["tools", "end"]:
    #     last = state["messages"][-1]
    #     print(f"[CryptoAgent._should_continue] Agent: {self.agent_id} | Checking last message type: {type(last)}") # Log message type
    #     # print(f"[CryptoAgent._should_continue] Agent: {self.agent_id} | Last message content: {getattr(last, 'content', 'N/A')[:100]}...") # Log content snippet
    #     
    #     if hasattr(last, "tool_calls") and last.tool_calls:
    #         print(f"[CryptoAgent._should_continue] Agent: {self.agent_id} | Decision: 'tools' (AIMessage has tool calls)") # Log decision
    #         return "tools"
    #     
    #     print(f"[CryptoAgent._should_continue] Agent: {self.agent_id} | Decision: 'end' (No tool calls in last message or not AIMessage)") # Log decision
    #     return "end"
    
    # процессим вход
    async def process_user_input(self, user_input: str) -> str:
        print(f"\n[CryptoAgent.process_user_input] Agent: {self.agent_id} | Received input: '{user_input}'") # Log
        initial_state = {"messages": [HumanMessage(content=user_input)]}
        # Log initial state messages
        print(f"[CryptoAgent.process_user_input] Initial State messages snapshot:")
        for i, msg in enumerate(initial_state["messages"]):
            print(f"  [{i}] Type: {type(msg).__name__}, content: {getattr(msg, 'content', None)}")

        # Запускаем выполнение графа (стандартный цикл LLM -> Tools -> LLM)
        final_state = await self.agent.ainvoke(initial_state)
        # Log final state messages
        print(f"[CryptoAgent.process_user_input] Final State messages snapshot:")
        messages = final_state.get("messages", []) if isinstance(final_state, dict) else []
        for i, msg in enumerate(messages):
            print(f"  [{i}] Type: {type(msg).__name__}, content: {getattr(msg, 'content', None)}, tool_calls: {getattr(msg, 'tool_calls', None)}")

        # Проверяем валидность финального состояния
        if not final_state or "messages" not in final_state or not final_state["messages"]:
             print(f"[CryptoAgent.process_user_input] Agent: {self.agent_id} | Error: Final state is empty or invalid.") # Log
             return "Ошибка: Не удалось получить ответ от агента."

        # В стандартном цикле последним сообщением должно быть AIMessage без tool_calls
        last_message = final_state["messages"][-1]
        print(f"[CryptoAgent.process_user_input] Agent: {self.agent_id} | Type of last message in final state: {type(last_message)}") # Log

        if isinstance(last_message, AIMessage):
             # Извлекаем контент из последнего AIMessage
             final_output = getattr(last_message, "content", "") 
             if not final_output:
                 # Случай, когда LLM вернул пустое сообщение
                 final_output = "(Агент вернул пустое сообщение)"
                 print(f"[CryptoAgent.process_user_input] Agent: {self.agent_id} | Warning: Last AIMessage has empty content.") # Log
             else:
                 print(f"[CryptoAgent.process_user_input] Agent: {self.agent_id} | Returning content from last AIMessage: {final_output[:100]}...") # Log
             return final_output
        else:
             # Если последнее сообщение НЕ AIMessage, значит что-то пошло не так с графом.
             # Этого не должно происходить в стандартном цикле, который завершается корректно.
             print(f"[CryptoAgent.process_user_input] Agent: {self.agent_id} | Error: Expected last message to be AIMessage, but got {type(last_message)}. State history: {final_state['messages']}") # Log state history
             
             # Попытка найти последнее AIMessage в истории как fallback
             for msg in reversed(final_state["messages"]):
                 if isinstance(msg, AIMessage):
                     fallback_output = getattr(msg, "content", "")
                     if fallback_output:
                          print(f"[CryptoAgent.process_user_input] Agent: {self.agent_id} | Fallback: Returning content from last found AIMessage: {fallback_output[:100]}...") # Log
                          return fallback_output
                          
             # Если не найдено ни одного AIMessage с контентом
             error_message = f"Ошибка: Неожиданное завершение работы агента. Последнее сообщение было типа {type(last_message)}."
             print(f"[CryptoAgent.process_user_input] Agent: {self.agent_id} | {error_message}") # Log
             return error_message

    
    # сброс состояния
    def reset_state(self) -> None:
        """
        Сбрасывает состояние агента, сохраняя last_plan если он существует
        """
        # Сохраняем last_plan если он есть
        last_plan = None
        if hasattr(self.state, 'last_plan'):
            last_plan = self.state.last_plan
            print(f"[CryptoAgent.reset_state] Preserving last_plan during reset")
            
        # Определяем текущий тип состояния для правильного пересоздания
        state_type = type(self.state)
        print(f"[CryptoAgent.reset_state] Detected state type: {state_type.__name__}")
        
        if state_type.__name__ == 'ConsultationState':
            # Создаем новое состояние ConsultationState
            from core.multi_flow import ConsultationState
            self.state = ConsultationState(last_plan=last_plan)
            self.state.add_system_message(self.system_prompt)
            print(f"[CryptoAgent.reset_state] Reset to new ConsultationState with last_plan preserved: {last_plan is not None}")
        else:
            # Создаем новое состояние AgentState (default)
            self.state = AgentState(last_plan=last_plan)
            self.state.add_system_message(self.system_prompt)
            print(f"[CryptoAgent.reset_state] Reset to new AgentState with last_plan preserved: {last_plan is not None}")