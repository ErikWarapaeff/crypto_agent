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
import json
from config.settings import LLM_MODEL, LLM_TEMPERATURE
from models.state import AgentState, MessageRole, Message, ToolCall, ToolResult
from models.tool_schemas import ToolType
from datetime import datetime 
from uuid import uuid4
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from core.Crypto_agent import CryptoAgent, AgentRole
from langchain_core.tools import tool as langchain_tool_decorator # Alias


# Импорт инструментов
# Инструменты CoinGecko
from tools.coingecko_tools import (
    get_token_price, # Market Analyst
    get_trending_coins, # Market Analyst
    search_cryptocurrencies # Market Analyst
)
# Инструменты LlamaFeed
from tools.llamafeed_tools import (
    get_crypto_news, # News Researcher
    get_crypto_tweets, # News Researcher
    get_crypto_hacks, # News Researcher
    get_token_unlocks, # News Researcher
    get_project_raises, # News Researcher
    get_polymarket_data, # News Researcher
    get_market_summary # News Researcher
)
# Инструменты HyperLiquid
from tools.hyperliquid_tools import (
    get_crypto_price, # Market Analyst, Technical Analyst?
    get_klines_history, # Technical Analyst
    execute_trade, # Trader
    confirm_trade, # Trader
    get_market_info, # Technical Analyst
    get_account_info # Trader
)
# Инструменты анализа холдеров
from tools.holder_analysis_tools import (
    analyze_token_holders # Protocol Analyst
)
# Инструменты анализа протоколов DeFi
from tools.defi_protocol_tools import (
    analyze_protocol, # Protocol Analyst
    analyze_pools_geckoterminal # Protocol Analyst
)
# Инструменты анализа исторических данных токена
from tools.token_analysis_tools import (
    get_token_historical_data # Technical Analyst
)
# (Инструмент fetch_crypto_news из crypto_news_tools дублирует get_crypto_news из llamafeed_tools,
# используем get_crypto_news из llamafeed_tools как основной)
# from tools.crypto_news_tools import fetch_crypto_news





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
    
    
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class ExecutedStep(BaseModel):
    step: int
    agent: str
    tool: str
    args: Dict[str, Any]
    status: str
    result: Any
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class HistoryEvent(BaseModel):
    event: str
    data: Any
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ConsultationState(BaseModel):
    """
    Pydantic-модель для хранения состояния диалога агента-консультанта.
    """
    user_query: Optional[Any] = None
    user_profile: Optional[Dict[str, Any]] = None
    agent_mapping: Optional[Dict[str, Any]] = None

    clarification_questions: List[str] = Field(default_factory=list)
    clarification_answers: Dict[str, Any] = Field(default_factory=dict)

    strategy: Optional[Dict[str, Any]] = None
    visualized_plan: Optional[str] = None
    last_plan: Optional[str] = None  # Для хранения последнего созданного плана

    current_step: int = 0
    executed_steps: List[ExecutedStep] = Field(default_factory=list)

    results: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Для хранения произвольных метаданных

    completed: bool = False
    need_clarification: bool = False
    fallback_applied: bool = False

    history: List[HistoryEvent] = Field(default_factory=list)
    
    # Дополнительные методы для совместимости с AgentState
    def add_system_message(self, content: str) -> None:
        """
        Имитирует добавление системного сообщения для совместимости с AgentState.
        В ConsultationState это просто сохраняет системное сообщение в метаданных.
        """
        if not hasattr(self, 'metadata'):
            self.metadata = {}
        
        if 'system_messages' not in self.metadata:
            self.metadata['system_messages'] = []
            
        self.metadata['system_messages'].append(content)
        print(f"[ConsultationState.add_system_message] Saved system message to metadata")
        
    def add_user_message(self, content: str) -> None:
        """
        Имитирует добавление сообщения пользователя для совместимости с AgentState.
        """
        self.user_query = content
        self.log_event("user_message", content)
        
    def add_assistant_message(self, content: str) -> None:
        """
        Имитирует добавление сообщения ассистента для совместимости с AgentState.
        """
        if not hasattr(self, 'metadata'):
            self.metadata = {}
            
        if 'assistant_messages' not in self.metadata:
            self.metadata['assistant_messages'] = []
            
        self.metadata['assistant_messages'].append(content)
        self.log_event("assistant_message", content)

    class Config:
        # При сериализации дат в JSON используем ISO-формат
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

    def reset(self, keep_profile: bool = True) -> None:
        """
        Сбросить всё состояние, опционально сохранив user_profile.
        """
        profile = self.user_profile if keep_profile else None
        last_plan = self.last_plan  # Сохраняем план
        self.__dict__.update(ConsultationState(user_profile=profile, last_plan=last_plan).dict())

    def log_event(self, event: str, data: Any = None) -> None:
        """
        Добавить запись в history.
        """
        self.history.append(HistoryEvent(event=event, data=data))


class MultiAgentSystem:
    def __init__(self):
        self.agents = {}
        self.tasks = {}
        self.global_state = {}
        self._agent_mapping_tool = None

        # 1) супервизор
        self.create_supervisor_agent()

        # 2) регистрируем mapper
        self._register_get_agent_tool_mapping()

        # 3) специалисты
        self.initialize_specialized_agents()

        # 4) консультант (единожды)
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
        У тебя две основные задачи:

        1. Выполнять планы анализа, созданные консультантом. План содержит последовательность шагов,
           где для каждого шага указан агент, инструмент и параметры. Твоя задача - последовательно
           делегировать каждый шаг соответствующему агенту с помощью инструмента delegate_task.

        2. Анализировать запросы пользователя, разбивать их на подзадачи и делегировать их 
           соответствующим агентам.
        
        Когда тебе передают план от консультанта, твоя задача - выполнить каждый шаг, делегируя
        задачи соответствующим агентам. Для каждого шага:
        1. Определи, какой агент должен выполнить шаг
        2. Для каждого агента используй соответствующий инструмент:
            - MARKET_ANALYST: для анализа цен и трендов
            - TECHNICAL_ANALYST: для технического анализа графиков
            - NEWS_RESEARCHER: для исследования новостей и событий 
            - TRADER: для торговых операций
            - PROTOCOL_ANALYST: для анализа протоколов и холдеров
        3. Используй инструмент delegate_task(agent_id, task_title, task_description, priority)
        4. После делегирования задачи проверь её статус с check_task_status(task_id)
        5. После выполнения всех шагов объедини результаты с merge_results([task_ids], summary_title)
        
        Доступные тебе инструменты:
        • delegate_task - делегирует задачу указанному агенту
        • check_task_status - проверяет статус задачи
        • merge_results - объединяет результаты нескольких задач
        
        **ВАЖНО:** Прежде чем использовать инструмент `delegate_task`, убедись, что ты не делегировал 
        **точно такую же** задачу этому же агенту **в рамках текущего ответа**. 
        Если задача уже была успешно делегирована (инструмент `delegate_task` вернул ID задачи), 
        не вызывай `delegate_task` для неё повторно. 
        Ты можешь использовать `check_task_status`, чтобы узнать статус уже делегированной задачи.
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
            get_token_price, # CoinGecko
            get_trending_coins, # CoinGecko
            search_cryptocurrencies, # CoinGecko
            get_crypto_price, # HyperLiquid
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
            get_token_historical_data, # Token Analysis Tools
            get_klines_history, # HyperLiquid
            get_market_info, # HyperLiquid
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
            get_crypto_news, # LlamaFeed
            get_crypto_tweets, # LlamaFeed
            get_crypto_hacks, # LlamaFeed
            get_token_unlocks, # LlamaFeed
            get_project_raises, # LlamaFeed
            get_polymarket_data, # LlamaFeed
            get_market_summary, # LlamaFeed
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
            execute_trade, # HyperLiquid
            confirm_trade, # HyperLiquid
            get_account_info, # HyperLiquid
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
            analyze_protocol, # Defi Protocol Tools
            analyze_pools_geckoterminal, # Defi Protocol Tools
            analyze_token_holders, # Holder Analysis Tools
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
        
        self._register_get_agent_tool_mapping()

        # 3) Наконец создаём агента-консультанта, отдавая ему этот инструмент
        self.create_consultant_agent()
        
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
            
            ВАЖНО: Эти инструменты недоступны для прямого использования консультантом!
            Консультант использует эту информацию только для планирования
            и должен создать текстовый план, а НЕ вызывать инструменты напрямую.
            """
            mapping = {}
            for ag_id, agent in self.agents.items():
                tools_info = []
                for fn in agent.tools:
                    # Корректное извлечение имени и описания для Langchain tools
                    tool_name = getattr(fn, 'name', None) 
                    description = getattr(fn, 'description', None)
                    
                    # Фоллбэк для обычных функций или если атрибуты не найдены
                    if tool_name is None:
                         tool_name = getattr(fn, '__name__', 'unknown_tool')
                    if description is None:
                         docstring = getattr(fn, '__doc__', None) or ""
                         description = docstring.strip().split("\n")[0] # Первая строка докстринга

                    tools_info.append({
                        "name": tool_name,
                        "description": description if description else "(No description)", # Добавим заглушку
                        "available": False if ag_id != "consultant" else True, # Отмечаем, что эти инструменты недоступны консультанту напрямую
                        "agent_only": ag_id != "consultant" # Только для данного агента
                    })
                mapping[ag_id] = tools_info
            print(f"[get_agent_tool_mapping Tool] Generated mapping for {len(mapping)} agents.") # Log
            # print(f"[get_agent_tool_mapping Tool] Mapping: {json.dumps(mapping, indent=2, ensure_ascii=False)}") # Optional: Log full mapping
            
            # Добавим предупреждение к результату
            result = {
                "warning": "ВАЖНО: Инструменты других агентов НЕДОСТУПНЫ консультанту напрямую! Используйте их только для составления плана.",
                "available_tools": ["get_agent_tool_mapping", "clarify_user_goal", "visualize_plan", "send_plan_to_supervisor"],
                "agents": mapping
            }
            return result

        # Decorate the inner function and return it
        return langchain_tool_decorator(get_agent_tool_mapping)
    
    def _register_get_agent_tool_mapping(self):
        """Регистрирует get_agent_tool_mapping в системном списке инструментов."""
        mapper_tool = self._create_get_agent_tool_mapping_tool()
        # Сохраняем его в атрибуте, чтобы потом передать в create_consultant_agent
        self._agent_mapping_tool = mapper_tool

    # Помощник для создания обёрток-функций консультанта
    def _wrap_consultant_tool(self, original_tool):
        """Создает обертку для инструмента консультанта, добавляя agent_mapping при необходимости."""
        tool_name = getattr(original_tool, 'name', getattr(original_tool, '__name__', 'unknown_tool'))
        # Описание будет взято из докстринга wrapper_with_mapping
        original_description = getattr(original_tool, 'description', getattr(original_tool, '__doc__', 'No description available'))

        def wrapper_with_mapping(*args, **kwargs):
            """Wrapper for tool that injects agent_mapping parameter if needed."""
            print(f"[ConsultantWrapper] Tool {tool_name} called with args={args}, kwargs={kwargs}")
            
            # Обработка аргумента 'args', если он передан через kwargs
            if 'args' in kwargs and tool_name in ['visualize_plan', 'send_plan_to_supervisor', 'save_text_plan']:
                # Извлекаем значение args для передачи в качестве strategy/plan
                args_value = kwargs.pop('args')
                print(f"[ConsultantWrapper] Extracting args to use as strategy/plan for {tool_name}: {args_value}")
                
                # Определяем имя первого параметра функции
                first_param_name = 'strategy' if tool_name == 'visualize_plan' else 'plan'
                
                # Если first_param_name не передан явно, добавляем его
                if first_param_name not in kwargs:
                    # Проверяем, если это список или кортеж с одним элементом, извлекаем его
                    if isinstance(args_value, (list, tuple)) and len(args_value) == 1:
                        # Передаем только первый элемент, чтобы избежать вложенности
                        kwargs[first_param_name] = args_value
                    else:
                        kwargs[first_param_name] = args_value
            
            # Конвертируем позиционные аргументы в именованные, если они есть
            if args and tool_name in ['visualize_plan', 'send_plan_to_supervisor', 'save_text_plan']:
                # Определяем имя первого параметра функции
                first_param_name = 'strategy' if tool_name == 'visualize_plan' else 'plan'
                
                # Если первый параметр не передан явно в kwargs, используем первый аргумент
                if first_param_name not in kwargs:
                    kwargs[first_param_name] = args[0] if len(args) > 0 else None
                    print(f"[ConsultantWrapper] Converting positional arg to {first_param_name}")
            
            # Добавляем agent_mapping для clarify_user_goal
            if tool_name in ['clarify_user_goal'] and 'agent_mapping' not in kwargs:
                if self._agent_mapping_tool:
                    mapping = self._agent_mapping_tool()
                    print(f"[ConsultantWrapper] Injecting agent_mapping into {tool_name}: type={type(mapping)}")
                    kwargs['agent_mapping'] = mapping
                else:
                    print(f"[ConsultantWrapper] Error: _agent_mapping_tool not available for {tool_name}")

            # Вызываем *оригинальную* функцию инструмента (не self.clarify_user_goal и т.д., чтобы избежать рекурсии)
            # Мы предполагаем, что original_tool - это уже декорированный объект @tool
            # и его исполняемая логика доступна через .func
            original_func = getattr(original_tool, 'func', original_tool) # Получаем исходную функцию
            print(f"[ConsultantWrapper] Calling original func for {tool_name} with args={args}, kwargs={kwargs}")
            try:
                result = original_func(*args, **kwargs)
                print(f"[ConsultantWrapper] Original func for {tool_name} returned: {result}")
                return result
            except Exception as e:
                print(f"[ConsultantWrapper] Error calling {tool_name}: {e}")
                return f"Произошла ошибка при вызове инструмента {tool_name}: {str(e)}"

        # Присваиваем имя и докстринг нашей обертке
        wrapper_with_mapping.__name__ = tool_name
        # Заменяем докстринг на более информативный перед декорированием
        wrapper_with_mapping.__doc__ = f"Wrapper for {tool_name} that injects agent_mapping if needed. Original description: {original_description}"

        # Декорируем обертку БЕЗ лишних аргументов
        wrapped_tool = langchain_tool_decorator(wrapper_with_mapping)
        return wrapped_tool

    @tool(description="Генерирует список уточняющих вопросов, основанных на недостающей информации и доступных инструментах.")
    def clarify_user_goal(
        user_query: str,
        agent_mapping: Dict[str, Any],
        args: List = None  # Добавляем для совместимости с разными форматами вызова
    ) -> List[str]:
        # DEBUG: Print received arguments
        print(f"[DEBUG clarify_user_goal ENTRY] user_query: {user_query!r}, agent_mapping: {type(agent_mapping)}")
        
        # Обработка нестандартных входных данных
        if not user_query and args and isinstance(args, list) and len(args) > 0:
            # Если user_query пустой, но переданы аргументы через args, пробуем использовать их
            print(f"[clarify_user_goal] Received args instead of user_query: {args}")
            if isinstance(args[0], str):
                user_query = args[0]
                print(f"[clarify_user_goal] Extracted user_query from args: {user_query}")
        
        """
        Генерирует список уточняющих вопросов, основанных
        на том, какую информацию недостаёт для построения стратегии
        и какие инструменты/агенты есть в системе.
        
        Вход:
        - user_query: исходный запрос пользователя
        - agent_mapping: карта всех агентов и их инструментов
            
        Выход:
        Список вопросов (List[str]). Если всё ясно - пустой список.
        """
        # Проверка обязательных аргументов
        if not agent_mapping:
            error_msg = "Ошибка: параметр agent_mapping обязателен и не может быть пустым"
            print(f"[clarify_user_goal] {error_msg}")
            return ["Не удалось сгенерировать уточняющие вопросы из-за отсутствия данных об агентах."]
            
        if not user_query:
            error_msg = "Ошибка: параметр user_query обязателен и не может быть пустым"
            print(f"[clarify_user_goal] {error_msg}")
            return ["Пожалуйста, укажите, что именно вы хотите проанализировать?"]

        # Адаптируем формат agent_mapping к ожидаемому формату
        agents_data = agent_mapping.get("agents", {}) if isinstance(agent_mapping, dict) else agent_mapping
        
        print(f"[clarify_user_goal] Получены аргументы: user_query='{user_query}', agent_mapping (keys)={list(agents_data.keys()) if isinstance(agents_data, dict) else 'не словарь'}")
        
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
        2) agent_mapping - JSON-карта всех агентов и их инструментов
        
        Твоя задача - проанализировать запрос и список доступных инструментов,
        выявить, какие параметры или детали пользователю нужно уточнить 
        (например: символ актива, горизонт анализа, тип анализа, профиль пользователя 
        и пр.), и сформулировать эти уточнения в виде массива вопросов.
        
        Требования:
        - Не хардкодь имена параметров; извлекай из названий и описаний инструментов.
        - Выводи **чистый** JSON-массив строк, без лишнего текста.
        - Если дополнительных вопросов не требуется - верни пустой массив `[]`.
        """
        
        # 3) Соберём сообщение пользователю с картой инструментов
        user_prompt = f"""
        Задача пользователя:
        {json.dumps(user_query, ensure_ascii=False)}
        Карта агентов и инструментария:
        {json.dumps(agents_data, indent=2, ensure_ascii=False)}
        """
        # 4) Вызов LLM
        response = llm([
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ])
        
        # 5) Парсим результат из JSON
        try:
            questions = json.loads(response.content)
            if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
                print(f"[clarify_user_goal] Сгенерировано {len(questions)} вопросов")
                return questions
        except Exception as e:
            print(f"[clarify_user_goal] Ошибка парсинга JSON: {e}")
            print(f"[clarify_user_goal] Ответ LLM: {response.content}")
        
        # В случае ошибки парсинга - возвращаем стандартный вопрос
        print(f"[clarify_user_goal] Возвращаем стандартный вопрос из-за ошибки")
        return ["Не удалось автоматически определить, какая информация нужна. Пожалуйста, уточните ваш запрос - что именно вы хотите проанализировать?"]
    
    @tool(description="Генерирует пошаговый план анализа на основе запроса, карты агентов и профиля пользователя.")
    def build_analysis_strategy(
        user_query: str,
        agent_mapping: Dict[str, Any],
        user_profile: Dict[str, Any] = None,
        args: List = None  # Добавляем параметр для совместимости с разными форматами вызова
    ) -> Dict[str, Any]:
        # DEBUG: Print received arguments
        print(f"[DEBUG build_analysis_strategy ENTRY] user_query: {user_query!r}, agent_mapping keys: {list(agent_mapping.keys()) if agent_mapping else 'None'}, profile: {user_profile is not None}")
        
        # Обработка нестандартных входных данных
        if not user_query and args and isinstance(args, list) and len(args) > 0:
            # Если user_query пустой, но переданы аргументы через args, пробуем использовать их
            print(f"[build_analysis_strategy] Received args instead of user_query: {args}")
            if isinstance(args[0], str):
                user_query = args[0]
                print(f"[build_analysis_strategy] Extracted user_query from args: {user_query}")
        
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
        # Проверка обязательных аргументов
        if not agent_mapping:
            error_msg = "Ошибка: параметр agent_mapping обязателен и не может быть пустым"
            print(f"[build_analysis_strategy] {error_msg}")
            return {
                "steps": [],
                "need_clarification": True,
                "error": error_msg
            }
            
        if not user_query:
            error_msg = "Ошибка: параметр user_query обязателен и не может быть пустым"
            print(f"[build_analysis_strategy] {error_msg}")
            return {
                "steps": [],
                "need_clarification": True,
                "error": error_msg
            }

        print(f"[build_analysis_strategy] Получены аргументы: user_query='{user_query}', agent_mapping (keys)={list(agent_mapping.keys())}, user_profile={user_profile is not None}")

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
        response = llm([
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ])

        # 5) Парсинг результата
        try:
            plan = json.loads(response.content)
            # Проверяем наличие обязательных полей
            if isinstance(plan, dict) and "steps" in plan and "need_clarification" in plan:
                print(f"[build_analysis_strategy] Сгенерирован план с {len(plan.get('steps', []))} шагами, need_clarification={plan.get('need_clarification')}")
                return plan
        except Exception as e:
            print(f"[build_analysis_strategy] Ошибка парсинга JSON: {e}")
            print(f"[build_analysis_strategy] Ответ LLM: {response.content}")

        # 6) Фоллбэк
        fallback_result = {
            "steps": [],
            "need_clarification": True,
            "error": "Не удалось создать план анализа"
        }
        print(f"[build_analysis_strategy] Возвращаем fallback результат")
        return fallback_result
        
    @tool(description="Преобразует текстовый план или JSON-стратегию в отформатированный план (markdown или mermaid).")
    def visualize_plan(
        strategy: Union[Dict, List, str, Tuple] = None,
        format: str = "markdown",
        auto_send: bool = True,  # Добавляем параметр для автоматической отправки плана супервизору
        args: Any = None  # Добавляем параметр args для совместимости
    ) -> str:
        """
        Преобразует план анализа в красиво отформатированную визуализацию.
        Параметры:
        - strategy: может быть в 3 форматах:
          1. JSON-объект {steps: [{step, agent, tool, args, reason}, ...]}
          2. Текстовый план с нумерованными шагами
          3. Список шагов
        - format: "markdown" (таблица) или "mermaid" (flowchart)
        - auto_send: автоматически отправлять план супервизору после визуализации
        - args: альтернативный способ передачи strategy (для совместимости)

        Возвращает строку с визуализацией.
        """
        # Если strategy не указан, но передан args, используем его
        if strategy is None and args is not None:
            strategy = args
            print(f"[visualize_plan] Using args as strategy: {type(args)}")
            
        # Если strategy - кортеж с одним элементом, извлекаем его содержимое
        if isinstance(strategy, tuple) and len(strategy) == 1:
            strategy = strategy[0]
            print(f"[visualize_plan] Extracted strategy from tuple: {type(strategy)}")
            
        if strategy is None:
            return "Ошибка: не указан план для визуализации"
            
        print(f"[visualize_plan] Input Strategy type: {type(strategy)}, value: {strategy}") # Debug log

        # Сохраняем план в состоянии консультанта, если вызов идет из MultiAgentSystem
        from inspect import currentframe, getouterframes
        frame = currentframe()
        try:
            for f in getouterframes(frame):
                if 'self' in f.frame.f_locals and isinstance(f.frame.f_locals['self'], MultiAgentSystem):
                    mas_instance = f.frame.f_locals['self']
                    if 'consultant' in mas_instance.agents:
                        consultant = mas_instance.agents['consultant']
                        
                        # Проверяем наличие state и создаем его при необходимости
                        if not hasattr(consultant, 'state'):
                            consultant.state = ConsultationState(last_plan="")
                            print(f"[visualize_plan] Created new state for consultant")
                        
                        # Проверяем наличие атрибута last_plan и создаем его при необходимости
                        if not hasattr(consultant.state, 'last_plan'):
                            consultant.state.last_plan = ""
                            print(f"[visualize_plan] Added last_plan attribute to consultant state")
                        
                        # Сохраняем оригинальный план
                        if isinstance(strategy, str):
                            consultant.state.last_plan = strategy
                        else:
                            consultant.state.last_plan = json.dumps(strategy, ensure_ascii=False)
                            
                        print(f"[visualize_plan] Plan saved in consultant state: length={len(consultant.state.last_plan)}")
                        break
        except Exception as e:
            print(f"[visualize_plan] Error while trying to save plan in state: {e}")
        finally:
            del frame  # Явно удаляем ссылку на фрейм
        
        # 1. Если передана строка - парсим её как текстовый план
        if isinstance(strategy, str):
            # Попробуем сначала распарсить как JSON
            try:
                strategy_dict = json.loads(strategy)
                if isinstance(strategy_dict, dict) and "steps" in strategy_dict:
                    strategy = strategy_dict
                    print(f"[visualize_plan] Parsed JSON string into dict with steps")
                else:
                    # Это JSON, но без steps - оборачиваем
                    strategy = {"steps": strategy_dict if isinstance(strategy_dict, list) else [strategy_dict]}
                    print(f"[visualize_plan] Wrapped JSON object as steps")
            except json.JSONDecodeError:
                # Не JSON - парсим как текстовый план с нумерацией
                print(f"[visualize_plan] Parsing as text plan")
                
                # Шаблон: ищем строки начинающиеся с цифры и точки/скобки, возможно с пробелами
                # Например: "1. Шаг первый" или "2) Шаг второй"
                import re
                steps = []
                
                # Сначала разбиваем на строки
                lines = strategy.split('\n')
                current_step = None
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Проверяем, это заголовок шага или продолжение описания
                    step_match = re.match(r'^\s*(\d+)[.):]\s+(.+)$', line)
                    
                    if step_match:
                        # Нашли новый шаг
                        step_num = int(step_match.group(1))
                        step_text = step_match.group(2)
                        
                        # Ищем информацию об агенте и инструменте
                        agent = ""
                        tool = ""
                        args = {}
                        reason = step_text
                        
                        # Проверяем на формат "через agent.tool" или "с помощью agent.tool"
                        tool_match = re.search(r'(?:через|с помощью|используя)\s+([a-z_]+)\.([a-z_]+)', step_text, re.IGNORECASE)
                        if tool_match:
                            agent = tool_match.group(1)
                            tool = tool_match.group(2)
                            
                        # Ищем параметры в квадратных или круглых скобках
                        args_match = re.search(r'[\[\(]([^\]\)]+)[\]\)]', step_text)
                        if args_match:
                            args_text = args_match.group(1)
                            # Разбиваем на пары ключ-значение
                            for pair in args_text.split(','):
                                if ':' in pair:
                                    k, v = pair.split(':', 1)
                                    args[k.strip()] = v.strip().strip('"\'')
                        
                        current_step = {
                            "step": step_num,
                            "agent": agent,
                            "tool": tool,
                            "args": args,
                            "reason": reason
                        }
                        steps.append(current_step)
                    elif current_step:
                        # Это продолжение описания текущего шага
                        current_step["reason"] += " " + line
                
                strategy = {"steps": steps}
                print(f"[visualize_plan] Extracted {len(steps)} steps from text plan")
        
        # 2. Обработка списка или других нестандартных форматов
        if not isinstance(strategy, dict) or "steps" not in strategy:
            # Если передан нестандартный формат (например, первый аргумент из args), 
            # попробуем восстановить из него корректную структуру
            if isinstance(strategy, list) and len(strategy) > 0:
                if isinstance(strategy[0], dict) and "steps" in strategy[0]:
                    strategy = strategy[0]  # Извлекаем вложенную стратегию из списка
                    print(f"[visualize_plan] Extracted nested strategy from list")
                else:
                    # Оборачиваем полученный список в словарь со steps
                    strategy = {"steps": strategy}
                    print(f"[visualize_plan] Wrapped list as strategy")
            else:
                # Крайний случай - создаем пустой план
                strategy = {"steps": []}
                print(f"[visualize_plan] Created empty plan (couldn't parse input)")
        
        # 3. Нормализация шагов
        steps = strategy.get("steps", [])
        if not steps:
            return "Нет шагов для визуализации."
            
        # Проверяем формат шагов и нормализуем их если нужно
        normalized_steps = []
        for idx, s in enumerate(steps, start=1):
            if not isinstance(s, dict):
                print(f"[visualize_plan] Skipping invalid step (not a dict): {s}")
                continue
                
            # Проверяем наличие основных полей и нормализуем их
            step_no = s.get("step", idx)
            if isinstance(step_no, str) and not step_no.isdigit():
                # Если это текстовое описание шага вместо номера
                s["reason"] = s.get("reason", step_no)  # Сохраняем текст как причину
                step_no = idx  # Нумеруем последовательно
                
            agent = s.get("agent", "")
            tool_name = s.get("tool", "")
            
            # Обрабатываем случай, когда инструмент указан в формате "agent.tool"
            if "." in agent and not tool_name:
                parts = agent.split(".", 1)
                agent = parts[0]
                tool_name = parts[1]
                print(f"[visualize_plan] Split agent.tool format: agent={agent}, tool={tool_name}")
            
            # Обработка аргументов из разных возможных полей
            args = s.get("args", s.get("params", {}))
                
            # Создаем нормализованный шаг
            normalized_step = {
                "step": step_no,
                "agent": agent,
                "tool": tool_name,
                "args": args,
                "reason": s.get("reason", "")
            }
            normalized_steps.append(normalized_step)
            
        print(f"[visualize_plan] Normalized Steps: {normalized_steps}")
            
        # MARKDOWN-ТАБЛИЦА
        if format.lower() == "markdown":
            header = (
                "| Step | Agent            | Tool               | Args                      | Reason                   |\n"
                "|------|------------------|--------------------|---------------------------|--------------------------|\n"
            )
            rows = []
            for s in normalized_steps:
                step_no = s.get("step", "")
                agent = s.get("agent", "")
                tool_name = s.get("tool", "")
                args = json.dumps(s.get("args", {}), ensure_ascii=False)
                reason = s.get("reason", "").replace("\n", " ")
                rows.append(
                    f"| {step_no}    | {agent:<16} | {tool_name:<18} | {args:<25} | {reason:<24} |"
                )
            visualization = header + "\n".join(rows)
        elif format.lower() == "mermaid":
            lines = ["graph TD"] # Use graph TD for top-down flowchart
            # создаём ноды
            for s in normalized_steps:
                node_id = f"step{s['step']}"
                label = (
                    f"{s['step']}. {s['agent']}\\n"
                    f"{s['tool']}\\n"
                    f"{json.dumps(s.get('args', {}), ensure_ascii=False)}"
                )
                # экранируем кавычки
                lines.append(f'{node_id}["{label}"]')
            # соединяем их стрелками
            for i in range(len(normalized_steps) - 1):
                src = f"step{normalized_steps[i]['step']}"
                dst = f"step{normalized_steps[i+1]['step']}"
                lines.append(f"{src} --> {dst}")
            visualization = "\n".join(lines)
        else:
            error_msg = f"Неизвестный формат визуализации: {format!r}. Поддерживаются 'markdown' и 'mermaid'."
            print(f"[visualize_plan] Error: {error_msg}") # Log
            raise ValueError(error_msg)
            
        print(f"[visualize_plan] Normalized Steps: {len(normalized_steps)}") # Log
        print(f"[visualize_plan] Format: {format}") # Log
        print(f"[visualize_plan] Output: {visualization[:100]}...") # Log snippet
        
        # Автоматически отправляем план супервизору, если установлен флаг
        if auto_send:
            from inspect import currentframe, getouterframes
            frame = currentframe()
            try:
                # Ищем экземпляр MultiAgentSystem в стеке вызовов
                send_plan_result = None
                for f in getouterframes(frame):
                    if 'self' in f.frame.f_locals and isinstance(f.frame.f_locals['self'], MultiAgentSystem):
                        mas_instance = f.frame.f_locals['self']
                        # Вызываем send_plan_to_supervisor
                        send_plan_result = mas_instance.send_plan_to_supervisor(strategy)
                        print(f"[visualize_plan] Plan automatically sent to supervisor: {send_plan_result}")
                        
                        # Добавляем информацию о передаче плана к визуализации
                        visualization += "\n\n### План автоматически отправлен супервизору для выполнения."
                        
                        break
                        
                if not send_plan_result:
                    visualization += "\n\n### Не удалось автоматически отправить план супервизору. Используйте функцию send_plan_to_supervisor."
            except Exception as e:
                print(f"[visualize_plan] Error while trying to auto-send plan: {e}")
                visualization += f"\n\n### Ошибка при автоматической отправке плана: {str(e)}"
            finally:
                del frame  # Явно удаляем ссылку на фрейм
        
        return visualization


    @tool(description="Оценивает результаты от агентов, проверяет полноту, выявляет риски и дает summary.")
    def validate_results(
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
            response = llm([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            content = response.content.strip()

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
            # 5) Фоллбэк при ошибках
            print(f"[validate_results] Input Results Keys: {list(results.keys())}") # Log keys
            print(f"[validate_results] LLM Call Error: {e}") # Log
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

    @tool(description="Предлагает альтернативный план, если шаги оригинальной стратегии не удалось выполнить.")
    def fallback_strategy(
        failed_steps: List[Dict[str, Any]],
        original_strategy: Dict[str, Any],
        agent_mapping: Dict[str, Any] # Добавим карту агентов для LLM
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
        print(f"[fallback_strategy] Original Strategy Steps: {len(original_strategy.get('steps', []))}") # Log
        print(f"[fallback_strategy] Failed Steps: {failed_steps}") # Log
        print(f"[fallback_strategy] Agent Mapping Keys: {list(agent_mapping.keys())}") # Log
        
        # Если нет упавших шагов - fallback не нужен
        if not failed_steps:
            fallback_plan = {
                "alternative_steps": original_strategy.get("steps", []),
                "note": "Не обнаружено сбойных шагов - возвращён оригинальный план.",
                "generated_at": datetime.utcnow().isoformat()
            }
            print(f"[fallback_strategy] Generated Fallback Plan (Simple): {fallback_plan}") # Log
            return fallback_plan

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
            resp = llm([
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ])
            content = resp.content.strip()
            plan = json.loads(content)

            # Проверка структуры
            if not isinstance(plan, dict) or "alternative_steps" not in plan:
                raise ValueError("В ответе нет ключа 'alternative_steps'")
            # Добавляем timestamp
            plan["generated_at"] = datetime.utcnow().isoformat()
            print(f"[fallback_strategy] Generated Fallback Plan (LLM): {plan}") # Log
            return plan

        except Exception as e:
            print(f"[fallback_strategy] LLM Call Error: {e}") # Log
            # Фоллбэк: простой вариант - пропускаем упавшие шаги и смещаем номера
            orig_steps = original_strategy.get("steps", [])
            failed_step_nums = {f.get("step") for f in failed_steps}
            ok_steps = [s for s in orig_steps if s.get("step") not in failed_step_nums]
            # перенумеруем
            for idx, s in enumerate(ok_steps, start=1):
                s["step"] = idx

            fallback_plan = {
                "alternative_steps": ok_steps,
                "note": (
                    "Не удалось сгенерировать расширенный fallback через LLM. "
                    f"Удалены упавшие шаги: {[f['step'] for f in failed_steps]}. "
                    "Оставшиеся шаги перенумерованы."
                ),
                "generated_at": datetime.utcnow().isoformat(),
                "error": str(e)
            }
            print(f"[fallback_strategy] Generated Fallback Plan (Simple): {fallback_plan}") # Log
            return fallback_plan


    def create_consultant_agent(self):
        """
        Создаёт агента-консультанта, который:
        1) берет у системы карту агентов и их инструментов,
        2) задает уточняющие вопросы (если нужно),
        3) строит пошаговую стратегию анализа,
        4) визуализирует её,
        5) при необходимости валидирует результаты и
        6) предлагает запасной план при сбоях.
        """
        # Инструмент для получения метаданных по всем агентам
        if not self._agent_mapping_tool:
             print("[MultiAgentSystem.create_consultant_agent] Error: _agent_mapping_tool not registered!")
             # Handle error appropriately, maybe raise exception or return
             return 

        # Формируем инструменты консультанта: mapper + наши обёртки
        base_tools = [
            self._agent_mapping_tool, # Всегда первый
            self.clarify_user_goal,    # Если нужно уточнить
            self.save_text_plan,       # Для сохранения текстового плана 
            self.visualize_plan,       # Для отображения плана
            self.send_plan_to_supervisor  # Новый инструмент для отправки плана супервизору
        ]
        consultant_tools = []
        # Оборачиваем ВСЕ инструменты, кроме _agent_mapping_tool (если он есть)
        if self._agent_mapping_tool:
            consultant_tools.append(self._agent_mapping_tool)
        for tool_func in base_tools[1:]: # Пропускаем _agent_mapping_tool, если он был первым
            wrapped_tool = self._wrap_consultant_tool(tool_func)
            consultant_tools.append(wrapped_tool)

        # Очень подробный system_prompt - диктует логику работы консультанта
        consultant_system_prompt = """
        Ты - AI-консультант по планированию анализа криптовалют. 
        
        **СТРОГИЕ ПРАВИЛА РАБОТЫ:**
        
        1. У тебя доступны ТОЛЬКО 5 инструментов:
           - get_agent_tool_mapping
           - clarify_user_goal
           - save_text_plan
           - visualize_plan
           - send_plan_to_supervisor
        
        2. Ты НЕ МОЖЕШЬ вызывать никакие другие инструменты, такие как get_token_price, 
           get_market_info и т.д. - они доступны только для специализированных агентов.
        
        3. Твоя ЕДИНСТВЕННАЯ задача - составить ТЕКСТОВЫЙ ПЛАН действий и передать его 
           супервизору, который уже выполнит эти действия.

        **РАБОЧИЙ ПРОЦЕСС:**

        1. Получи список доступных агентов и их инструментов с помощью `get_agent_tool_mapping()`. 
           Используй эту информацию ТОЛЬКО для планирования.
        
        2. Если запрос пользователя неясен, используй `clarify_user_goal(user_query="текст запроса")` 
           для получения уточняющих вопросов.
        
        3. Составь текстовый план анализа в виде пронумерованных шагов. Например:
           ```
           План анализа BTC:
           1. Получить текущую цену BTC через market_analyst.get_token_price
           2. Получить исторические данные через technical_analyst.get_token_historical_data
           3. Проанализировать новости через news_researcher.get_crypto_news
           ```
           
        4. Сохрани созданный план с помощью `save_text_plan(plan="текст плана")`
        
        5. Визуализируй план с помощью `visualize_plan(strategy="текст плана")`
        
        6. Спроси пользователя, хочет ли он отправить план супервизору для выполнения
        
        7. Если пользователь отвечает утвердительно ("Да", "Давай", "Согласен", и т.д.), 
           отправь план супервизору с помощью `send_plan_to_supervisor(plan="текст плана")`
        
        **ОБРАБОТКА УТВЕРДИТЕЛЬНЫХ ОТВЕТОВ:**
        
        Если пользователь пишет "Да", "Давай", "Согласен" и т.п. после того, как ты
        показал план, это означает, что нужно:
        1. Визуализировать план с помощью `visualize_plan`
        2. Отправить его супервизору с помощью `send_plan_to_supervisor`
        
        **ЗАПРЕЩЕНО И БЕСПОЛЕЗНО:**
        
        - Вызывать инструменты других агентов (get_token_price, get_crypto_news и т.д.)
        - Пытаться самостоятельно анализировать данные
        - Выполнять план вместо его передачи супервизору
        
        **ВАЖНО: Вызывай только инструменты из списка доступных 5 инструментов!**
        
        Если пользователь вводит только название токена (например, "BTC"), это означает, 
        что нужно создать план его базового анализа, а не пытаться сразу получить данные.
        """

        # Создаем экземпляр ConsultationState с явно инициализированным last_plan
        initial_state = ConsultationState(last_plan="")
        print(f"[MultiAgentSystem.create_consultant_agent] Created initial consultation state with empty last_plan")

        # Регистрируем агента
        self.agents["consultant"] = CryptoAgent(
            agent_id="consultant",
            role=AgentRole.CONSULTANT,
            system_prompt=consultant_system_prompt,
            tools=consultant_tools,
            state=initial_state
        )
        print(f"[MultiAgentSystem.create_consultant_agent] Consultant agent created with {len(consultant_tools)} wrapped tools.") # Log
    
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
    
    
    # Метод для сохранения текущего плана в состоянии консультанта
    def save_plan_to_consultant_state(self, plan: Union[Dict, List, str]) -> bool:
        """
        Сохраняет план в состоянии консультанта для последующего использования.
        
        Args:
            plan: План в виде строки, словаря или списка
            
        Returns:
            bool: True если план успешно сохранен, иначе False
        """
        try:
            if 'consultant' not in self.agents:
                print(f"[save_plan_to_consultant_state] Error: consultant agent not found")
                return False
                
            consultant = self.agents['consultant']
            if not hasattr(consultant, 'state') or not hasattr(consultant.state, 'last_plan'):
                print(f"[save_plan_to_consultant_state] Error: consultant state doesn't have last_plan attribute")
                return False
            
            # Сохраняем план в подходящем формате
            if isinstance(plan, str):
                consultant.state.last_plan = plan
            else:
                consultant.state.last_plan = json.dumps(plan, ensure_ascii=False)
                
            print(f"[save_plan_to_consultant_state] Plan saved in consultant state: length={len(consultant.state.last_plan)}")
            return True
            
        except Exception as e:
            print(f"[save_plan_to_consultant_state] Error: {e}")
            return False
    
    # Helper для определения, является ли сообщение утвердительным ответом
    def is_affirmative_response(self, text: str) -> bool:
        """
        Проверяет, является ли текст утвердительным ответом.
        
        Args:
            text: Текст для проверки
            
        Returns:
            bool: True если текст похож на утвердительный ответ
        """
        affirmative_phrases = [
            "да", "давай", "согласен", "выполняй", "начинай", "запускай", 
            "ок", "окей", "хорошо", "отлично", "конечно", "точно",
            "я согласен", "сделай", "сделай это", "делай", "приступай",
            "согласен с планом", "полностью согласен", "верно", "всё верно"
        ]
        
        # Очищаем и нормализуем текст
        cleaned_text = text.lower().strip()
        
        # Проверяем каждую фразу
        for phrase in affirmative_phrases:
            if cleaned_text.startswith(phrase) or cleaned_text == phrase or phrase in cleaned_text:
                return True
                
        return False
        
    async def process_user_input(self, user_input: str) -> str:
        """
        Обрабатывает запрос пользователя, начиная с агента-консультанта.
        
        Args:
            user_input: Запрос пользователя
            
        Returns:
            Ответ от агента (пока что от консультанта)
        """
        print(f"[MultiAgentSystem.process_user_input] Routing input to consultant: '{user_input}'") # Log
        
        # Проверяем, есть ли активные задачи супервизора
        active_supervisor_tasks = [
            task_id for task_id, task in self.tasks.items()
            if task.assigned_agent_id == "supervisor" and task.status == "in_progress"
        ]
        
        # Если есть активные задачи для супервизора, и запрос похож на команду управления задачей
        task_control_keywords = ["статус", "status", "выполни", "execute", "отмени", "cancel", "результат", "result"]
        is_task_control = any(keyword in user_input.lower() for keyword in task_control_keywords)
        
        if active_supervisor_tasks and is_task_control:
            print(f"[MultiAgentSystem.process_user_input] Есть активные задачи супервизора, перенаправляю запрос: {active_supervisor_tasks}")
            
            # Добавляем контекст о текущих задачах к запросу
            tasks_context = "\n".join([
                f"Задача {task_id}: {self.tasks[task_id].title}, статус: {self.tasks[task_id].status}"
                for task_id in active_supervisor_tasks
            ])
            enhanced_input = f"""
            {user_input}
            
            Активные задачи супервизора:
            {tasks_context}
            """
            
            # Перенаправляем запрос супервизору
            supervisor = self.agents["supervisor"]
            return await supervisor.process_user_input(enhanced_input)
        
        consultant = self.agents["consultant"]
        
        # Проверяем, является ли запрос утвердительным ответом
        is_affirmative = self.is_affirmative_response(user_input)
        has_saved_plan = hasattr(consultant.state, 'last_plan') and consultant.state.last_plan
        
        if is_affirmative and has_saved_plan:
            # Если это утвердительный ответ и у консультанта есть сохранённый план,
            # добавляем специальное указание к запросу
            print(f"[MultiAgentSystem.process_user_input] Detected affirmative response, will visualize and send plan to supervisor")
            plan = consultant.state.last_plan
            
            # Визуализируем план
            visualization = self.visualize_plan(plan)
            print(f"[MultiAgentSystem.process_user_input] Plan visualized, sending to supervisor")
            
            # Отправляем план супервизору
            send_result = self.send_plan_to_supervisor(plan)
            
            # Формируем ответ пользователю
            return f"{visualization}\n\n### План успешно отправлен супервизору! {send_result}"
        
        # Пытаемся определить, содержит ли запрос пользователя какой-либо план
        plan_keywords = ["plan", "план", "шаг", "step", "анализ", "strategy", "стратегия"]
        contains_plan = any(keyword in user_input.lower() for keyword in plan_keywords)
        crypto_keywords = ["btc", "bitcoin", "eth", "ethereum", "токен", "монета", "coin", "crypto", "крипто"]  
        contains_crypto = any(keyword in user_input.lower() for keyword in crypto_keywords)
        
        if contains_plan or contains_crypto:
            # Если запрашивается план или упоминается криптовалюта, не сбрасываем состояние
            print(f"[MultiAgentSystem.process_user_input] Request about plan or crypto detected, keeping state")
            return await consultant.process_user_input(user_input)
        
        # Сбрасываем состояние консультанта только при смене темы
        print(f"[MultiAgentSystem.process_user_input] Resetting consultant state before processing new request") # Log
        consultant.reset_state()
        
        # Обрабатываем запрос через консультанта
        return await consultant.process_user_input(user_input)
    
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
        
        # Особая обработка для задач выполнения плана супервизором
        if agent_id == "supervisor" and "план" in task.title.lower():
            print(f"Обработка плана супервизором: {task.task_id}")
            task.status = "in_progress"
            
            try:
                # Парсим план из описания задачи
                plan_data = json.loads(task.description)
                
                # Формируем специальное сообщение для супервизора с планом
                supervisor = self.agents["supervisor"]
                
                # Извлекаем шаги плана
                steps = []
                if "steps" in plan_data:
                    steps = plan_data["steps"]
                elif "text_plan" in plan_data:
                    # Если это текстовый план без структуры, просто отображаем его
                    prompt = f"""
                    Мне нужно, чтобы ты выполнил следующий план анализа:
                    
                    {plan_data.get('text_plan')}
                    
                    Пожалуйста, проанализируй этот план и делегируй соответствующие задачи агентам.
                    Когда все задачи будут выполнены, объедини результаты в общий отчет.
                    """
                    
                    result = await supervisor.process_user_input(prompt)
                    task.result = result
                    task.status = "completed"
                    print(f"План выполнен супервизором: {task.task_id}")
                    task.updated_at = datetime.now()
                    return task.result
                
                # Если есть структурированные шаги, обрабатываем их последовательно
                if steps:
                    print(f"План содержит {len(steps)} шагов")
                    results = {}
                    
                    for step in steps:
                        step_num = step.get("step", 0)
                        agent = step.get("agent", "")
                        tool = step.get("tool", "")
                        args = step.get("args", {})
                        reason = step.get("reason", "")
                        
                        if not agent or not tool:
                            # Пытаемся извлечь агента и инструмент из текста причины
                            import re
                            tool_match = re.search(r'(?:через|с помощью|используя)\s+([a-z_]+)\.([a-z_]+)', reason, re.IGNORECASE)
                            if tool_match:
                                agent = tool_match.group(1)
                                tool = tool_match.group(2)
                        
                        if agent and tool:
                            # Формируем описание задачи для агента
                            task_desc = f"""
                            Выполни шаг {step_num} плана анализа:
                            
                            Используй инструмент {tool} со следующими параметрами:
                            {json.dumps(args, ensure_ascii=False, indent=2)}
                            
                            Причина: {reason}
                            """
                            
                            # Делегируем задачу соответствующему агенту
                            delegate_result = self._create_delegate_task_tool()(
                                agent_id=agent,
                                task_title=f"Шаг {step_num}: {tool}",
                                task_description=task_desc,
                                priority=2
                            )
                            
                            print(f"Шаг {step_num} делегирован: {delegate_result}")
                            
                            # Парсим ID задачи из результата
                            import re
                            task_id_match = re.search(r'ID задачи: ([a-f0-9-]+)', delegate_result)
                            if task_id_match:
                                step_task_id = task_id_match.group(1)
                                
                                # Ждем выполнения задачи
                                await asyncio.sleep(1)  # Небольшая пауза перед проверкой
                                
                                # Проверяем статус до 3 раз
                                for _ in range(3):
                                    status = self._create_check_task_status_tool()(task_id=step_task_id)
                                    if status.get("status") == "completed":
                                        results[f"step_{step_num}"] = status.get("result", "Результат отсутствует")
                                        print(f"Шаг {step_num} выполнен успешно")
                                        break
                                    
                                    # Запускаем выполнение задачи, если ещё не запущена
                                    if status.get("status") == "pending":
                                        await self.execute_task(step_task_id)
                                    
                                    # Ждем завершения
                                    await asyncio.sleep(2)
                            else:
                                results[f"step_{step_num}"] = f"Ошибка при делегировании: {delegate_result}"
                        else:
                            results[f"step_{step_num}"] = f"Невозможно определить агента или инструмент: {reason}"
                    
                    # Объединяем результаты всех шагов
                    prompt = f"""
                    План анализа выполнен. Вот результаты каждого шага:
                    
                    {json.dumps(results, ensure_ascii=False, indent=2)}
                    
                    Пожалуйста, объедини эти результаты в единый отчет с выводами.
                    """
                    
                    final_result = await supervisor.process_user_input(prompt)
                    task.result = final_result
                    task.status = "completed"
                    print(f"План выполнен супервизором: {task.task_id}")
                    task.updated_at = datetime.now()
                    return task.result
                
            except Exception as e:
                task.status = "failed"
                task.result = {"error": str(e)}
                print(f"Ошибка при выполнении плана: {e}")
                task.updated_at = datetime.now()
                return task.result
        
        # Стандартное выполнение для других задач
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
    
    @tool(description="Отправляет готовый план агенту-супервизору для выполнения.")
    def send_plan_to_supervisor(
        plan: Union[Dict, List, str, Tuple] = None,
        args: Any = None  # Добавляем параметр args для совместимости
    ) -> str:
        """
        Отправляет готовый план агенту-супервизору для выполнения.
        Параметры:
        - plan: план в одном из форматов (JSON, текст с шагами, список, кортеж)
        - args: альтернативный способ передачи плана (для совместимости)
        
        Возвращает: Сообщение об успешной передаче
        """
        # Если plan не указан, но передан args, используем его
        if plan is None and args is not None:
            plan = args
            print(f"[send_plan_to_supervisor] Using args as plan: {type(args)}")
            
        # Если plan - кортеж с одним элементом, извлекаем его содержимое
        if isinstance(plan, tuple) and len(plan) == 1:
            plan = plan[0]
            print(f"[send_plan_to_supervisor] Extracted plan from tuple: {type(plan)}")
            
        # Проверяем, является ли план пустым
        is_empty_plan = plan is None or (isinstance(plan, (list, tuple, dict)) and not plan) or (isinstance(plan, str) and not plan.strip())
        
        if is_empty_plan:
            print(f"[send_plan_to_supervisor] План пустой, попробуем найти сохраненный план")
            
            # Попробуем найти сохраненный план в состоянии консультанта
            from inspect import currentframe, getouterframes
            frame = currentframe()
            try:
                for f in getouterframes(frame):
                    if 'self' in f.frame.f_locals and isinstance(f.frame.f_locals['self'], MultiAgentSystem):
                        mas_instance = f.frame.f_locals['self']
                        if 'consultant' in mas_instance.agents:
                            consultant = mas_instance.agents['consultant']
                            
                            # Проверяем наличие state и создаем его при необходимости
                            if not hasattr(consultant, 'state'):
                                consultant.state = ConsultationState(last_plan="")
                                print(f"[send_plan_to_supervisor] Created new state for consultant")
                            
                            # Проверяем наличие атрибута last_plan и создаем его при необходимости
                            if not hasattr(consultant.state, 'last_plan'):
                                consultant.state.last_plan = ""
                                print(f"[send_plan_to_supervisor] Added last_plan attribute to consultant state")
                            
                            # Получаем сохраненный план, если он есть
                            if consultant.state.last_plan:
                                plan = consultant.state.last_plan
                                print(f"[send_plan_to_supervisor] Найден сохраненный план длиной {len(plan)}")
                            else:
                                print(f"[send_plan_to_supervisor] В состоянии консультанта нет сохраненного плана")
                            break
            except Exception as e:
                print(f"[send_plan_to_supervisor] Ошибка при поиске сохраненного плана: {e}")
            finally:
                del frame
        
        if is_empty_plan and (plan is None or (isinstance(plan, str) and not plan.strip())):
            return "Ошибка: не удалось найти план для отправки супервизору"
        
        try:
            print(f"[send_plan_to_supervisor] Передача плана супервизору. Тип плана: {type(plan)}")
            
            # Подготовка плана в стандартный формат
            if isinstance(plan, str):
                # Пробуем сначала разобрать как JSON
                try:
                    plan_data = json.loads(plan)
                except:
                    # Текстовый план - преобразуем в структурированный формат
                    # Шаблон: ищем строки начинающиеся с цифры и точки/скобки
                    import re
                    steps = []
                    
                    # Сначала разбиваем на строки
                    lines = plan.split('\n')
                    current_step = None
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                            
                        # Проверяем, это заголовок шага или продолжение описания
                        step_match = re.match(r'^\s*(\d+)[.):]\s+(.+)$', line)
                        
                        if step_match:
                            # Нашли новый шаг
                            step_num = int(step_match.group(1))
                            step_text = step_match.group(2)
                            
                            # Ищем информацию об агенте и инструменте
                            agent = ""
                            tool = ""
                            args = {}
                            reason = step_text
                            
                            # Проверяем на формат "через agent.tool" или "с помощью agent.tool"
                            tool_match = re.search(r'(?:через|с помощью|используя)\s+([a-z_]+)\.([a-z_]+)', step_text, re.IGNORECASE)
                            if tool_match:
                                agent = tool_match.group(1)
                                tool = tool_match.group(2)
                                
                            # Ищем параметры в квадратных или круглых скобках
                            args_match = re.search(r'[\[\(]([^\]\)]+)[\]\)]', step_text)
                            if args_match:
                                args_text = args_match.group(1)
                                # Разбиваем на пары ключ-значение
                                for pair in args_text.split(','):
                                    if ':' in pair:
                                        k, v = pair.split(':', 1)
                                        args[k.strip()] = v.strip().strip('"\'')
                            
                            current_step = {
                                "step": step_num,
                                "agent": agent,
                                "tool": tool,
                                "args": args,
                                "reason": reason
                            }
                            steps.append(current_step)
                        elif current_step:
                            # Это продолжение описания текущего шага
                            current_step["reason"] += " " + line
                    
                    # Если нашли структурированные шаги, создаем план
                    if steps:
                        plan_data = {"steps": steps}
                        print(f"[send_plan_to_supervisor] Извлечено {len(steps)} шагов из текстового плана")
                    else:
                        # Иначе просто сохраняем текст как есть
                        plan_data = {"text_plan": plan}
            else:
                plan_data = plan
                
            # Сохраняем план в состоянии консультанта
            from inspect import currentframe, getouterframes
            frame = currentframe()
            try:
                for f in getouterframes(frame):
                    if 'self' in f.frame.f_locals and isinstance(f.frame.f_locals['self'], MultiAgentSystem):
                        mas_instance = f.frame.f_locals['self']
                        
                        # 1. Сохраняем план в консультанте
                        if 'consultant' in mas_instance.agents:
                            consultant = mas_instance.agents['consultant']
                            
                            # Проверяем наличие state и создаем его при необходимости
                            if not hasattr(consultant, 'state'):
                                consultant.state = ConsultationState(last_plan="")
                                print(f"[send_plan_to_supervisor] Created new state for consultant")
                            
                            # Проверяем наличие атрибута last_plan и создаем его при необходимости
                            if not hasattr(consultant.state, 'last_plan'):
                                consultant.state.last_plan = ""
                                print(f"[send_plan_to_supervisor] Added last_plan attribute to consultant state")
                            
                            # Сохраняем план
                            if isinstance(plan, str):
                                consultant.state.last_plan = plan
                            else:
                                consultant.state.last_plan = json.dumps(plan, ensure_ascii=False)
                                
                            print(f"[send_plan_to_supervisor] План сохранен в состоянии консультанта")
                        else:
                            print(f"[send_plan_to_supervisor] Не удалось сохранить план в состоянии консультанта - агент не найден")
                        
                        # 2. Создаем задачу для супервизора
                        task = Task(
                            title="Выполнение плана анализа",
                            description=json.dumps(plan_data, ensure_ascii=False),
                            assigned_agent_id="supervisor",
                            priority=1
                        )
                        
                        # Добавляем задачу в систему
                        mas_instance.tasks[task.task_id] = task
                        print(f"[send_plan_to_supervisor] Создана задача для супервизора с ID: {task.task_id}")
                        
                        # 3. Запускаем выполнение задачи в отдельном потоке
                        import threading
                        threading.Thread(
                            target=lambda: asyncio.run(mas_instance.execute_task(task.task_id))
                        ).start()
                        
                        return f"План успешно отправлен супервизору. ID задачи: {task.task_id}"
                        
                        break
            except Exception as e:
                print(f"[send_plan_to_supervisor] Ошибка при создании задачи: {e}")
            finally:
                del frame
                
            return "План успешно отправлен супервизору. Ожидайте выполнения."
            
        except Exception as e:
            error_msg = f"Ошибка при отправке плана супервизору: {str(e)}"
            print(f"[send_plan_to_supervisor] {error_msg}")
            return error_msg

    @tool(description="Сохраняет текстовый план для последующей отправки супервизору.")
    def save_text_plan(
        plan: Union[Dict, List, str, Tuple] = None
    ) -> str:
        """
        Сохраняет текстовый план в состоянии консультанта для последующей отправки супервизору.
        
        Параметры:
        - plan: текстовый план в виде строки или другого конвертируемого типа
        
        Возвращает: Сообщение об успешном сохранении
        """
        if plan is None:
            return "Ошибка: план не может быть пустым"
            
        # Конвертируем план в строку, если он не строка
        if not isinstance(plan, str):
            try:
                # Если это кортеж с одним элементом (часто случается при передаче через args)
                if isinstance(plan, tuple) and len(plan) == 1:
                    item = plan[0]
                    # Если этот элемент сам строка - используем его напрямую
                    if isinstance(item, str):
                        plan = item
                    else:
                        # Иначе преобразуем элемент в JSON
                        plan = json.dumps(item, ensure_ascii=False)
                else:
                    # Преобразуем другие типы в JSON
                    plan = json.dumps(plan, ensure_ascii=False)
                    
                print(f"[save_text_plan] Конвертирован план из {type(plan).__name__} в строку, длина: {len(plan)}")
            except Exception as e:
                print(f"[save_text_plan] Ошибка при конвертации плана: {e}")
                return f"Ошибка: не удалось конвертировать план в строку: {str(e)}"
        
        if not plan:
            return "Ошибка: план не может быть пустой строкой"
            
        print(f"[save_text_plan] Сохранение плана длиной {len(plan)} символов")
        
        # Сохраняем план в состоянии консультанта
        from inspect import currentframe, getouterframes
        frame = currentframe()
        try:
            for f in getouterframes(frame):
                if 'self' in f.frame.f_locals and isinstance(f.frame.f_locals['self'], MultiAgentSystem):
                    mas_instance = f.frame.f_locals['self']
                    if 'consultant' in mas_instance.agents:
                        consultant = mas_instance.agents['consultant']
                        
                        # Проверяем наличие state и создаем его при необходимости
                        if not hasattr(consultant, 'state'):
                            consultant.state = ConsultationState(last_plan="")
                            print(f"[save_text_plan] Created new state for consultant")
                        
                        # Проверяем наличие атрибута last_plan и создаем его при необходимости
                        if not hasattr(consultant.state, 'last_plan'):
                            consultant.state.last_plan = ""
                            print(f"[save_text_plan] Added last_plan attribute to consultant state")
                        
                        # Сохраняем план
                        consultant.state.last_plan = plan
                        print(f"[save_text_plan] План успешно сохранен в состоянии консультанта")
                        
                        return "План успешно сохранен. Теперь вы можете визуализировать его с помощью visualize_plan или отправить супервизору с помощью send_plan_to_supervisor."
                    else:
                        return "Ошибка: агент-консультант не найден"
        except Exception as e:
            print(f"[save_text_plan] Ошибка при сохранении плана: {e}")
            return f"Ошибка при сохранении плана: {str(e)}"
        finally:
            del frame
            
        return "Ошибка: не удалось сохранить план"





# Функция для создания мультиагентной системы
def create_multi_agent_system() -> MultiAgentSystem:
    """Создает и возвращает мультиагентную систему для анализа криптовалют."""
    return MultiAgentSystem()
