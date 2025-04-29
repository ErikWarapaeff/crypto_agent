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
from typing import Any, Dict, List, Optional
from uuid import uuid4
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from core.Crypto_agent import CryptoAgent, AgentRole


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

    current_step: int = 0
    executed_steps: List[ExecutedStep] = Field(default_factory=list)

    results: Dict[str, Any] = Field(default_factory=dict)

    completed: bool = False
    need_clarification: bool = False
    fallback_applied: bool = False

    history: List[HistoryEvent] = Field(default_factory=list)

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
        self.__dict__.update(ConsultationState(user_profile=profile).dict())

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
            """
            mapping = {}
            for ag_id, agent in self.agents.items():
                tools_info = []
                for fn in agent.tools:
                    tools_info.append({
                        "name": fn.__name__,
                        "description": (fn.__doc__ or "").strip().split("\n")[0]
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
        - Выводи **чистый** JSON-массив строк, без лишнего текста.
        - Если дополнительных вопросов не требуется - верни пустой массив `[]`.
        """
        
        # 3) Соберём сообщение пользователю с картой инструментов
        user_prompt = f"""
        Задача пользователя:
        {json.dumps(user_query, ensure_ascii=False)}
        Карта агентов и инструментария:
        {json.dumps(agent_mapping, indent=2, ensure_ascii=False)}
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
                return questions
        except Exception:
            pass
        
        # В случае ошибки парсинга - возвращаем пустой список
        return []
    
    @tool
    def build_analysis_strategy(
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
        response = llm([
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ])

        # 5) Парсинг результата
        try:
            plan = json.loads(response.content)
            # Проверяем наличие обязательных полей
            if isinstance(plan, dict) and "steps" in plan and "need_clarification" in plan:
                return plan
        except Exception:
            pass

        # 6) Фоллбэк
        return {
            "steps": [],
            "need_clarification": True
        }
        
    @tool
    def visualize_plan(
        strategy: Dict,
        format: str = "markdown"
    ) -> str:
        """
        Преобразует стратегию (output build_analysis_strategy) в красиво отформатированный план.
        Параметры:
        - strategy: {
            "steps": [
                {"step":1, "agent":"market_analyst", "tool":"get_token_price",
                "args":{"symbol":"ETH"}, "reason":"оценить цену"},
                ...
            ],
            "need_clarification": false
        }
        - format: "markdown" (таблица) или "mermaid" (flowchart)

        Возвращает строку с визуализацией.
        """
        steps = strategy.get("steps", [])
        if not steps:
            return "Нет шагов для визуализации."

        # MARKDOWN-ТАБЛИЦА
        if format.lower() == "markdown":
            header = (
                "| Step | Agent            | Tool               | Args                      | Reason                   |\n"
                "|------|------------------|--------------------|---------------------------|--------------------------|\n"
            )
            rows = []
            for s in steps:
                step_no = s.get("step", "")
                agent = s.get("agent", "")
                tool_name = s.get("tool", "")
                args = json.dumps(s.get("args", {}), ensure_ascii=False)
                reason = s.get("reason", "").replace("\n", " ")
                rows.append(
                    f"| {step_no}    | {agent:<16} | {tool_name:<18} | {args:<25} | {reason:<24} |"
                )
            return header + "\n".join(rows)

        # MERMAID FLOWCHART
        elif format.lower() == "mermaid":
            lines = ["```"]
            # создаём ноды
            for s in steps:
                node_id = f"step{s['step']}"
                label = (
                    f"{s['step']}. {s['agent']}\\n"
                    f"{s['tool']}\\n"
                    f"{json.dumps(s.get('args', {}), ensure_ascii=False)}"
                )
                # экранируем кавычки
                lines.append(f'{node_id}["{label}"]')
            # соединяем их стрелками
            for i in range(len(steps) - 1):
                src = f"step{steps[i]['step']}"
                dst = f"step{steps[i+1]['step']}"
                lines.append(f"{src} --> {dst}")
            lines.append("```")
            return "\n".join(lines)

        # Нераспознанный формат
        else:
            raise ValueError(f"Неизвестный формат визуализации: {format!r}. Поддерживаются 'markdown' и 'mermaid'.")


    @tool
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
            return plan

        except Exception as e:
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
        4) визуализирует её,
        5) при необходимости валидирует результаты и
        6) предлагает запасной план при сбоях.
        """
        # Инструмент для получения метаданных по всем агентам
        self._agent_mapping_tool

        # Собираем все инструменты, которые будет использовать консультант
        consultant_tools = [
        self._agent_mapping_tool,
        self.clarify_user_goal,
        self.build_analysis_strategy,
        self.visualize_plan,
        self.validate_results,
        self.fallback_strategy
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
        4) Преобразуй стратегию в удобный вид:
            visualized = visualize_plan(strategy, format="markdown")
        • Отправь visualized пользователю.
        5) После того как пользователь проверил/прокомментировал шаги,
        при запросе «выполнить» или «проверить» вызывай validate_results(results).
        6) Если какой-то шаг упал (пустой результат, ошибка, таймаут), передай failed_steps
        и original_strategy в fallback_strategy и снова визуализируй альтернативу.

        При любом выводе:
        - Чётко подписывай, какой инструмент ты вызываешь и с какими аргументами.
        - Вся промежуточная логика должна жить в system_prompt, а не в теле кода.
        - Возвращай пользователю **чистый** JSON или markdown - без лишних пояснений вне кода.
        """

        # Регистрируем агента
        self.agents["consultant"] = CryptoAgent(
            agent_id="consultant",
            role=AgentRole.CONSULTANT,
            system_prompt=consultant_system_prompt,
            tools=consultant_tools,
            state=ConsultationState()
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
    





# Функция для создания мультиагентной системы
def create_multi_agent_system() -> MultiAgentSystem:
    """Создает и возвращает мультиагентную систему для анализа криптовалют."""
    return MultiAgentSystem()
