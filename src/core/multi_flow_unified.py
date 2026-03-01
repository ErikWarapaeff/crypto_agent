"""
Единый LangGraph-граф мультиагентной системы для анализа криптовалют.

Архитектура:
    START → consultant ⇄ consultant_tools → supervisor → specialist → supervisor → … → END

Ключевые улучшения по сравнению с multi_flow.py:
  - Нет inspect.currentframe() — инструменты используют Command API для обновления State
  - Нет threading.Thread + asyncio.run() — LangGraph сам управляет маршрутизацией
  - Нет самодельной Task-очереди — данные хранятся в OverallState.results
  - Supervisor использует structured output (Pydantic) вместо delegate_task/check_task_status
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from pydantic import BaseModel

from config.settings import LLM_MODEL, LLM_TEMPERATURE
from core.Crypto_agent import AgentRole, CryptoAgent

# Инструменты специалистов (импорты без изменений)
from tools.coingecko_tools import (
    get_token_price,
    get_trending_coins,
    search_cryptocurrencies,
)
from tools.defi_protocol_tools import analyze_pools_geckoterminal, analyze_protocol
from tools.holder_analysis_tools import analyze_token_holders
from tools.hyperliquid_tools import (
    confirm_trade,
    execute_trade,
    get_account_info,
    get_crypto_price,
    get_klines_history,
    get_market_info,
)
from tools.llamafeed_tools import (
    get_crypto_hacks,
    get_crypto_news,
    get_crypto_tweets,
    get_market_summary,
    get_polymarket_data,
    get_project_raises,
    get_token_unlocks,
)
from tools.token_analysis_tools import get_token_historical_data


# ---------------------------------------------------------------------------
# Единый State графа
# ---------------------------------------------------------------------------


def _merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Reducer: объединяет два словаря (b перекрывает a)."""
    return {**a, **b}


class OverallState(TypedDict):
    """Общее состояние для всего графа."""

    messages: Annotated[List[BaseMessage], add_messages]
    plan: Optional[Dict[str, Any]]              # сохраняется через save_text_plan
    results: Annotated[Dict[str, Any], _merge_dicts]  # результаты специалистов
    next: str                                   # сигнал маршрутизации


# ---------------------------------------------------------------------------
# Инструменты консультанта (без inspect.currentframe() и threading)
# ---------------------------------------------------------------------------

_AGENT_TOOL_MAP: Dict[str, List[Dict[str, str]]] = {
    "market_analyst": [
        {"name": "get_token_price",        "description": "Текущая цена токена (CoinGecko)"},
        {"name": "get_trending_coins",     "description": "Трендовые монеты (CoinGecko)"},
        {"name": "search_cryptocurrencies","description": "Поиск криптовалют (CoinGecko)"},
        {"name": "get_crypto_price",       "description": "Цена актива (HyperLiquid)"},
    ],
    "technical_analyst": [
        {"name": "get_token_historical_data", "description": "Исторические данные токена"},
        {"name": "get_klines_history",        "description": "Свечи (HyperLiquid)"},
        {"name": "get_market_info",           "description": "Информация о рынке (HyperLiquid)"},
    ],
    "news_researcher": [
        {"name": "get_crypto_news",     "description": "Новости криптовалют"},
        {"name": "get_crypto_tweets",   "description": "Твиты о криптовалютах"},
        {"name": "get_crypto_hacks",    "description": "Данные о взломах"},
        {"name": "get_token_unlocks",   "description": "Разблокировки токенов"},
        {"name": "get_project_raises",  "description": "Раунды инвестиций"},
        {"name": "get_polymarket_data", "description": "Предсказательный рынок"},
        {"name": "get_market_summary",  "description": "Сводка рынка"},
    ],
    "trader": [
        {"name": "execute_trade",   "description": "Исполнение сделки (HyperLiquid)"},
        {"name": "confirm_trade",   "description": "Подтверждение сделки (HyperLiquid)"},
        {"name": "get_account_info","description": "Информация об аккаунте (HyperLiquid)"},
    ],
    "protocol_analyst": [
        {"name": "analyze_protocol",            "description": "Анализ DeFi-протокола"},
        {"name": "analyze_pools_geckoterminal", "description": "Пулы ликвидности (GeckoTerminal)"},
        {"name": "analyze_token_holders",       "description": "Анализ холдеров токена"},
    ],
}


@tool
def get_agent_tool_mapping() -> Dict[str, Any]:
    """
    Возвращает карту всех агентов системы и их инструментов.
    Консультант использует эту информацию ТОЛЬКО для планирования.
    Эти инструменты недоступны консультанту напрямую.
    """
    return {
        "warning": (
            "Инструменты других агентов НЕДОСТУПНЫ консультанту! "
            "Используй только для составления плана."
        ),
        "available_tools": [
            "get_agent_tool_mapping",
            "clarify_user_goal",
            "save_text_plan",
            "visualize_plan",
            "send_plan_to_supervisor",
        ],
        "agents": _AGENT_TOOL_MAP,
    }


@tool
def clarify_user_goal(user_query: str, agent_mapping: Dict[str, Any]) -> List[str]:
    """
    Генерирует список уточняющих вопросов на основе запроса пользователя
    и карты доступных агентов/инструментов.

    Параметры:
    - user_query: исходный запрос пользователя
    - agent_mapping: результат get_agent_tool_mapping()

    Возвращает: список строк с вопросами (пустой список, если всё ясно).
    """
    agents_data = (
        agent_mapping.get("agents", {})
        if isinstance(agent_mapping, dict)
        else agent_mapping
    )

    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0.0,
        base_url="https://api.vsegpt.ru/v1",
    )

    system_prompt = (
        "Ты — ассистент, генерирующий уточняющие вопросы для составления "
        "пошагового плана анализа. Анализируй запрос пользователя и список "
        "доступных инструментов, выявляй недостающие детали. "
        "Выводи ЧИСТЫЙ JSON-массив строк. Если вопросов нет — верни []."
    )
    user_prompt = (
        f"Задача пользователя:\n{json.dumps(user_query, ensure_ascii=False)}\n\n"
        f"Карта агентов:\n{json.dumps(agents_data, indent=2, ensure_ascii=False)}"
    )

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ])

    try:
        questions = json.loads(response.content)
        if isinstance(questions, list):
            return questions
    except Exception:
        pass

    return ["Пожалуйста, уточните ваш запрос — что именно хотите проанализировать?"]


@tool
def visualize_plan(strategy: str, format: str = "markdown") -> str:
    """
    Форматирует план анализа в читаемый вид.

    Параметры:
    - strategy: JSON-строка с планом {"steps": [...]} или текстовый план
    - format: "markdown" (таблица) или "mermaid" (flowchart)

    Возвращает: отформатированную строку.
    """
    steps: List[Dict[str, Any]] = []

    try:
        data = json.loads(strategy)
        if isinstance(data, dict) and "steps" in data:
            steps = data["steps"]
        elif isinstance(data, list):
            steps = data
    except (json.JSONDecodeError, TypeError):
        for line in strategy.split("\n"):
            m = re.match(r"^\s*(\d+)[.):\s]+(.+)$", line.strip())
            if m:
                txt = m.group(2)
                agent_name, tool_name = "", ""
                tm = re.search(r"([a-z_]+)\.([a-z_]+)", txt, re.IGNORECASE)
                if tm:
                    agent_name, tool_name = tm.group(1), tm.group(2)
                steps.append({
                    "step": int(m.group(1)),
                    "agent": agent_name,
                    "tool": tool_name,
                    "args": {},
                    "reason": txt,
                })

    if not steps:
        return "Нет шагов для визуализации."

    if format.lower() == "markdown":
        header = (
            "| Шаг | Агент | Инструмент | Аргументы | Описание |\n"
            "|-----|-------|------------|-----------|----------|\n"
        )
        rows = [
            f"| {s.get('step', i + 1)} | {s.get('agent', '')} | {s.get('tool', '')} | "
            f"{json.dumps(s.get('args', {}), ensure_ascii=False)} | {s.get('reason', '')} |"
            for i, s in enumerate(steps)
        ]
        return header + "\n".join(rows)

    # mermaid
    lines = ["graph TD"]
    for i, s in enumerate(steps):
        nid = f"step{s.get('step', i + 1)}"
        label = (
            f"{s.get('step', '?')}. {s.get('agent', '')}\\n"
            f"{s.get('tool', '')}\\n"
            f"{json.dumps(s.get('args', {}), ensure_ascii=False)}"
        )
        lines.append(f'{nid}["{label}"]')
    for i in range(len(steps) - 1):
        src = f"step{steps[i].get('step', i + 1)}"
        dst = f"step{steps[i + 1].get('step', i + 2)}"
        lines.append(f"{src} --> {dst}")
    return "\n".join(lines)


@tool
def save_text_plan(plan: str) -> Command:
    """
    Сохраняет текстовый план в состоянии системы.
    Используй этот инструмент ПЕРЕД send_plan_to_supervisor.

    Параметры:
    - plan: текст плана (нумерованные шаги) или JSON-строка {"steps": [...]}

    Возвращает: Command, обновляющий state["plan"].
    """
    plan_data = _parse_plan(plan)
    return Command(update={"plan": plan_data})


@tool
def send_plan_to_supervisor(plan: str = "") -> Command:
    """
    Завершает работу консультанта и передаёт готовый план супервизору для выполнения.
    Если plan передан — он также сохраняется в state.
    Если plan пустой — используется ранее сохранённый план.

    Параметры:
    - plan: текст или JSON плана (опционально, если уже вызван save_text_plan)
    """
    update: Dict[str, Any] = {"next": "supervisor"}
    if plan and plan.strip():
        update["plan"] = _parse_plan(plan)
    return Command(update=update)


# ---------------------------------------------------------------------------
# Вспомогательная функция парсинга плана
# ---------------------------------------------------------------------------


def _parse_plan(plan: str) -> Dict[str, Any]:
    """Разбирает строку плана в структурированный словарь."""
    try:
        parsed = json.loads(plan)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"steps": parsed}
    except (json.JSONDecodeError, TypeError):
        pass

    steps: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for line in plan.split("\n"):
        m = re.match(r"^\s*(\d+)[.):\s]+(.+)$", line.strip())
        if m:
            txt = m.group(2)
            agent_name, tool_name = "", ""
            tm = re.search(r"([a-z_]+)\.([a-z_]+)", txt, re.IGNORECASE)
            if tm:
                agent_name, tool_name = tm.group(1), tm.group(2)
            current = {
                "step": int(m.group(1)),
                "agent": agent_name,
                "tool": tool_name,
                "args": {},
                "reason": txt,
            }
            steps.append(current)
        elif current and line.strip():
            current["reason"] += " " + line.strip()

    return {"steps": steps, "text_plan": plan}


# ---------------------------------------------------------------------------
# Узел consultant + ToolNode
# ---------------------------------------------------------------------------

_CONSULTANT_TOOLS = [
    get_agent_tool_mapping,
    clarify_user_goal,
    save_text_plan,
    visualize_plan,
    send_plan_to_supervisor,
]

_CONSULTANT_SYSTEM = """Ты — AI-консультант по планированию анализа криптовалют.

У тебя доступны ТОЛЬКО 5 инструментов:
  1. get_agent_tool_mapping   — получить список агентов и их инструментов
  2. clarify_user_goal        — сгенерировать уточняющие вопросы (передай user_query и agent_mapping)
  3. save_text_plan           — сохранить составленный план (текст или JSON)
  4. visualize_plan           — визуализировать план (markdown или mermaid)
  5. send_plan_to_supervisor  — отправить план супервизору для выполнения

Рабочий процесс:
  1. Вызови get_agent_tool_mapping() — узнай доступные агенты/инструменты.
  2. При необходимости — clarify_user_goal(user_query="...", agent_mapping=<результат п.1>).
  3. Составь нумерованный план (агент.инструмент для каждого шага).
  4. Сохрани: save_text_plan(plan="...").
  5. Визуализируй: visualize_plan(strategy="...").
  6. Отправь: send_plan_to_supervisor().

ЗАПРЕЩЕНО вызывать любые инструменты помимо 5 перечисленных выше.
"""


def _make_llm(**kwargs: Any) -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        base_url="https://api.vsegpt.ru/v1",
        **kwargs,
    )


def _build_consultant():
    """Возвращает (consultant_node_fn, consultant_tools_node)."""
    llm_with_tools = _make_llm().bind_tools(_CONSULTANT_TOOLS)

    async def consultant_node(state: OverallState) -> Dict[str, Any]:
        msgs = [SystemMessage(content=_CONSULTANT_SYSTEM)] + list(state["messages"])
        response = await llm_with_tools.ainvoke(msgs)
        return {"messages": [response]}

    return consultant_node, ToolNode(_CONSULTANT_TOOLS)


# ---------------------------------------------------------------------------
# Узел supervisor (structured output → Command goto)
# ---------------------------------------------------------------------------

_SpecialistName = Literal[
    "market_analyst",
    "technical_analyst",
    "news_researcher",
    "trader",
    "protocol_analyst",
    "FINISH",
]


class SupervisorDecision(BaseModel):
    """Решение супервизора: какой агент выполняет следующий шаг."""

    next: _SpecialistName
    task: str       # конкретная задача для агента (пустая при FINISH)
    reasoning: str  # обоснование выбора


_SUPERVISOR_SYSTEM = """Ты — супервизорный агент, координирующий выполнение плана анализа криптовалют.

Доступные специалисты:
  - market_analyst   : цены, тренды       (get_token_price, get_trending_coins, search_cryptocurrencies, get_crypto_price)
  - technical_analyst: теханализ          (get_token_historical_data, get_klines_history, get_market_info)
  - news_researcher  : новости            (get_crypto_news, get_crypto_tweets, get_crypto_hacks, get_token_unlocks, get_project_raises, get_polymarket_data, get_market_summary)
  - trader           : торговые операции  (execute_trade, confirm_trade, get_account_info)
  - protocol_analyst : DeFi-анализ        (analyze_protocol, analyze_pools_geckoterminal, analyze_token_holders)

Алгоритм:
  1. Изучи поле plan (шаги плана от консультанта).
  2. Изучи поле results (уже выполненные шаги).
  3. Выбери следующий невыполненный шаг → верни next=<агент> и task=<конкретную задачу>.
  4. Если все шаги выполнены → next="FINISH", task=краткое резюме всех результатов.

Отвечай строго в формате SupervisorDecision (JSON).
"""


def _build_supervisor():
    """Возвращает supervisor_node_fn."""
    llm_structured = _make_llm().with_structured_output(SupervisorDecision)

    async def supervisor_node(state: OverallState) -> Command:
        plan = state.get("plan") or {}
        results = state.get("results") or {}

        context = (
            f"ПЛАН:\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n\n"
            f"ВЫПОЛНЕННЫЕ РЕЗУЛЬТАТЫ:\n{json.dumps(results, ensure_ascii=False, indent=2)}"
        )

        decision: SupervisorDecision = await llm_structured.ainvoke([
            SystemMessage(content=_SUPERVISOR_SYSTEM),
            HumanMessage(content=context),
        ])

        if decision.next == "FINISH":
            summary = decision.task or "Анализ завершён. Все шаги плана выполнены."
            return Command(
                goto=END,
                update={
                    "next": "FINISH",
                    "messages": [AIMessage(content=summary)],
                },
            )

        task_msg = HumanMessage(
            content=(
                f"Задача: {decision.task}\n\n"
                f"Обоснование выбора: {decision.reasoning}"
            )
        )
        return Command(
            goto=decision.next,
            update={
                "next": decision.next,
                "messages": [task_msg],
            },
        )

    return supervisor_node


# ---------------------------------------------------------------------------
# Узлы специалистов (обёртки над CryptoAgent subgraph)
# ---------------------------------------------------------------------------


def _build_specialist_node(agent: CryptoAgent):
    """
    Оборачивает скомпилированный CryptoAgent-граф (agent.agent) в node-функцию.
    После выполнения накапливает результат в state["results"][agent_id].
    """
    agent_id = agent.agent_id
    compiled = agent.agent

    async def specialist_node(state: OverallState) -> Dict[str, Any]:
        result_state = await compiled.ainvoke({"messages": state["messages"]})
        new_messages: List[BaseMessage] = result_state.get("messages", [])
        last_msg = new_messages[-1] if new_messages else None
        result_content = getattr(last_msg, "content", "") if last_msg else ""
        return {
            "messages": new_messages,
            "results": {agent_id: result_content},
        }

    specialist_node.__name__ = f"{agent_id}_node"
    return specialist_node


# ---------------------------------------------------------------------------
# Маршрутизация
# ---------------------------------------------------------------------------


def _route_consultant(state: OverallState) -> str:
    """После consultant: есть tool_calls → consultant_tools, иначе → END."""
    last = state["messages"][-1] if state["messages"] else None
    if last and getattr(last, "tool_calls", None):
        return "consultant_tools"
    return END


def _route_after_tools(state: OverallState) -> str:
    """
    После consultant_tools:
      - если send_plan_to_supervisor установил next='supervisor' → supervisor
      - иначе → обратно к consultant (продолжаем цикл)
    """
    if state.get("next") == "supervisor":
        return "supervisor"
    return "consultant"


# ---------------------------------------------------------------------------
# Сборка единого графа
# ---------------------------------------------------------------------------

_SPECIALIST_CONFIGS = [
    (
        "market_analyst",
        AgentRole.MARKET_ANALYST,
        (
            "Ты — агент-аналитик рынка. Анализируй текущие цены, тренды и рыночные "
            "показатели криптовалют с помощью доступных инструментов."
        ),
        [get_token_price, get_trending_coins, search_cryptocurrencies, get_crypto_price],
    ),
    (
        "technical_analyst",
        AgentRole.TECHNICAL_ANALYST,
        (
            "Ты — агент технического анализа. Анализируй исторические данные, графики "
            "и технические индикаторы, выявляй паттерны в движении цен."
        ),
        [get_token_historical_data, get_klines_history, get_market_info],
    ),
    (
        "news_researcher",
        AgentRole.NEWS_RESEARCHER,
        (
            "Ты — агент-исследователь новостей. Собирай и анализируй новости, твиты "
            "и события, связанные с криптовалютами."
        ),
        [
            get_crypto_news, get_crypto_tweets, get_crypto_hacks,
            get_token_unlocks, get_project_raises, get_polymarket_data, get_market_summary,
        ],
    ),
    (
        "trader",
        AgentRole.TRADER,
        (
            "Ты — агент-трейдер. Выполняй торговые операции на основе аналитических "
            "данных, учитывай риски и контролируй исполнение сделок."
        ),
        [execute_trade, confirm_trade, get_account_info],
    ),
    (
        "protocol_analyst",
        AgentRole.PROTOCOL_ANALYST,
        (
            "Ты — агент-аналитик протоколов. Анализируй блокчейн-протоколы, пулы "
            "ликвидности и данные о холдерах."
        ),
        [analyze_protocol, analyze_pools_geckoterminal, analyze_token_holders],
    ),
]

_SPECIALIST_IDS = [cfg[0] for cfg in _SPECIALIST_CONFIGS]


def build_graph():
    """Собирает и компилирует единый LangGraph-граф мультиагентной системы."""

    # --- Specialist agents ---
    specialists: Dict[str, CryptoAgent] = {
        agent_id: CryptoAgent(
            agent_id=agent_id,
            role=role,
            system_prompt=prompt,
            tools=tools,
        )
        for agent_id, role, prompt, tools in _SPECIALIST_CONFIGS
    }

    # --- Node builders ---
    consultant_node, consultant_tools_node = _build_consultant()
    supervisor_node = _build_supervisor()

    # --- Graph ---
    g = StateGraph(OverallState)

    g.add_node("consultant",       consultant_node)
    g.add_node("consultant_tools", consultant_tools_node)
    g.add_node("supervisor",       supervisor_node)

    for agent_id, agent in specialists.items():
        g.add_node(agent_id, _build_specialist_node(agent))

    # --- Edges ---
    g.add_edge(START, "consultant")

    g.add_conditional_edges(
        "consultant",
        _route_consultant,
        {"consultant_tools": "consultant_tools", END: END},
    )

    g.add_conditional_edges(
        "consultant_tools",
        _route_after_tools,
        {"supervisor": "supervisor", "consultant": "consultant"},
    )

    # supervisor использует Command(goto=...) — явные рёбра из него не нужны,
    # но LangGraph требует объявить возможные цели для валидации графа
    for agent_id in _SPECIALIST_IDS:
        g.add_edge(agent_id, "supervisor")

    return g.compile()


# ---------------------------------------------------------------------------
# Публичный API — совместимый с new_main.py
# ---------------------------------------------------------------------------


class _AgentProxy:
    """Заглушка агента для display_agents_list в ui/interface.py."""

    def __init__(self, agent_id: str, role: AgentRole) -> None:
        self.agent_id = agent_id
        self.role = role
        self.state = True  # «активен»


class MultiAgentSystem:
    """
    Тонкая обёртка над единым LangGraph-графом.
    Сохраняет совместимость с new_main.py (process_user_input,
    execute_all_pending_tasks, get_task_status, agents, tasks).
    """

    def __init__(self) -> None:
        self.graph = build_graph()
        self.tasks: Dict[str, Any] = {}  # пуст — задачи живут внутри ainvoke

        # Для display_agents_list
        self.agents: Dict[str, _AgentProxy] = {
            "consultant":       _AgentProxy("consultant",       AgentRole.CONSULTANT),
            "supervisor":       _AgentProxy("supervisor",       AgentRole.SUPERVISOR),
            "market_analyst":   _AgentProxy("market_analyst",   AgentRole.MARKET_ANALYST),
            "technical_analyst":_AgentProxy("technical_analyst",AgentRole.TECHNICAL_ANALYST),
            "news_researcher":  _AgentProxy("news_researcher",  AgentRole.NEWS_RESEARCHER),
            "trader":           _AgentProxy("trader",           AgentRole.TRADER),
            "protocol_analyst": _AgentProxy("protocol_analyst", AgentRole.PROTOCOL_ANALYST),
        }

    async def process_user_input(self, user_input: str) -> str:
        """Обрабатывает запрос пользователя через единый граф."""
        final_state = await self.graph.ainvoke({
            "messages": [HumanMessage(content=user_input)],
            "plan":     None,
            "results":  {},
            "next":     "",
        })
        messages: List[BaseMessage] = final_state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content
        return "Ответ не получен."

    async def execute_all_pending_tasks(self) -> List[Dict[str, Any]]:
        """
        В едином графе задачи выполняются внутри process_user_input.
        Метод оставлен для совместимости с new_main.py.
        """
        return []

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        return {"info": "В единой архитектуре задачи не хранятся отдельно."}

    def reset_system(self) -> None:
        """Состояние графа не персистентно между вызовами ainvoke — ничего не нужно."""


def create_multi_agent_system() -> MultiAgentSystem:
    """Создаёт мультиагентную систему на основе единого LangGraph-графа."""
    return MultiAgentSystem()
