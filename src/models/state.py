"""Модели состояния для криптоаналитического агента."""

from enum import Enum
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator


class MessageRole(str, Enum):
    """Роли сообщений в диалоге."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class MessageContent(BaseModel):
    """Содержимое сообщения с поддержкой разных типов контента."""
    text: str = ""
    content_type: str = "text"
    additional_data: Optional[Dict[str, Any]] = None


class Message(BaseModel):
    """Модель для представления сообщения в диалоге."""
    role: MessageRole
    content: Union[str, MessageContent]
    timestamp: datetime = Field(default_factory=datetime.now)
    message_id: str = Field(default_factory=lambda: f"msg_{datetime.now().timestamp()}")

    @validator("content", pre=True)
    def validate_content(cls, v):
        """Преобразует строковый контент в объект MessageContent."""
        if isinstance(v, str):
            return MessageContent(text=v, content_type="text")
        return v


class ToolCall(BaseModel):
    """Модель вызова инструмента."""
    tool_name: str
    arguments: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    call_id: str = Field(default_factory=lambda: f"tool_{datetime.now().timestamp()}")


class ToolResult(BaseModel):
    """Результат выполнения инструмента."""
    call_id: str
    tool_name: str
    result: Any
    success: bool = True
    error: Optional[str] = None
    execution_time: float = 0.0


class AgentState(BaseModel):
    """Основная модель состояния агента."""
    conversation_id: str = Field(default_factory=lambda: f"conv_{datetime.now().timestamp()}")
    messages: List[Message] = Field(default_factory=list)
    tool_calls: List[ToolCall] = Field(default_factory=list)
    tool_results: List[ToolResult] = Field(default_factory=list)
    current_context: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Конфигурация модели."""
        validate_assignment = True

    def add_user_message(self, content: str) -> Message:
        """Добавляет сообщение пользователя в историю."""
        message = Message(role=MessageRole.USER, content=content)
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message

    def add_assistant_message(self, content: str) -> Message:
        """Добавляет сообщение ассистента в историю."""
        message = Message(role=MessageRole.ASSISTANT, content=content)
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message

    def add_system_message(self, content: str) -> Message:
        """Добавляет системное сообщение в историю."""
        message = Message(role=MessageRole.SYSTEM, content=content)
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message

    def add_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> ToolCall:
        """Регистрирует вызов инструмента."""
        tool_call = ToolCall(tool_name=tool_name, arguments=arguments)
        self.tool_calls.append(tool_call)
        self.updated_at = datetime.now()
        return tool_call

    def add_tool_result(self, call_id: str, tool_name: str, result: Any,
                        success: bool = True, error: Optional[str] = None,
                        execution_time: float = 0.0) -> ToolResult:
        """Регистрирует результат выполнения инструмента."""
        tool_result = ToolResult(
            call_id=call_id,
            tool_name=tool_name,
            result=result,
            success=success,
            error=error,
            execution_time=execution_time
        )
        self.tool_results.append(tool_result)
        self.updated_at = datetime.now()
        return tool_result

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Возвращает историю диалога в формате, подходящем для LLM."""
        return [
            {"role": msg.role, "content": msg.content.text if isinstance(msg.content, MessageContent) else msg.content}
            for msg in self.messages
        ]

    def get_last_n_messages(self, n: int) -> List[Message]:
        """Возвращает последние N сообщений."""
        return self.messages[-n:] if len(self.messages) >= n else self.messages[:]

    def clear_history(self) -> None:
        """Очищает историю диалога и инструментов."""
        self.messages = []
        self.tool_calls = []
        self.tool_results = []
        self.updated_at = datetime.now()
