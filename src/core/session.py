from models.state import AgentState, Message, MessageRole


class Session:
    """Класс для управления сессией взаимодействия с пользователем."""

    def __init__(self):
        """Инициализирует новую сессию."""
        self.state = AgentState()

    def add_user_message(self, message: str) -> Message:
        """Добавляет сообщение пользователя в историю."""
        return self.state.add_user_message(message)

    def add_assistant_message(self, message: str) -> Message:
        """Добавляет сообщение ассистента в историю."""
        return self.state.add_assistant_message(message)

    def get_messages(self):
        """Возвращает полную историю сообщений."""
        return self.state.get_conversation_history()

    def clear_history(self):
        """Очищает историю сообщений."""
        self.state.clear_history()
        
    def get_state(self) -> AgentState:
        """Возвращает полное состояние сессии."""
        return self.state