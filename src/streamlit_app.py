import streamlit as st
import asyncio
from datetime import datetime
import re

from config.settings import setup_environment, APP_NAME, APP_COLOR, OPENAI_API_KEY
from core.agent import create_agent

# Инициализация окружения
setup_environment()

# Настройка страницы
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🚀",
    layout="wide",
)

# Установка переменных состояния сессии
if 'agent' not in st.session_state:
    st.session_state.agent = create_agent()

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'thinking' not in st.session_state:
    st.session_state.thinking = False
    
if 'chats' not in st.session_state:
    # Структура: {chat_id: {title: "Название чата", messages: [список сообщений]}}
    st.session_state.chats = {
        "default": {"title": "Новый чат", "messages": st.session_state.messages.copy() if 'messages' in st.session_state else []}
    }

if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = "default"

if 'chat_counter' not in st.session_state:
    st.session_state.chat_counter = 1
    
if 'chat_to_rename' not in st.session_state:
    st.session_state.chat_to_rename = None

# CSS для улучшения внешнего вида
st.markdown("""
<style>
    
    /* Стили для формы переименования */
    .sidebar .stForm {
        background-color: #f1f3f4;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-bottom: 0.5rem;
    }
    
    .sidebar .stForm .stButton {
        margin-top: 0;
    }
    
    .sidebar .stTextInput > div > div > input {
        font-size: 0.9rem;
        padding: 0.3rem;
    }
    
    /* Стили для сайдбара */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Стили для кнопок в сайдбаре */
    .sidebar .stButton > button {
        background-color: transparent;
        border: none;
        text-align: left;
        padding: 0.5rem 0;
        color: #333;
        width: 100%;
    }
    
    .sidebar .stButton > button:hover {
        background-color: #e9ecef;
        border-radius: 0.3rem;
    }
    
    /* Основные стили */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Стили для кнопок */
    .stButton > button {
        width: 100%;
        height: 2.75rem;
        padding: 0 0.5rem;
        white-space: nowrap;
    }

    /* Одинаковые колонки для кнопок */
    .button-cols {
        min-width: 12rem;
    }

    /* Стили сообщений */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        display: flex;
        flex-direction: column;
    }

    .chat-message.user {
        background-color: #E3F2FD;
        border-left: 4px solid #1E88E5;
        margin-left: 60px;
    }

    .chat-message.assistant {
        background-color: #F5F5F5;
        border-left: 4px solid #7E57C2;
        margin-right: 60px;
    }

    .chat-message.system {
        background-color: #FFF8E1;
        border-left: 4px solid #FFC107;
    }

    .message-content {
        display: flex;
        margin-bottom: 0.5rem;
    }

    .message-content img {
        width: 2.5rem;
        height: 2.5rem;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }

    .message-content p {
        margin: 0;
    }

    /* Прикрепить поле ввода к низу */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 1rem;
        background-color: white;
        z-index: 100;
        border-top: 1px solid #ddd;
    }

    /* Добавить отступ для сообщений, чтобы они не перекрывались с полем ввода */
    .chat-window {
        margin-bottom: 5rem;
    }

    /* Стили для спиннера в области ввода */
        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-left: 10px;
            border: 3px solid rgba(0, 0, 0, 0.1);
            border-radius: 50%;
            border-top-color: #2196F3;
            animation: spin 1s ease-in-out infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    .input-with-spinner {
        display: flex;
        align-items: center;
    }

    .status-message {
        margin-left: 10px;
        color: #2196F3;
        font-size: 0.9em;
    }

    /* Скрыть стандартные элементы streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def create_new_chat():
    chat_id = f"chat_{st.session_state.chat_counter}"
    st.session_state.chat_counter += 1
    st.session_state.chats[chat_id] = {
        "title": f"Новый чат {st.session_state.chat_counter}", 
        "messages": []
    }
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = st.session_state.chats[chat_id]["messages"]
    st.rerun()

def switch_chat(chat_id):
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = st.session_state.chats[chat_id]["messages"]
    st.rerun()

def delete_chat(chat_id):
    if len(st.session_state.chats) > 1:
        del st.session_state.chats[chat_id]
        # Переключаемся на первый доступный чат
        st.session_state.current_chat_id = next(iter(st.session_state.chats.keys()))
        st.session_state.messages = st.session_state.chats[st.session_state.current_chat_id]["messages"]
    st.rerun()

def rename_chat(chat_id, new_title):
    if new_title and new_title.strip():
        st.session_state.chats[chat_id]["title"] = new_title.strip()
    st.session_state.chat_to_rename = None
    st.rerun()

# Функция для асинхронной обработки сообщений
async def process_message(message):
    try:
        return await st.session_state.agent.process_user_input(message)
    except Exception as e:
        return f"Произошла ошибка при обработке запроса: {str(e)}"

def process_pending_request():
    if st.session_state.thinking and hasattr(st.session_state, 'current_question'):
        with st.spinner("ИИ обрабатывает ваш запрос..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                response = loop.run_until_complete(process_message(st.session_state.current_question))
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.session_state.messages.append({
                    "role": "system",
                    "content": f"⚠️ Произошла ошибка: {str(e)}"
                })
            finally:
                loop.close()
                st.session_state.thinking = False
                if hasattr(st.session_state, 'current_question'):
                    delattr(st.session_state, 'current_question')
                # Синхронизируем сообщения с текущим чатом
                st.session_state.chats[st.session_state.current_chat_id]["messages"] = st.session_state.messages

# Обработка отправки сообщения
def handle_submit():
    user_message = st.session_state.user_input
    
    if user_message.strip():
        # Добавляем сообщение пользователя в историю
        st.session_state.messages.append({"role": "user", "content": user_message})
        # Синхронизируем с текущим чатом
        st.session_state.chats[st.session_state.current_chat_id]["messages"] = st.session_state.messages
        
        # Очищаем поле ввода и устанавливаем флаг thinking
        st.session_state.user_input = ""
        st.session_state.thinking = True
        st.session_state.current_question = user_message

# Заголовок приложения
st.title(APP_NAME)

# Боковая панель с историей чатов
with st.sidebar:
    st.title("История чатов")
    
    # Кнопка для создания нового чата
    if st.button("➕ Новый чат", key="new_chat"):
        create_new_chat()
    
    st.markdown("---")
    
    # Список существующих чатов
    for chat_id, chat_data in st.session_state.chats.items():
        # Если этот чат в режиме редактирования
        if st.session_state.chat_to_rename == chat_id:
            with st.form(key=f"rename_form_{chat_id}", clear_on_submit=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    new_title = st.text_input("Новое название", value=chat_data["title"], 
                                              key=f"new_title_{chat_id}", label_visibility="collapsed")
                with col2:
                    submit_button = st.form_submit_button("✓")
                    
                if submit_button:
                    rename_chat(chat_id, new_title)
        else:
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                # Визуально выделяем текущий чат
                if chat_id == st.session_state.current_chat_id:
                    st.markdown(f"**🟢 {chat_data['title']}**")
                else:
                    if st.button(f"📝 {chat_data['title']}", key=f"select_{chat_id}"):
                        switch_chat(chat_id)
            
            with col2:
                # Кнопка редактирования названия
                if st.button("✏️", key=f"edit_{chat_id}"):
                    st.session_state.chat_to_rename = chat_id
                    st.rerun()
            
            with col3:
                # Кнопка удаления чата (если их больше одного)
                if len(st.session_state.chats) > 1:
                    if st.button("🗑️", key=f"delete_{chat_id}"):
                        delete_chat(chat_id)

process_pending_request()

# Если нет сообщений, показываем приветствие
if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Привет! Я криптоаналитический ассистент. Задайте мне вопрос о криптовалютах, токенах, DeFi или рынке в целом!"
    })

    # Добавляем подсказку о командах
    st.session_state.messages.append({
        "role": "system",
        "content": "💡 **Подсказка**: Вы можете использовать специальные команды, такие как `/research BTC` для глубокого исследования токена."
    })

# Контейнер для сообщений
chat_container = st.container()

# Отображение сообщений
with chat_container:
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        # Определяем иконку для роли
        icon = "👤" if role == "user" else "🤖" if role == "assistant" else "ℹ️"

        # Создаем стилизованный контейнер для сообщения
        with st.container():
            col1, col2 = st.columns([1, 20])
            with col1:
                st.markdown(f"<div style='font-size:1.5rem;'>{icon}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='chat-message {role}'>{content}</div>", unsafe_allow_html=True)

# Разделитель перед полем ввода
st.markdown("<hr>", unsafe_allow_html=True)

# Проверяем статус обработки запроса
if st.session_state.thinking:
    # Отображаем поле ввода с спиннером
    input_col, spinner_col, send_col, clear_col = st.columns([4, 1, 0.8, 0.8])

    with input_col:
        st.text_input(
            "Введите ваш вопрос",
            key="user_input",
            placeholder="Введите запрос или команду...",
            label_visibility="collapsed",
            disabled=True  # Отключаем поле ввода во время обработки
        )

    with spinner_col:
        st.markdown('<div class="loading-spinner"></div><span class="status-message">ИИ думает...</span>', unsafe_allow_html=True)

    with send_col:
        st.button("Отправить", disabled=True)

    with clear_col:
        st.button("Очистить историю", disabled=True)

else:
    # Обычное отображение поля ввода и кнопки
    input_col, send_col, clear_col = st.columns([4, 0.9, 1.2])

    with input_col:
        user_input = st.text_input(
            "Введите ваш вопрос",
            key="user_input",
            placeholder="Введите запрос или команду...",
            label_visibility="collapsed"
        )

    with send_col:
        st.button("Отправить", on_click=handle_submit)

    with clear_col:
        if st.button("Очистить историю"):
            st.session_state.messages = []
            st.rerun()

# Полезные подсказки внизу
with st.expander("Список доступных команд и возможностей"):
    st.markdown("""
    ### Доступные команды:
    - `/research SYMBOL` - глубокое исследование токена
    - `/price SYMBOL` - быстрый запрос цены токена

    ### Примеры запросов:
    - "Каковы текущие тренды на криптовалютном рынке?"
    - "Проанализируй токен ETH"
    - "Что происходит с DeFi проектами?"
    """)

# Всегда прокручиваем вниз при обновлении
st.markdown("""
<script>
    function scroll_to_bottom() {
        window.scrollTo(0, document.body.scrollHeight);
    }
    scroll_to_bottom();
</script>
""", unsafe_allow_html=True)
