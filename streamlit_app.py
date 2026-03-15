import streamlit as st
from uuid import uuid4
from langchain_core.messages import HumanMessage

from app import app


def add_to_all_threads(thread_id):
    existing_ids = [t["thread_id"] for t in st.session_state.all_threads]
    if thread_id not in existing_ids:
        thread_num = len(st.session_state.all_threads) + 1
        st.session_state.all_threads.append({
            "thread_id": thread_id,
            "name": f"Thread {thread_num}"
        })


def new_thread():
    st.session_state.all_messages[st.session_state.thread_id] = st.session_state.messages
    st.session_state.thread_id = str(uuid4())
    st.session_state.messages = []
    add_to_all_threads(st.session_state.thread_id)


def load_thread(thread_id):
    st.session_state.all_messages[st.session_state.thread_id] = st.session_state.messages
    st.session_state.thread_id = thread_id
    st.session_state.messages = st.session_state.all_messages.get(thread_id, [])


if "all_threads" not in st.session_state:
    st.session_state.all_threads = []

if "all_messages" not in st.session_state:
    st.session_state.all_messages = {}

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())
    add_to_all_threads(st.session_state.thread_id)

if "messages" not in st.session_state:
    st.session_state.messages = []


st.set_page_config(page_title="LangGraph Chatbot", page_icon="🤖")
st.title("🤖 LangGraph Chatbot")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("How can I help you today?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.status("Thinking...")

        def stream_response():
            first_chunk = True
            for chunk, metadata in app.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages"
            ):
                if chunk.content:
                    if first_chunk:
                        placeholder.empty()
                        first_chunk = False
                    yield chunk.content

        ai_content = st.write_stream(stream_response())

    st.session_state.messages.append({"role": "assistant", "content": ai_content})
    st.session_state.all_messages[st.session_state.thread_id] = st.session_state.messages

with st.sidebar:
    # st.markdown(f"### 💬 {next(t['name'] for t in st.session_state.all_threads if t['thread_id'] == st.session_state.thread_id)}")
    if st.button("+ New Conversation"):
        new_thread()
        st.rerun()

    st.markdown("### Conversations")
    for thread in st.session_state.all_threads[::-1]:
        label = f"🟢 {thread['name']}" if thread["thread_id"] == st.session_state.thread_id else thread["name"]
        if st.button(label, key=thread["thread_id"]):
            load_thread(thread["thread_id"])
            st.rerun()