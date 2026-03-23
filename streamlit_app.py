import streamlit as st
from uuid import uuid4
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from app import app


def msg_to_dict(msg):
    if isinstance(msg, ToolMessage):
        return None
    if isinstance(msg, AIMessage) and (msg.tool_calls or not msg.content):
        return None
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    content = msg.content if isinstance(msg.content, str) else "".join(
        block["text"] for block in msg.content
        if isinstance(block, dict) and block.get("type") == "text"
    )
    return {"role": role, "content": content}

def load_all_threads_from_db():
    return list(dict.fromkeys(
        cp.config["configurable"]["thread_id"]
        for cp in app.checkpointer.list(None)
    ))

def load_messages_from_db(thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    state = app.get_state(config)
    raw_messages = state.values.get("messages", [])
    return [d for d in (msg_to_dict(m) for m in raw_messages) if d is not None]

def add_to_all_threads(thread_id):
    existing_ids = [t["thread_id"] for t in st.session_state.all_threads]
    if thread_id not in existing_ids:
        st.session_state.all_threads.append({
            "thread_id": thread_id,
            "name": f"Thread: {thread_id}"
        })

def new_thread():
    st.session_state.thread_id = str(uuid4())
    st.session_state.messages = []
    add_to_all_threads(st.session_state.thread_id)

def load_thread(thread_id):
    st.session_state.thread_id = thread_id
    st.session_state.messages = load_messages_from_db(thread_id)


if "all_threads" not in st.session_state:
    thread_ids = load_all_threads_from_db()
    st.session_state.all_threads = []

    if thread_ids:
        for tid in reversed(thread_ids):
            add_to_all_threads(tid)
        most_recent = thread_ids[0]
        st.session_state.thread_id = most_recent
        st.session_state.messages = load_messages_from_db(most_recent)
    else:
        st.session_state.thread_id = str(uuid4())
        st.session_state.messages = []
        add_to_all_threads(st.session_state.thread_id)


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

    config = {
        "configurable": {"thread_id": st.session_state.thread_id},
        "metadata": {"thread_id": st.session_state.thread_id},
        "run_name": "chat_run",
    }

    with st.chat_message("assistant"):
        thinking = st.empty()
        thinking.status("Thinking...")
        status_holder = {"box": None}

        def stream_response():
            for chunk, metadata in app.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                if isinstance(chunk, ToolMessage):
                    thinking.empty()
                    tool_name = getattr(chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(f"🔧 Using `{tool_name}`", expanded=True)
                    else:
                        status_holder["box"].update(label=f"🔧 Using `{tool_name}`", state="running", expanded=True)

                if isinstance(chunk, AIMessage) and chunk.content:
                    thinking.empty()
                    content = chunk.content if isinstance(chunk.content, str) else "".join(
                        block["text"] for block in chunk.content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                    if content:
                        yield content

        ai_content = st.write_stream(stream_response())

        if status_holder["box"] is not None:
            status_holder["box"].update(label="✅ Tool usage finished", state="complete", expanded=False)

    st.session_state.messages.append({"role": "assistant", "content": ai_content})

with st.sidebar:
    if st.button("+ New Conversation"):
        new_thread()
        st.rerun()

    st.markdown("### Conversations")
    for thread in st.session_state.all_threads[::-1]:
        label = f"🟢 {thread['name']}" if thread["thread_id"] == st.session_state.thread_id else thread["name"]
        if st.button(label, key=thread["thread_id"]):
            load_thread(thread["thread_id"])
            st.rerun()