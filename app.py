from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.messages import SystemMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

MODEL_DEFAULT = "gemini-2.5-flash"
llm = ChatGoogleGenerativeAI(model=MODEL_DEFAULT)


class ChatBotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatBotState) -> dict:
    return {"messages": llm.invoke(state["messages"])}


workflow = StateGraph(ChatBotState)
workflow.add_node("chat_node", chat_node)
workflow.add_edge(START, "chat_node")
workflow.add_edge("chat_node", END)

memory = InMemorySaver()
app = workflow.compile(checkpointer=memory)


if __name__ == "__main__":
    from uuid import uuid4

    thread_id = str(uuid4())
    print(f"thread_id: {thread_id}")

    while True:
        user_message = input("Type here: ")
        if user_message.strip().lower() in ["exit", "quit", "bye", ""]:
            print("Breaking out of loop")
            break
        config = {"configurable": {"thread_id": thread_id}}
        response = app.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
        )
        print("AI:", response["messages"][-1].content)