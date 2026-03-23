import os
import requests
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.messages import SystemMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv(override=True)

MODEL_DEFAULT = "gemini-2.5-flash"
llm = ChatGoogleGenerativeAI(model=MODEL_DEFAULT)

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    if data.get("cod") != 200:
        return f"Could not fetch weather for {city}."
    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    description = data["weather"][0]["description"]
    humidity = data["main"]["humidity"]
    return f"{city}: {description}, {temp}°C (feels like {feels_like}°C), humidity {humidity}%"

@tool 
def get_exchange_rate(base_currency: str, target_currency: str) -> str:
    """Get the current exchange rate between two currencies. 
    Use ISO 4217 currency codes e.g. usd, inr, eur, gbp, jpy."""
    base = base_currency.lower()
    target = target_currency.lower()
    url = f"https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/{base}.json"
    response = requests.get(url)
    if response.status_code != 200:
        return f"Could not fetch exchange rate for {base_currency}."
    data = response.json()
    rate = data.get(base, {}).get(target)
    if rate is None: 
        return f"Could not find exchange rate for {base_currency} to {target_currency}."
    return f"1 {base_currency.upper()} = {rate:.3f} {target_currency.upper()}"

search_tool = TavilySearch(max_results=3)
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000))
tools = [get_weather, get_exchange_rate, search_tool, wiki_tool]
llm_with_tools = llm.bind_tools(tools)

class ChatBotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatBotState) -> dict:
    return {"messages": llm_with_tools.invoke(state["messages"])}

tool_node = ToolNode(tools)

conn = sqlite3.connect("conversations.db", check_same_thread=False)
memory = SqliteSaver(conn)
app = (
    StateGraph(ChatBotState)
    .add_node("chat_node", chat_node)
    .add_node("tools", tool_node)
    .add_edge(START, "chat_node")
    .add_conditional_edges("chat_node", tools_condition)
    .add_edge("tools", "chat_node")
    .compile(checkpointer=memory)
)

if __name__ == "__main__":
    # from uuid import uuid4

    # thread_id = str(uuid4())
    # print(f"thread_id: {thread_id}")

    # while True:
    #     user_message = input("Type here: ")
    #     if user_message.strip().lower() in ["exit", "quit", "bye", ""]:
    #         print("Breaking out of loop")
    #         break
    #     config = {"configurable": {"thread_id": thread_id}}
    #     response = app.invoke(
    #         {"messages": [HumanMessage(content=user_message)]},
    #         config=config,
    #     )
    #     print("AI:", response["messages"][-1].content)

    print(get_weather.invoke({"city": "Delhi"}))
    print(wiki_tool.invoke({"query": "Virat Kohli"}))