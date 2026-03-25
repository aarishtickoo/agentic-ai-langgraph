import os
import requests
import asyncio
import tempfile
from typing import Annotated, TypedDict, Any
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_tavily import TavilySearch
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools import tool, ToolRuntime
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv(override=True)

MODEL_DEFAULT = "gemini-2.5-flash"
EMBEDDING_MODEL_DEFAULT="gemini-embedding-2-preview"

llm = ChatGoogleGenerativeAI(model=MODEL_DEFAULT)
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_DEFAULT)

class ThreadRAGEntry(TypedDict):
    retriever: Any
    filename: str
    num_documents: int
    num_chunks: int

_THREAD_RAG_STORE: dict[str, ThreadRAGEntry] = {}


def ingest_pdf_sync(file_bytes: bytes, thread_id: str, filename: str) -> dict:
    if not file_bytes:
        raise ValueError("No PDF bytes received.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )

        _THREAD_RAG_STORE[thread_id] = {
            "retriever": retriever,
            "filename": filename,
            "num_documents": len(docs),
            "num_chunks": len(chunks),
        }

        return {
            "filename": filename,
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


async def ingest_pdf(file_bytes: bytes, thread_id: str, filename: str) -> dict:
    # heavy blocking work moved off the event loop
    return await asyncio.to_thread(
        ingest_pdf_sync,
        file_bytes,
        thread_id,
        filename,
    )


@tool
def pdf_rag(query: str, runtime: ToolRuntime) -> dict:
    """
    Retrieve relevant context from the uploaded PDF for the current chat thread.
    Use this when the user asks about the uploaded document.
    """
    thread_id = runtime.state["thread_id"]
    retriever = _THREAD_RAG_STORE[thread_id]["retriever"]

    if retriever is None:
        return {
            "error": "No PDF has been indexed for this thread yet.",
            "query": query,
        }

    docs = retriever.invoke(query)

    return {
        "query": query,
        "source_file": _THREAD_RAG_STORE[thread_id]["filename"],
        "context": [doc.page_content for doc in docs],
        "metadata": [doc.metadata for doc in docs],
    }

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "OPENWEATHER_API_KEY is not set."

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url, timeout=15)
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
    Use ISO 4217 currency codes e.g. usd, inr, eur, gbp, jpy.
    """
    base = base_currency.lower()
    target = target_currency.lower()
    url = f"https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/{base}.json"
    response = requests.get(url, timeout=15)

    if response.status_code != 200:
        return f"Could not fetch exchange rate for {base_currency}."

    data = response.json()
    rate = data.get(base, {}).get(target)

    if rate is None:
        return f"Could not find exchange rate for {base_currency} to {target_currency}."

    return f"1 {base_currency.upper()} = {rate:.3f} {target_currency.upper()}"


search_tool = TavilySearch(max_results=3)
wiki_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
)

local_tools = [get_weather, get_exchange_rate, search_tool, wiki_tool, pdf_rag]


async def load_mcp_tools():
    github_pat = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not github_pat:
        raise ValueError("GITHUB_PERSONAL_ACCESS_TOKEN is missing in .env")

    client = MultiServerMCPClient(
        {
            "github": {
                "transport": "streamable_http",
                "url": "https://api.githubcopilot.com/mcp",
                "headers": {
                    "Authorization": f"Bearer {github_pat}"
                },
            }
        }
    )

    return await client.get_tools()


class ChatBotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: str

async def build_app(memory):
    mcp_tools = await load_mcp_tools()

    tools = local_tools + mcp_tools
    llm_with_tools = llm.bind_tools(tools)

    async def chat_node(state: ChatBotState) -> dict:
        system_message = SystemMessage(
            content=(
                "You are a helpful assistant. "
                "If the user's question is about an uploaded PDF or document, use the "
                "`pdf_rag` tool first. "
                "Use web/wiki/MCP tools only when they are more appropriate."
            )
        )

        response = await llm_with_tools.ainvoke(
            [system_message, *state["messages"]]
        )
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    app = (
        StateGraph(ChatBotState)
        .add_node("chat_node", chat_node)
        .add_node("tools", tool_node)
        .add_edge(START, "chat_node")
        .add_conditional_edges("chat_node", tools_condition)
        .add_edge("tools", "chat_node")
        .compile(checkpointer=memory)
    )
    return app
def extract_ai_text(message) -> str:
    content = message.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(part for part in parts if part)

    return str(content)


async def main():
    async with AsyncSqliteSaver.from_conn_string("conversations.db") as memory:
        app = await build_app(memory)

        from uuid import uuid4
        thread_id = str(uuid4())
        print(f"thread_id: {thread_id}")
        print("Use: /pdf /absolute/path/to/file.pdf")
        print("Then ask questions about that PDF.")

        while True:
            user_message = input("Type here: ").strip()

            if user_message.lower() in ["exit", "quit", "bye", ""]:
                print("Breaking out of loop")
                break

            if user_message.startswith("/pdf "):
                pdf_path = user_message[len("/pdf "):].strip()

                if not os.path.exists(pdf_path):
                    print("PDF path does not exist.")
                    continue

                with open(pdf_path, "rb") as f:
                    summary = await ingest_pdf(
                        file_bytes=f.read(),
                        thread_id=thread_id,
                        filename=os.path.basename(pdf_path),
                    )

                print(
                    f"Indexed PDF: {summary['filename']} | "
                    f"pages={summary['documents']} | chunks={summary['chunks']}"
                )
                continue

            config = {"configurable": {"thread_id": thread_id}}

            response = await app.ainvoke(
                {
                    "messages": [HumanMessage(content=user_message)],
                    "thread_id": thread_id,
                },
                config=config,
            )

            print("AI:", extract_ai_text(response["messages"][-1]))


if __name__ == "__main__":
    asyncio.run(main())