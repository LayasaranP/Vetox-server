from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

prompt = "You are a math solver. At the end of the solution give some follow-up questions or suggestions."

maths_llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ1"),
    base_url="https://api.groq.com/openai/v1",
    stream_usage=True,
    reasoning_effort="high",
    max_retries=2,
    temperature=0.6,
    top_p=0.95,
)

client = TavilyClient(os.getenv("WEB"))


@tool("basic_web_search",
      description="Perform a web search using a search engine to retrieve relevant results based on the provided "
                  "query. Supports basic search operators. Ideal for fact-checking, research, or chaining with "
                  "browse tools.")
def basic_web_search(query: str) -> str:
    """
    Perform a web search using a search engine to retrieve relevant results based on the provided query.

    This tool enables searching the internet for information. It supports basic search operators
    (e.g., site:reddit.com, filetype:pdf, "exact phrase") to refine results and improve precision.
    Results include titles, URLs, and brief snippets for quick assessment.
   """

    response = client.search(
        query=query,
        auto_parameters=True,
        max_results=6
    )

    contents = ""
    for res in response["results"]:
        contents = res['content']

    return contents


maths_agent = create_agent(
    model=maths_llm,
    tools=[basic_web_search],
    system_prompt=prompt,
    middleware=[
        ToolCallLimitMiddleware(
            tool_name="basic_web_search",
            run_limit=5,
        )
    ],
    checkpointer=InMemorySaver()
)


def maths_solver(question: str):
    for token, metadata in maths_agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="messages"
    ):
        if token.content:
            yield token.content
