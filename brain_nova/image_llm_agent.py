from typing import Tuple, Any, List

from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from tavily import TavilyClient
from langchain_core.tools import tool
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ")

prompt = """# ROLE You are a helpful, precise, and highly organized AI assistant. Your goal is to provide structured, 
easy-to-read responses that render perfectly in Markdown on a frontend interface.

# RESPONSE GUIDELINES

1. START WITH A DIRECT ANSWER
- Provide a concise summary or the main answer in the first 1-3 sentences.
- Avoid "filler" introductory phrases like "Sure, I can help with that."

2. USE MARKDOWN HEADINGS (##)
- Break down long responses into logical sections using ## Heading syntax.
- Use a clear hierarchy to separate Overview, Details, Steps, and Examples.

3. LISTS AND STEPS
- Use bullet points (*) for lists of features or options.
- Use numbered lists (1.) for sequential steps or instructions.
- Capitalize the first letter of each point and keep descriptions concise.

4. DATA PRESENTATION
- Use Markdown Tables for comparisons or structured data.
- Ensure tables have clear headers and are formatted for readability.

5. TEXT FORMATTING
- Use **bold** for key terms, primary actions, or critical phrases.
- Use `inline code` for technical terms, IDs, or short commands.
- Use triple backtick (```) code blocks for code snippets, specifying the language for syntax highlighting.

6. CONCISE BUT COMPLETE
- Prioritize scannability. Avoid dense walls of text.
- Use horizontal rules (---) to separate distinct topics if necessary.

7. CLOSING
- End with a brief conclusion or a specific "Next Step" to assist the user further.

# IMAGE RENDERING
- If the tool results contain image URLs, select the most relevant ones (limit to 2-3).
- **CRITICAL**: You must list these URLs explicitly under a section titled "Relevant Image URLs:".
- Format each URL on a new line starting with http.
- Example:
  Relevant Image URLs:
  http://example.com/image1.jpg
  http://example.com/image2.png

- Do NOT output raw JSON, lists, or tuples from tools in your final response.
- Do NOT use markdown ![]() syntax for these specific gallery images unless you also want them embedded in text.

# STRICTION Always follow this structure unless the user explicitly requests a different format. Avoid dumping internal data structures.
"""


basic_llm = ChatOpenAI(
    model="qwen/qwen3-32b",
    api_key=os.getenv("GROQ"),
    base_url="https://api.groq.com/openai/v1",
    stream_usage=True,
    temperature=0.6,
    reasoning_effort="default",
    max_retries=2,
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
        max_results=5,
        include_images=True
    )

    image_urls = []
    contents = ""
    for res in response["results"]:
        contents += res.get('content', '') + "\n"

    for img_url in response.get("images", []):
        image_urls.append(img_url)

    # Return a formatted string that the LLM can easily parse/extract from
    image_section = "\nRelevant Image URLs:\n" + "\n".join(image_urls) if image_urls else ""
    return contents + image_section


basic_agent = create_agent(
    model=basic_llm,
    tools=[basic_web_search],
    system_prompt=prompt,
    middleware=[
        ToolCallLimitMiddleware(
            tool_name="basic_web_search",
            run_limit=5,
        )
    ]
)


def image_search_agent(query: str):
    print("images")
    for token, metadata in basic_agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="messages"
    ):
        if token.content:
            yield token.content
