from brain_nova.main_llm_agent import main_router
from brain_nova.basic_llm_agent import basic_internet_search_agent
from brain_nova.advance_llm_agent import llm
from brain_nova.maths_agent import maths_solver
from brain_nova.coding_agent import stream_chat_with_prompt
from brain_nova.image_llm_agent import image_search_agent
from langchain_groq import ChatGroq

ma_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.6,
    reasoning_effort="high",
    model_kwargs={
        "top_p": 0.95,
    }
)


def get_response_brain_nova(query: str, chat_id):
    res = main_router(query, chat_id)

    if res['action'] == "DIRECT_RESPONSE":
        yield f"reasoning: {res['reasoning']}, content: {res['response_content']}"
    elif res['action'] == "BASIC_SEARCH":
        for chunk in basic_internet_search_agent(query, chat_id):
            yield chunk
    elif res['action'] == "ADVANCED_SEARCH":
        for chunk in llm(query, chat_id):
            yield chunk
    elif res['action'] == "MATHS_SOLVER":
        for chunk in maths_solver(query):
            yield chunk
    elif res['action'] == "CODE_ASSISTANT":
        for chunk in stream_chat_with_prompt(query):
            yield chunk
    else:
        for chunk in image_search_agent(query):
            yield chunk
