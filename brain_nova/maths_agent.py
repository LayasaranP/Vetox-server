from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a math solver. At the end of the solution give some follow-up questions or suggestions."),
    ("human", "{question}")
])

maths_llm = ChatOpenAI(
    model="deepseek-r1-0528:free",
    openai_api_key="sk--dkWRoL1V4zO08e3uS427x7W_eZ4VjenRJiPyYz6bUhIxPbuBBqlYGdSFjuz918n6tBFQZqnOX7DMOE",
    openai_api_base="https://api.routeway.ai/v1",
    streaming=True,
)


def maths_solver(question: str, llm=maths_llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a math solver."),
        ("human", "{question}")
    ])

    chain = prompt | maths_llm

    full_response = ""
    for chunk in chain.stream({"question": question}):
        if chunk.content:
            full_response += chunk.content
            yield chunk.content


