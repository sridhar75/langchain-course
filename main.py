from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for the agent's response"""

    answer: str = Field(description="The answer to the user's query")
    sources: List[Source] = Field(
        default_factory=list, description="The sources used to answer the query"
    )


# from tavily import TavilyClient

# tavily = TavilyClient()


# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over the internet
#     Args:
#         query: The query to search for
#     Returns:
#         The search results
#     """
#     print(f"Search results for '{query}'")
#     return tavily.search(query=query)

llm = ChatOllama(
    model="qwen3:14b",
    temperature=0.0,
    num_ctx=8192,
    timeout=120,
)
# tools = [search]
tools = [TavilySearch(max_results=3, time_range="month")]
agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse,
    system_prompt="You MUST use the available search tools to answer questions. Never answer from memory. Always search first.",
)


def main():
    print("Hello from langchain-course!")
    # result = agent.invoke({"messages": HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details")})
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="search for only top 3 job postings in the bay area on linkedin and list their details for an ai engineer that needs langchain skill"
                )
            ]
        }
    )
    print(result)


if __name__ == "__main__":
    main()
