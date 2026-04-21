from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
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

llm = ChatOllama(model="gpt-oss:20b", temperature=0.0)  
# tools = [search]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)
def main():
    print("Hello from langchain-course!")
    # result = agent.invoke({"messages": HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details")})
    result = agent.invoke({"messages": [HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details")]})
    print(result)


if __name__ == "__main__":
    main()
