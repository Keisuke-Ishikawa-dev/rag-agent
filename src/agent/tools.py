from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()


def get_rag_tool(session):
    from src.rag.embeddings import similarity_search

    @tool
    async def search_knowledge_base(query: str) -> str:
        """Search the internal knowledge base for relevant information. Use this before web search."""
        results = await similarity_search(session, query)
        if not results:
            return "No relevant documents found in knowledge base."
        return "\n\n".join(
            f"[Score: {r['score']:.2f}]\n{r['content']}"
            for r in results
        )

    return search_knowledge_base


@tool
def web_search(query: str) -> str:
    """Search the web for current information not found in the knowledge base."""
    return search_tool.run(query)


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Input must be a valid Python expression."""
    try:
        allowed = {k: v for k, v in __builtins__.items()
                   if k in ("abs", "round", "min", "max", "sum", "pow")}
        result = eval(expression, {"__builtins__": allowed})
        return str(result)
    except Exception as e:
        return f"Error: {e}"
