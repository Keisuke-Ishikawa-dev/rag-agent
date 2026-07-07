from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from src.config import settings
from src.agent.tools import web_search, calculate

REACT_PROMPT = PromptTemplate.from_template("""You are a helpful AI assistant with access to a knowledge base and web search.
Always search the knowledge base first before using web search.
Answer in the same language as the question.

You have access to the following tools:
{tools}

Use this format strictly:

Question: the input question you must answer
Thought: think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Chat History:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}""")


def build_agent(session) -> AgentExecutor:
    from src.agent.tools import get_rag_tool

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        openai_api_key=settings.openai_api_key,
    )

    tools = [get_rag_tool(session), web_search, calculate]

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=5,
        return_messages=False,
    )

    agent = create_react_agent(llm=llm, tools=tools, prompt=REACT_PROMPT)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=8,
    )
