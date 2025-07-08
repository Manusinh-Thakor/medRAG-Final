import os
import sys
import yaml
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


# === Add project root to sys.path ===
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

sys.path.append(project_root)

# === Import local modules ===
from agent.llm_loader import load_llm_from_config
from tools.websearch import websearch_tool
from tools.medclip import medclip_tool
from tools.medgemma import medgemma_tool
from tools.websearch_serpapi import serpapi_search_tool

# provider = config["tools"]["websearch"]["provider"]
# if provider == "tavily":
#     from langchain_tavily import TavilySearch
#     os.environ["TAVILY_API_KEY"] = config["tools"]["websearch"]["tavily_api_key"]
#     websearch_tool = TavilySearch(max_results=5)

# elif provider == "openai":
#     from tools.openai_websearch import openai_websearch_tool
#     websearch_tool = openai_websearch_tool 

# === Load config ===
config_path = os.path.join(project_root, "config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

def load_prompt_txt():
    prompt_path = os.path.join("agent_prompt.txt")
    with open(prompt_path, "r") as f:
        return f.read().strip()

# === Load LLM and tools ===
llm = load_llm_from_config(path=config_path)

# Agent expects system + placeholder for agent-built messages
custom_prompt = ChatPromptTemplate.from_messages([("system", load_prompt_txt()),("placeholder", "{messages}") ])

#print(custom_prompt)
## Agent Defination
tools=[serpapi_search_tool,medclip_tool,medgemma_tool]
memory = MemorySaver()
agent_executor = create_react_agent(llm, tools, checkpointer=memory, prompt=custom_prompt)
config = {"configurable": {"thread_id": "medrag_agent_123"}}

# def call_agent(qry):
#     input_message = {"role": "user","content": qry,}
#     for step in agent_executor.stream({"messages": [input_message]}, config, stream_mode="values"):
#         print("step....")
#         step["messages"][-1].pretty_print()


def call_agent(qry):
    input_message = {"role": "user", "content": qry}
    for step in agent_executor.stream({"messages": [input_message]}, config, stream_mode="values"):
        if "messages" in step:
            for msg in step["messages"]:
                if isinstance(msg, HumanMessage):
                    yield {"type": "human", "data": msg.content}
                elif isinstance(msg, AIMessage):
                    yield {"type": "ai", "data": msg.content}
                elif isinstance(msg, ToolMessage):
                    yield {"type": "tool", "data": msg.content}
                else:
                    yield {"type": "unknown", "data": str(msg)}






        