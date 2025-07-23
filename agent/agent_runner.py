import os
import sys
import yaml
import time
import json
import uuid
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

# === Project Path Setup ===
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

# === Config ===
config_path = os.path.join(project_root, "config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

# === Tool Imports ===
from agent.llm_loader import load_llm_from_config
from tools.websearch_serpapi import serpapi_search_tool
from tools.medclip import medclip_tool
from tools.medgemma import medgemma_tool

# === Load Agent Prompt ===
def load_prompt_txt():
    base_path = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(base_path, "agent_prompt2.txt")
    with open(prompt_path, "r") as f:
        return f.read().strip()

# === LangGraph Agent ===
llm = load_llm_from_config(path=config_path)
custom_prompt = ChatPromptTemplate.from_messages([
    ("system", load_prompt_txt()),
    ("placeholder", "{messages}")
])
tools = [serpapi_search_tool, medclip_tool, medgemma_tool]
memory = MemorySaver()
agent_executor = create_react_agent(llm, tools, checkpointer=memory, prompt=custom_prompt)

# === OpenAI LLM for direct pipeline ===
llm_cfg = config.get("llm", {})
openai_llm = ChatOpenAI(
    model=llm_cfg.get("model", "gpt-4"),
    openai_api_key=llm_cfg["openai_api_key"]
)

# === New Image Pipeline (no agent) ===
def image_pipeline_openai(query: str, image_path: str):
    print("\U0001F4E6 Stage: MedGEMMA")
    medgemma_result = medgemma_tool.invoke(image_path)
    yield {"type": "medgemma", "data": medgemma_result}
    time.sleep(1)

    search_prompt = f"Extract a concise medical search phrase (2-3 words) to find similar cases based on this radiology report:\n\n{medgemma_result}"
    search_term = 'a chest xray of ' + openai_llm.invoke(search_prompt).content.strip()
    print(f"\U0001F50D Search term: {search_term}")
    time.sleep(1)

    print("\U0001F4E6 Stage: MedCLIP")
    try:
        db_result = medclip_tool.invoke(search_term)
        db_result = json.loads(db_result) if isinstance(db_result, str) else db_result
        db_urls = db_result.get("images", db_result)
        yield {"type": "db_images", "data": db_urls}
    except Exception as e:
        print(f"\u274C MedCLIP error: {e}")
        yield {"type": "db_images", "data": []}
    time.sleep(1)

    print("\U0001F4E6 Stage: Web Search")
    try:
        web_result = serpapi_search_tool.invoke(search_term)
        web_result = json.loads(web_result) if isinstance(web_result, str) else web_result
        web_urls = [x["image_url"] for x in web_result if "image_url" in x]
        yield {"type": "web_images", "data": web_urls}
    except Exception as e:
        print(f"\u274C Web search error: {e}")
        yield {"type": "web_images", "data": []}
    time.sleep(1)

    yield {"type": "ai", "data": "All tool outputs have been streamed."}

pipe = False

def call_agent(qry, image=False, image_path=None):
    print("\U0001F4E9 call_agent got:", qry, image, image_path)
    seen = set()

    if image and image_path and pipe:
        print("using pipeline")
        generator = image_pipeline_openai(qry, image_path)
    else:
        print("using agent")
        thread_id = f"session-{uuid.uuid4()}"
        final_input_str = f"Query: {qry}\nImage: {image}\nImage Path: {image_path}"
        input_message = {"role": "user", "content": final_input_str}
        generator = agent_executor.stream(
            {"messages": [input_message]},
            config={"configurable": {"thread_id": thread_id}},
            stream_mode="values"
        )

    for step in generator:
        if isinstance(step, dict) and "messages" in step:
            for msg in step["messages"]:
                if isinstance(msg, AIMessage):
                    raw = msg.content.strip()
                    if "\ntype:" in raw:
                        raw = raw.split("\ntype:", 1)[1].strip()
                    if "\ndata:" in raw:
                        raw = raw.split("\ndata:", 1)[1].strip()
                    key = ("ai", raw)
                    if key in seen:
                        continue
                    seen.add(key)
                    yield {"type": "ai", "data": raw}
                    time.sleep(0.5)

                elif isinstance(msg, ToolMessage):
                    try:
                        parsed = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                        if isinstance(parsed, list):
                            tool_type = "web_images"
                        elif isinstance(parsed, dict) and "images" in parsed:
                            tool_type = "db_images"
                            parsed = parsed["images"]
                        else:
                            tool_type = "tool"
                    except:
                        tool_type = "tool"
                        parsed = msg.content

                    key = (tool_type, json.dumps(parsed))
                    if key in seen:
                        continue
                    seen.add(key)
                    yield {"type": tool_type, "data": parsed}
                    time.sleep(0.5)

        elif isinstance(step, dict):
            step_type = step.get("type", "ai")
            content = step.get("data", "")
            key = (step_type, json.dumps(content))
            if key in seen:
                continue
            seen.add(key)
            yield {"type": step_type, "data": content}
            time.sleep(0.5)
