import os
import yaml
import boto3
from langchain.chat_models import init_chat_model
from langchain_ollama.llms import OllamaLLM
from langchain_aws.chat_models import ChatBedrock

def load_llm_from_config(path="config.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)["llm"]

    provider = cfg["provider"]
    model_id = cfg["model"]

    if provider == "openai":
        api_key = cfg.get("openai_api_key")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        return init_chat_model(f"openai:{model_id}")

    elif provider == "ollama":
        ollama_cfg = cfg.get("ollama", {})
        base_url = ollama_cfg.get("base_url", "http://localhost:11434")
        return OllamaLLM(model=model_id, base_url=base_url)

    elif provider == "bedrock":
        bedrock_cfg = cfg.get("bedrock", {})
        client = boto3.client(
            service_name="bedrock-runtime",
            region_name=bedrock_cfg.get("region_name", "us-east-1"),
            aws_access_key_id=bedrock_cfg.get("aws_access_key_id"),
            aws_secret_access_key=bedrock_cfg.get("aws_secret_access_key")
        )
        return ChatBedrock(model_id=model_id, client=client)

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
