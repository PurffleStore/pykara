# chat_chain.py
import os
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---- Configuration (via env, with safe defaults) ----
HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN")
MODEL_ID: str = os.environ.get("MODEL_ID", "openai/gpt-oss-20b:nebius")
BASE_URL: str = os.environ.get("BASE_URL", "https://router.huggingface.co/v1")
TEMP: float = float(os.environ.get("TEMPERATURE", "0.2"))


# ---- Build the chain once (module-level cache) ----
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set. Add it to your environment or Spaces → Settings → Secrets.")

_llm = ChatOpenAI(
    model=MODEL_ID,
    api_key=HF_TOKEN,
    base_url=BASE_URL,
    temperature=TEMP,
)

_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, precise assistant. Reply in simple, neutral English."),
    ("user", "{message}")
])

_chain = _prompt | _llm | StrOutputParser()


def get_answer(message: str) -> str:
    """
    Generate a single reply for the given user message.
    Keeps LangChain initialization separate from the web layer.
    """
    if not message or not message.strip():
        raise ValueError("message cannot be empty.")
    return _chain.invoke({"message": message.strip()})
