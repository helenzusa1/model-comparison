import os
import time
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import OpenAI, NotFoundError
import tiktoken

load_dotenv()

# ---- 1) Endpoint ----
endpoint = "https://mcp-snippy-resource.cognitiveservices.azure.com"
base_url = f"{endpoint}/openai/v1/"

# ---- 2) Entra token provider ----
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default"
)

client = OpenAI(
    base_url=base_url,
    api_key=token_provider
)

# ---- 3) Preflight ----
try:
    _ = client.models.list()
except NotFoundError:
    raise RuntimeError("Invalid Azure OpenAI endpoint")

PROMPT = """Explain GraphRAG in detail.
Cover:
1) The core architecture and data flow
2) How graph construction differs from vector-only RAG
3) The role of entities, relationships, and community summaries
4) Trade-offs in latency, accuracy, and scalability
5) When GraphRAG is preferable over classic RAG
Use clear sections and concrete examples.
"""

print("Prompt:", PROMPT)

# ---- Helpers ----
def fmt(x, spec):
    return f"{x:{spec}}" if x is not None else "n/a"

def label_for(model, reasoning):
    if not reasoning:
        return f"{model} (default)"
    return f"{model} (reasoning={reasoning.get('effort')})"

enc = tiktoken.get_encoding("cl100k_base")

def is_chat_model(model: str) -> bool:
    # Only legacy chat models that actually stream via ChatCompletions
    return False

def supports_reasoning(model: str) -> bool:
    return model == "gpt-5"

def extract_text(event):
    return getattr(event, "delta", None) or getattr(event, "text", None) or ""

# ---- Metrics ----
def measure_metrics(model_name: str, reasoning=None):
    t_start = time.time()
    t_first = None
    output_text = ""

    req = {
        "model": model_name,
        "input": PROMPT,
    }
    if reasoning and supports_reasoning(model_name):
        req["reasoning"] = reasoning

    with client.responses.stream(**req) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                if t_first is None:
                    t_first = time.time()
                output_text += event.delta or ""

    t_end = time.time()

    out_tokens = len(enc.encode(output_text))
    ttft = None if t_first is None else (t_first - t_start)
    e2e = t_end - t_start

    tpot = None
    out_tps = None
    if ttft is not None and out_tokens > 0:
        gen_time = t_end - t_first
        tpot = gen_time / out_tokens
        out_tps = out_tokens / gen_time

    return {
        "ttft_s": ttft,
        "e2e_s": e2e,
        "out_tokens": out_tokens,
        "tpot_s": tpot,
        "out_tps": out_tps,
    }

# ---- Tests ----
tests = [
    ("gpt-4.1", None),
    ("gpt-5", {"effort": "medium"}),
    ("gpt-5", {"effort": "low"}),
    ("gpt-5.3-chat", None),
    ("gpt-5.4", None),
    ("gpt-5.4-mini", None),
    ("gpt-5.4-nano", None),
]

print("\nModel metrics:")
for model, reasoning in tests:
    m = measure_metrics(model, reasoning)
    print(
        f"{label_for(model, reasoning):30} "
        f"TTFT={fmt(m['ttft_s'], '.3f')}s "
        f"E2E={fmt(m['e2e_s'], '.3f')}s "
        f"OutTok={m['out_tokens']:4d} "
        f"TPOT={fmt(m['tpot_s'], '.4f')}s/tok "
        f"OutTPS={fmt(m['out_tps'], '.1f')} tok/s"
    )
