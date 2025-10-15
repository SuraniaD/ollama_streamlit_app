import os
import json
import requests
import streamlit as st

# --- Config ---
DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")  # Cloud: set to https://ollama.com
DEFAULT_API_KEY = os.getenv("OLLAMA_API_KEY", "")

st.set_page_config(page_title="Ollama Chat (Cloud/Local)", page_icon="💬")

# --- Helpers ---
def api_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"

def auth_headers(api_key: str | None) -> dict:
    hdrs = {"Content-Type": "application/json"}
    if api_key:
        hdrs["Authorization"] = f"Bearer {api_key}"  # Cloud uses Bearer token
    return hdrs

@st.cache_data(show_spinner=False, ttl=120)
def list_models(base_url: str, api_key: str) -> list[str]:
    """
    Cloud: returns cloud-available models; Local: returns pulled local models.
    Endpoint: GET /api/tags
    """
    try:
        resp = requests.get(api_url(base_url, "/api/tags"), headers=auth_headers(api_key), timeout=20)
        resp.raise_for_status()
        data = resp.json()
        # Cloud & local both return {"models":[{"name":"model:tag", ...}, ...]}
        return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        st.warning(f"Could not fetch models from {base_url}: {e}")
        # Sensible defaults if fetch fails
        return ["llama3.1", "llama3", "qwen2.5", "phi4"]

def stream_chat(base_url: str, api_key: str, model: str, messages: list[dict], temperature: float = 0.2):
    """
    POST /api/chat (streams JSON lines with fields like {"message":{"content":"..."}, "done":false})
    Set stream=True (default) and incrementally yield assistant text chunks.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,                 # streaming responses (default is streaming per docs)
        "options": {"temperature": temperature},
    }
    with requests.post(
        api_url(base_url, "/api/chat"),
        headers=auth_headers(api_key),
        data=json.dumps(payload),
        stream=True,
        timeout=300,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Sometimes servers send keepalives or partials; skip safely
                continue
            if "error" in obj:
                raise RuntimeError(obj["error"])
            # Typical objects contain {"message":{"role":"assistant","content":"..."}} or {"done":true}
            msg = obj.get("message", {})
            chunk = msg.get("content", "")
            if chunk:
                yield chunk
            if obj.get("done"):
                break

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    base_choice = st.radio(
        "Endpoint",
        ["Local (http://localhost:11434)", "Cloud (https://ollama.com)", "Custom"],
        index=1 if DEFAULT_BASE_URL.startswith("https://ollama.com") else 0,
    )
    if base_choice.startswith("Local"):
        base_url = "http://localhost:11434"
    elif base_choice.startswith("Cloud"):
        base_url = "https://ollama.com"
    else:
        base_url = st.text_input("Base URL", value=DEFAULT_BASE_URL, help="e.g., https://ollama.com or http://localhost:11434")

    api_key = st.text_input("Ollama API Key (Cloud only)", type="password", value=DEFAULT_API_KEY)
    models = list_models(base_url, api_key)
    model = st.selectbox("Model", options=models, index=0 if models else None)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.2, 0.1)
    if st.button("🔄 Refresh models"):
        st.cache_data.clear()
        st.rerun()

st.title("What are you working on?")

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]

# Render chat history
for m in st.session_state.messages:
    if m["role"] == "system":
        continue  # don't render system in chat UI
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- Input ---
prompt = st.chat_input("Ask anything")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant streaming reply
    with st.chat_message("assistant"):
        placeholder = st.empty()
        acc = []

        try:
            for chunk in stream_chat(base_url, api_key, model, st.session_state.messages, temperature):
                acc.append(chunk)
                placeholder.markdown("".join(acc))
        except Exception as e:
            st.error(f"Request failed: {e}")
        else:
            assistant_text = "".join(acc).strip()
            st.session_state.messages.append({"role": "assistant", "content": assistant_text})

# Utilities
col1, col2 = st.columns(2)
with col1:
    if st.button("Clear chat"):
        st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
        st.rerun()
with col2:
    st.caption(f"Endpoint: `{base_url}`  •  Model: `{model}`")
