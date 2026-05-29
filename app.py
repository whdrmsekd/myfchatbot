import os
import time
import re
import streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# ── [고정] 페이지 설정 ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="종근당 챗봇",
    page_icon="💻",
    layout="centered",
)

# ── 사이드바: 파라미터 설정 ──────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 설정")

    max_tokens = st.slider("최대 토큰 (max_tokens)", 100, 8000, 4000, 100)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
    top_p = st.slider("Top-p", 0.0, 1.0, 0.95, 0.01)

    st.divider()
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── 환경변수 로드 및 검증 ──────────────────────────────────────────────────
endpoint = os.getenv("AZURE_OAI_ENDPOINT") or os.getenv("ENDPOINT_URL", "")
api_key = os.getenv("AZURE_OAI_KEY") or os.getenv("AZURE_OPENAI_API_KEY", "")
deployment = os.getenv("AZURE_OAI_DEPLOYMENT") or os.getenv("DEPLOYMENT_NAME", "")

search_endpoint = os.getenv("SEARCH_ENDPOINT", "")
search_key = os.getenv("SEARCH_KEY", "")
search_index = os.getenv("SEARCH_INDEX_NAME", "rag-10ai017safety")

if not endpoint or not api_key or not deployment:
    st.error("❌ OpenAI 환경변수가 설정되지 않았습니다. .env 파일을 확인하세요.")
    st.stop()

if not search_endpoint or not search_key:
    st.error("❌ Azure AI Search 환경변수가 설정되지 않았습니다. .env 파일을 확인하세요.")
    st.stop()

# 클라이언트 초기화
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version="2025-01-01-preview",
)

# ── 세션 상태 초기화 ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── [고정] 헤더 ──────────────────────────────────────────────────────────────
st.title("종근당의 챗봇")
st.caption("Azure OpenAI & AI Search 기반 RAG 챗봇")
st.divider()

# ── 기존 대화 출력 ───────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🙋" if msg["role"] == "user" else "💬"):
        if msg.get("content"):
            st.markdown(msg["content"])

# ── 사용자 입력 및 처리 ──────────────────────────────────────────────────────
user_input = st.chat_input("무엇이든 물어보세요...")

if user_input:
    # 1. 사용자 메시지 화면 출력 및 세션 저장
    with st.chat_message("user", avatar="🙋"):
        st.markdown(user_input)

    st.session_state.messages.append({
        "role": "user", 
        "content": user_input
    })

    # 2. API 요청을 위한 메시지 배열 생성
    chat_prompt = [
        {"role": "system", "content": "사용자가 정보를 찾는 데 도움이 되는 AI 도우미입니다. 사내 문서 내용을 기반으로 정확하게 답변하세요."}
    ]
    
    # 이전 대화 문맥 추가
    for msg in st.session_state.messages:
        chat_prompt.append({"role": msg["role"], "content": msg["content"]})

    # 3. 어시스턴트 답변 생성 (스트리밍)
    with st.chat_message("assistant", avatar="💬"):
        placeholder = st.empty()
        full_response = ""

        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=chat_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
                extra_body={
                    "data_sources": [{
                        "type": "azure_search",
                        "parameters": {
                            "endpoint": f"{search_endpoint}",
                            "index_name": search_index,
                            "semantic_configuration": "rag-10ai017safety-semantic-configuration",
                            "query_type": "semantic",
                            "fields_mapping": {},
                            "in_scope": True,
                            "filter": None,
                            "strictness": 3,
                            "top_n_documents": 5,
                            "authentication": {
                                "type": "api_key",
                                "key": f"{search_key}"
                            }
                        }
                    }]
                }
            )

            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        full_response += delta.content
                        placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"❌ 오류 발생: {e}")
