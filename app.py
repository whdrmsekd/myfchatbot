import os
import time
import re
import streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv

# 필수 라이브러리 로드
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

load_dotenv()

# ── [고정] 페이지 설정 ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="종근당 통합 AI 챗봇",
    page_icon="💻",
    layout="centered",
)

# ── [한글 깨짐 전역 방지] ──────────────────────────────────────────────────
import platform
try:
    if platform.system() == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif platform.system() == 'Darwin':  # Mac
        plt.rc('font', family='AppleGothic')
    else:
        plt.rc('font', family='sans-serif')
    plt.rc('axes', unicode_minus=False)
except Exception:
    pass

# ── 사이드바: 파라미터 및 작업 모드 설정 ─────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 설정 및 모드 변경")

    bot_mode = st.radio(
        "🤖 작업 모드 선택",
        ["📝 사내 문서/산안법 검색 (RAG)", "📊 사칙연산/파이썬 데이터 분석"],
        index=0,
    )

    st.divider()
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

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version="2025-01-01-preview",
)

# ── 세션 상태 초기화 ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── [함수] 답변에서 파이썬 코드 블록만 지우고 텍스트만 추출 ──────────────────
def get_clean_text_without_code(text):
    """답변 내용 중 ```python ... ``` 구조를 완전히 제거한 순수 설명글만 반환합니다."""
    clean_text = re.sub(r"```python(.*?)```", "", text, flags=re.DOTALL)
    return clean_text.strip()

# ── [함수] 백그라운드 그래프 엔진 ───────────────────────────────────────────
def render_dynamic_graph(text):
    """답변 텍스트에서 파이썬 코드를 추출해 UI에 그래프만 깔끔하게 렌더링합니다."""
    code_blocks = re.findall(r"```python(.*?)```", text, re.DOTALL)
    for code in code_blocks:
        cleaned_code = code.strip()
        
        if "plt." in cleaned_code:
            try:
                plt.switch_backend('Agg') 
                plt.clf() 
                
                if platform.system() == 'Windows':
                    plt.rc('font', family='Malgun Gothic')
                elif platform.system() == 'Darwin':
                    plt.rc('font', family='AppleGothic')
                
                local_vars = {"plt": plt, "np": np, "pd": pd}
                exec(cleaned_code, globals(), local_vars)
                
                st.pyplot(plt.gcf())
            except Exception:
                pass

# ── 기존 대화 출력 ───────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🙋" if msg["role"] == "user" else "💬"):
        if msg.get("content"):
            # 데이터 분석 모드이면서 생성된 파이썬 코드 블록이 본문에 포함되어 있을 때만 필터링 적용
            if msg["role"] == "assistant" and "📊" in msg.get("mode", "") and "```python" in msg.get("content", ""):
                display_text = get_clean_text_without_code(msg["content"])
                if display_text:
                    st.markdown(display_text)
                render_dynamic_graph(msg["content"])
            else:
                st.markdown(msg["content"])

# ── 사용자 입력 및 처리 ──────────────────────────────────────────────────────
user_input = st.chat_input("선택한 모드에 맞춰 질문을 입력하세요...")

if user_input:
    with st.chat_message("user", avatar="🙋"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    # 💡 [핵심 추가] 사용자의 질문에 그래프 관련 키워드가 있는지 검사합니다.
    graph_keywords = ["그래프", "그려", "차트", "시각화", "plot", "graph", "chart"]
    wants_graph = any(keyword in user_input.lower() for keyword in graph_keywords)

    if "사내 문서" in bot_mode:
        system_instruction = (
            "너는 종근당의 [규정 및 산안법 분석 전문가]이다. "
            "연동된 사내 문서(azure_search) 데이터 소스를 바탕으로 사실에 기반하여 정확하게 답변하라."
        )
        extra_body_config = {
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
                    "authentication": {"type": "api_key", "key": f"{search_key}"}
                }
            }]
        }
    else:
        # 💡 [프롬프트 조건부 분리] 사용자가 그래프를 원할 때만 코드를 작성하도록 지침 하달
        if wants_graph:
            system_instruction = (
                "너는 종근당의 [통합 데이터 분석 및 시각화 전문가]이다. "
                "사용자가 데이터를 시각화하거나 그래프를 그려달라고 요청했으니, 요구에 맞는 설명과 함께 "
                "그 아래에 데이터 시각화를 위한 완벽한 파이썬 코드 블록(```python ... ```)을 반드시 포함하라.\n"
                "⚠️ 필수 제약사항:\n"
                "1. 시각화 코드는 반드시 matplotlib.pyplot(plt), numpy(np), pandas(pd)만 사용할 것.\n"
                "2. 그래프 제목과 라벨에 한글을 그대로 사용할 것.\n"
                "3. 코드 마지막 줄에는 항상 'plt.show()'를 기재하라."
            )
        else:
            system_instruction = (
                "너는 종근당의 [통합 데이터 분석 전문가]이다. "
                "사용자가 일반적인 질문이나 연산을 요청했으니, 친절하고 명확한 텍스트 답변으로만 정보를 제공하라. "
                "⚠️ 그래프를 그리라는 말이 없으므로, 파이썬 시각화 코드 블록(```python ... ```)은 절대 작성하지 마라."
            )
        extra_body_config = None

    chat_prompt = [{"role": "system", "content": system_instruction}]
    for msg in st.session_state.messages:
        if msg.get("content"):
            chat_prompt.append({"role": msg["role"], "content": msg["content"]})

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
                extra_body=extra_body_config if extra_body_config else None
            )

            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        full_response += delta.content
                        
                        # 화면 표시용 텍스트 스트리밍 분기
                        if "사내 문서" in bot_mode:
                            placeholder.markdown(full_response + "▌")
                        else:
                            if wants_graph:
                                placeholder.markdown(get_clean_text_without_code(full_response) + "\n\n(📊 그래프 차트 렌더링 중...) ▌")
                            else:
                                placeholder.markdown(full_response + "▌")

            # 최종 출력 정돈
            if "사내 문서" in bot_mode or not wants_graph:
                placeholder.markdown(full_response)
            else:
                display_text = get_clean_text_without_code(full_response)
                if display_text:
                    placeholder.markdown(display_text)
                else:
                    placeholder.empty()
            
            # 사용자가 원했을 때만 그래프 빌드 구동
            if "사내 문서" not in bot_mode and wants_graph:
                render_dynamic_graph(full_response)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "mode": bot_mode
            })

        except Exception as e:
            st.error(f"❌ 오류 발생: {e}")
