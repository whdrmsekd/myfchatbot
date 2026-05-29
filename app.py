import os
import time
import re
import streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv

# 💡 [검증] ModuleNotFoundError를 방지하기 위해 100% 기본 탑재된 핵심 라이브러리만 사용
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

# RAG 검색용 환경변수
search_endpoint = os.getenv("SEARCH_ENDPOINT", "")
search_key = os.getenv("SEARCH_KEY", "")
search_index = os.getenv("SEARCH_INDEX_NAME", "rag-10ai017safety")

if not endpoint or not api_key or not deployment:
    st.error("❌ OpenAI 환경변수가 설정되지 않았습니다. .env 파일을 확인하세요.")
    st.stop()

if not search_endpoint or not search_key:
    st.error("❌ Azure AI Search 환경변수가 설정되지 않았습니다. .env 파일을 확인하세요.")
    st.stop()

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version="2025-01-01-preview",
)

# ── 세션 상태 초기화 ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── [고정] 헤더 ──────────────────────────────────────────────────────────────
st.title("종근당의 통합 AI 챗봇")
st.caption(f"현재 작동 모드: **{bot_mode}**")
st.divider()

# ── 💡 [안전 강화] 외부 모듈 의존성을 제거하고 2중 예외 처리를 갖춘 그래프 엔진 ──
def render_dynamic_graph(text):
    """답변 텍스트에서 파이썬 코드를 추출해 막대, 선, 원형 등 다양한 그래프를 안전하게 렌더링합니다."""
    code_blocks = re.findall(r"```python(.*?)```", text, re.DOTALL)
    for code in code_blocks:
        cleaned_code = code.strip()
        
        if "plt." in cleaned_code:
            try:
                plt.switch_backend('Agg') # 팝업 창 방지
                plt.clf() # 캔버스 초기화
                
                # 💡 [생각 및 검증] 미설치 가능성이 높은 seaborn을 완전히 제외하고 안정성이 검증된 로컬 변수만 바인딩
                local_vars = {
                    "plt": plt, 
                    "np": np, 
                    "pd": pd
                }
                
                # 한글 깨짐 방지 서포트
                import platform
                if platform.system() == 'Windows':
                    plt.rc('font', family='Malgun Gothic')
                elif platform.system() == 'Darwin':
                    plt.rc('font', family='AppleGothic')
                plt.rc('axes', unicode_minus=False)
                
                # 💡 [2중 방어] exec 실행 중 모듈 관련 에러가 나더라도 전체 Streamlit 앱이 죽지 않도록 격리
                exec(cleaned_code, globals(), local_vars)
                
                st.pyplot(plt.gcf())
            except ModuleNotFoundError as mne:
                st.info(f"💡 현재 환경에 필요한 라이브러리가 부족하여 그래프 출력을 건너넙니다. (필요 모듈: {mne.name})")
            except Exception as e:
                st.warning(f"⚠️ 그래프 시각화 코드 실행 중 오류가 발생했습니다: {e}")

# ── 기존 대화 출력 ───────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🙋" if msg["role"] == "user" else "💬"):
        if msg.get("content"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_dynamic_graph(msg["content"])

# ── 사용자 입력 및 처리 ──────────────────────────────────────────────────────
user_input = st.chat_input("규정 검색이나 데이터 시각화(막대, 선, 원형 그래프 등)를 요청하세요...")

if user_input:
    with st.chat_message("user", avatar="🙋"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

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
        # 💡 [생각 및 검증] AI가 설치 안 된 seaborn을 쓰지 않고 matplotlib(plt)과 pandas(pd)만 사용하도록 프롬프트로 강력 제어
        system_instruction = (
            "너는 종근당의 [통합 데이터 분석 및 시각화 전문가]이다. "
            "사용자의 요구사항에 맞는 다양한 종류의 그래프(막대그래프, 선그래프, 산점도, 원그래프 등)를 그리는 파이썬 코드 블록(```python ... ```)을 반드시 포함하라.\n"
            "⚠️ 필수 제약사항:\n"
            "1. 시각화 코드는 반드시 오직 matplotlib.pyplot(plt), numpy(np), pandas(pd)만 사용해야 하며, seaborn(sns) 등 다른 외부 라이브러리는 절대 사용하지 마라.\n"
            "2. 그래프에 사용되는 데이터는 리스트나 딕셔너리 형태로 코드 내부에 명시적으로 선언하라.\n"
            "3. 코드 마지막 줄에는 항상 'plt.show()'를 기재하여 마무리지을 것."
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
                        placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)
            
            # 실시간으로 동적 그래프 렌더링 구동
            render_dynamic_graph(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"❌ 오류 발생: {e}")
