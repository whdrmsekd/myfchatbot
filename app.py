import os
import time
import re
import streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv

# 💡 다양한 그래프(추세선, 산점도, 통계 차트 등) 완벽 지원을 위해 주요 라이브러리 미리 로드
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

# ── 💡 [범용 그래프 엔진] 어떤 그래프 코드든 해석해서 실행하는 함수 ──────
def render_dynamic_graph(text):
    """답변 텍스트에서 파이썬 코드를 추출해 막대, 선, 원형, 산점도 등 모든 그래프를 그립니다."""
    code_blocks = re.findall(r"```python(.*?)```", text, re.DOTALL)
    for code in code_blocks:
        cleaned_code = code.strip()
        
        # plt.(matplotlib)나 sns.(seaborn)가 포함된 모든 시각화 코드를 감지
        if "plt." in cleaned_code or "sns." in cleaned_code:
            try:
                plt.switch_backend('Agg') # 백그라운드 렌더링 설정 (윈도우 팝업 방지)
                plt.clf() # 이전 그래프 잔상 초기화
                
                # 💡 다양한 패키지(plt, np, pd, sns)를 실행 환경에 전부 매핑하여 유연성 극대화
                local_vars = {
                    "plt": plt, 
                    "np": np, 
                    "pd": pd, 
                    "sns": sns
                }
                
                # 한글 깨짐을 방지하기 위한 폰트 설정을 실행 환경에 강제 주입 (운영체제 맞춤형)
                import platform
                if platform.system() == 'Windows':
                    plt.rc('font', family='Malgun Gothic')
                elif platform.system() == 'Darwin':
                    plt.rc('font', family='AppleGothic')
                plt.rc('axes', unicode_minus=False) # 마이너스 기호 깨짐 방지
                
                # AI가 작성한 다채로운 그래프 코드 백그라운드 실행
                exec(cleaned_code, globals(), local_vars)
                
                # Streamlit 컨테이너에 실제 완성된 형태의 그래프 렌더링
                st.pyplot(plt.gcf())
            except Exception as e:
                st.warning(f"⚠️ 그래프 시각화 중 오류가 발생했습니다: {e}")

# ── 기존 대화 출력 ───────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🙋" if msg["role"] == "user" else "💬"):
        if msg.get("content"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_dynamic_graph(msg["content"])

# ── 사용자 입력 및 처리 ──────────────────────────────────────────────────────
user_input = st.chat_input("규정 검색이나 원하시는 그래프 종류(막대, 선, 원형 등)를 요청하세요...")

if user_input:
    with st.chat_message("user", avatar="🙋"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    # 모드별 시스템 지침 설정
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
        # 💡 다양한 시각화 기법을 적극 활용하도록 프롬프트 고도화
        system_instruction = (
            "너는 종근당의 [통합 데이터 분석 및 시각화 전문가]이다. "
            "사용자의 요구사항에 가장 적절한 그래프 종류(막대그래프, 선그래프, 산점도, 히스토그램, 원그래프 등)를 스스로 판단하여 "
            "데이터를 완벽히 시각화하는 파이썬 코드 블록(```python ... ```)을 답변에 반드시 포함하라.\n"
            "지시사항:\n"
            "1. 시각화 시 matplotlib(plt) 또는 seaborn(sns)을 자유롭게 활용하라.\n"
            "2. 데이터는 딕셔너리나 리스트 형태로 코드 내에 명시적으로 선언하여 작동에 지반이 없게 하라.\n"
            "3. 코드 마지막에는 항상 'plt.show()'를 기재하여 마무리지을 것."
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
            
            # 실시간으로 생성된 텍스트 속 모든 종류의 그래프 일괄 빌드 및 출력
            render_dynamic_graph(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"❌ 오류 발생: {e}")
