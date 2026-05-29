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

# 💡 [한글 깨짐 원천 차단] 시스템 운영체제에 맞는 한글 폰트를 앱 시작 시 전역 설정합니다.
import platform
try:
    if platform.system() == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif platform.system() == 'Darwin':  # Mac
        plt.rc('font', family='AppleGothic')
    else:  # 리눅스/서버 환경 등 기본 폰트가 없을 때 깨짐 방지용 기본 설정
        plt.rc('font', family='sans-serif')
    plt.rc('axes', unicode_minus=False)  # 마이너스 기호 깨짐 방지
except Exception as font_err:
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

# ── 💡 [안전 강화] 백그라운드 그래프 엔진 ───────────────────────────────────
def render_dynamic_graph(text):
    """답변 텍스트에서 파이썬 코드를 추출해 UI에 그래프만 깔끔하게 렌더링합니다."""
    code_blocks = re.findall(r"```python(.*?)```", text, re.DOTALL)
    for code in code_blocks:
        cleaned_code = code.strip()
        
        if "plt." in cleaned_code:
            try:
                plt.switch_backend('Agg') 
                plt.clf() 
                
                # 실행 컨텍스트에 폰트 재주입 (AI가 오버라이딩하는 것 방지)
                if platform.system() == 'Windows':
                    plt.rc('font', family='Malgun Gothic')
                elif platform.system() == 'Darwin':
                    plt.rc('font', family='AppleGothic')
                
                local_vars = {"plt": plt, "np": np, "pd": pd}
                exec(cleaned_code, globals(), local_vars)
                
                # 📊 그래프 이미지만 화면에 표시
                st.pyplot(plt.gcf())
            except Exception as e:
                pass

# ── 기존 대화 출력 ───────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🙋" if msg["role"] == "user" else "💬"):
        # 💡 [코드 숨김 제어] 데이터 분석 모드이면서 어시스턴트 답변일 때는 본문(텍스트/코드)을 그리지 않고 그래프만 복원
        if msg["role"] == "assistant" and "📊" in msg.get("mode", ""):
            render_dynamic_graph(msg["content"])
        else:
            if msg.get("content"):
                st.markdown(msg["content"])

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
        system_instruction = (
            "너는 종근당의 [통합 데이터 분석 및 시각화 전문가]이다. "
            "사용자의 요구사항에 맞는 다양한 종류의 그래프를 그리는 파이썬 코드 블록(```python ... ```)만 정확히 생성하라. "
            "텍스트 설명은 제외하거나 최소화하고, 반드시 실행 가능한 완벽한 코드만 전달하라.\n"
            "⚠️ 필수 제약사항:\n"
            "1. 시각화 코드는 반드시 matplotlib.pyplot(plt), numpy(np), pandas(pd)만 사용할 것.\n"
            "2. 그래프의 제목(title), 축 이름(xlabel, ylabel) 등 한글 텍스트는 문자열 그대로 작성하라. (예: plt.title('2의 제곱수 그래프'))\n"
            "3. 코드 마지막 줄에는 항상 'plt.show()'를 기재하라."
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
                        # 데이터 분석 모드가 아닐 때(즉, RAG 모드일 때만) 텍스트 글자를 화면에 실시간 스트리밍 노출
                        if "사내 문서" in bot_mode:
                            placeholder.markdown(full_response + "▌")
                        else:
                            placeholder.markdown("📊 그래프 차트를 그리는 중입니다... ▌")

            # 💡 [핵심 수정] 최종 완료 후 데이터 분석 모드라면 텍스트 코드 창을 완전히 지우고 그래프만 출력
            if "사내 문서" in bot_mode:
                placeholder.markdown(full_response)
            else:
                placeholder.empty() # "그래프 그리는 중" 메시지 및 원본 파이썬 코드 텍스트 숨김 처리
            
            # 그래프 빌드 엔진 호출 (화면에 진짜 차트 이미지 생성)
            render_dynamic_graph(full_response)
            
            # 이전 히스토리에 현재 모드 정보를 매킹하여 세션 저장 (새로고침 시 코드 숨김 유지용)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "mode": bot_mode
            })

        except Exception as e:
            st.error(f"❌ 오류 발생: {e}")
