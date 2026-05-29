import os
import time
import re
import streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# ── [고정] 페이지 설정 ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="종근당 통합 AI 챗봇",
    page_icon="💻",
    layout="centered",
)

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

# 클라이언트 초기화
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version="2025-01-01-preview",
)

# ── 세션 상태 초기화 ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── 사이드바: 파라미터 및 작업 모드 설정 ─────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 설정 및 모드 변경")

    # 💡 사용자가 직관적으로 업무 모드를 선택할 수 있도록 하여 API 400 에러를 원천 차단합니다.
    bot_mode = st.radio(
        "🤖 작업 모드 선택",
        ["📝 사내 문서/산안법 검색 (RAG)", "📊 사칙연산/파이썬 데이터 분석"],
        index=0,
        help="수행할 업무에 맞는 모드를 선택하면 최적의 도구가 자동으로 매핑됩니다."
    )

    st.divider()
    max_tokens = st.slider("최대 토큰 (max_tokens)", 100, 8000, 4000, 100)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
    top_p = st.slider("Top-p", 0.0, 1.0, 0.95, 0.01)

    st.divider()
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── [고정] 헤더 ──────────────────────────────────────────────────────────────
st.title("종근당의 통합 AI 챗봇")
st.caption(f"현재 작동 모드: **{bot_mode}**")
st.divider()

# ── 기존 대화 출력 ───────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🙋" if msg["role"] == "user" else "💬"):
        if msg.get("content"):
            st.markdown(msg["content"])

# ── 사용자 입력 및 처리 ──────────────────────────────────────────────────────
user_input = st.chat_input("선택한 모드에 맞춰 질문을 입력하세요...")

if user_input:
    # 1. 사용자 메시지 화면 출력 및 세션 저장
    with st.chat_message("user", avatar="🙋"):
        st.markdown(user_input)

    st.session_state.messages.append({
        "role": "user", 
        "content": user_input
    })

    # 2. 선택된 모드에 따른 시스템 지침 및 단일 데이터 소스(data_source) 동적 세팅
    if "사내 문서" in bot_mode:
        system_instruction = (
            "너는 종근당의 [규정 및 산안법 분석 전문가]이다. "
            "현재 사내 문서 지식 검색 모드이므로 연동된 azure_search 데이터 소스를 바탕으로 "
            "철저히 문서 내용에 기반하여 정확하고 사실적인 답변을 작성하라."
        )
        chosen_source = {
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
        }
    else:
        system_instruction = (
            "너는 종근당의 [통합 데이터 분석 전문가]이다. "
            "현재 사칙연산 및 파이썬 분석 모드이므로 계산 요청을 받으면 절대로 암산하거나 임의로 계산하지 말고, "
            "반드시 내장된 파이썬 code_interpreter를 실행하여 소수점까지 오차 없는 완벽한 계산 결과를 도출하라."
        )
        chosen_source = {
            "type": "azure_vnet_code_interpreter",
            "parameters": {
                "auth": {"type": "access_token"}
            }
        }

    # 대화 프롬프트 배열 생성
    chat_prompt = [{"role": "system", "content": system_instruction}]
    for msg in st.session_state.messages:
        chat_prompt.append({"role": msg["role"], "content": msg["content"]})

    # 3. 어시스턴트 답변 생성 (스트리밍)
    with st.chat_message("assistant", avatar="💬"):
        placeholder = st.empty()
        full_response = ""

        try:
            # 완벽히 정렬된 단일 데이터 소스 구조로 API 호출 (400 에러 원천 해결)
            response = client.chat.completions.create(
                model=deployment,
                messages=chat_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
                extra_body={
                    "data_sources": [chosen_source]
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
