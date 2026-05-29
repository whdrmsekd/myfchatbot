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
    api_version="2025-01-01-preview",  # 최신 다중 도구 연동 버전 지원
)

# ── 세션 상태 초기화 ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── [고정] 헤더 ──────────────────────────────────────────────────────────────
st.title("종근당의 통합 AI 챗봇")
st.caption("Azure OpenAI 기반 [RAG 문서 검색 + 파이썬 데이터 분석] 통합 에이전트")
st.divider()

# ── 기존 대화 출력 ───────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🙋" if msg["role"] == "user" else "💬"):
        if msg.get("content"):
            st.markdown(msg["content"])

# ── 사용자 입력 및 처리 ──────────────────────────────────────────────────────
user_input = st.chat_input("규정 검색이나 수식 계산/데이터 분석을 요청하세요...")

if user_input:
    # 1. 사용자 메시지 화면 출력 및 세션 저장
    with st.chat_message("user", avatar="🙋"):
        st.markdown(user_input)

    st.session_state.messages.append({
        "role": "user", 
        "content": user_input
    })

    # 2. [Assistant Index]가 내장된 시스템 가이드라인 정의
    system_instruction = (
        "너는 종근당의 [통합 데이터 및 규정 분석 전문가]이다. "
        "사용자의 질문 성격에 따라 아래의 [Assistant Index] 규칙을 엄격히 준수하여 도구를 사용하라.\n\n"
        "[Assistant Index]\n"
        "1. INDEX-RAG (사내 문서 및 산안법 조회): 안전 규정, 산안법, 사내 지침 등 지식 검색이 필요한 경우 "
        "연동된 azure_search 데이터 소스를 바탕으로 사실에 기반하여 답변할 것.\n"
        "2. INDEX-CALC (파이썬 기반 사칙연산 및 통계): 복잡한 수학 계산, 통계 공식, 사칙연산 수식 요청이 들어오면 "
        "암산하지 말고 반드시 내장된 파이썬 code_interpreter를 실행하여 완벽한 계산 결과를 도출할 것.\n"
        "3. INDEX-ANALYTICS (데이터 분석 및 시각화): 데이터 요약이나 차트 생성을 요청받으면 "
        "파이썬 환경에서 분석 코드를 수행하고 정교한 인사이트를 도출할 것.\n\n"
        "지시사항: 질문의 의도를 파악하여 적절한 INDEX 모드로 전환 후 답변을 작성하라."
    )

    chat_prompt = [
        {"role": "system", "content": system_instruction}
    ]
    
    # 이전 대화 문맥 추가
    for msg in st.session_state.messages:
        chat_prompt.append({"role": msg["role"], "content": msg["content"]})

    # 3. 어시스턴트 답변 생성 (스트리밍)
    with st.chat_message("assistant", avatar="💬"):
        placeholder = st.empty()
        full_response = ""

        # 💡 [교정 완료] try 구문이 with 절 안쪽으로 올바르게 들어왔습니다.
        try:
            # RAG(Azure Search)와 가상 파이썬 환경(Code Interpreter)을 올바른 Azure 스펙으로 병렬 배치
            response = client.chat.completions.create(
                model=deployment,
                messages=chat_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
                extra_body={
                    "data_sources": [
                        # 1. RAG 지식 검색 소스 설정
                        {
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
                        },
                        # 2. 💻 파이썬 가상 머신(Advanced Data Analytics) 올바른 확장 규격 배치
                        {
                            "type": "azure_vnet_code_interpreter",
                            "parameters": {
                                "auth": {
                                    "type": "access_token"
                                }
                            }
                        }
                    ]
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
