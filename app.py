import os
import streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# ── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="컴퓨터 전문가 AI",
    page_icon="💻",
    layout="centered",
)

# ── 사이드바: 파라미터 설정 ──────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 설정")

    message_cnt = st.slider("대화 기억 수 (message_cnt)", 1, 20, 3,
                            help="AI가 기억하는 이전 대화 쌍의 수")
    max_tokens = st.slider("최대 토큰 (max_tokens)", 100, 2000, 800)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
    top_p = st.slider("Top-p", 0.0, 1.0, 0.95, 0.01)
    frequency_penalty = st.slider("Frequency Penalty", -2.0, 2.0, 0.0, 0.1)
    presence_penalty = st.slider("Presence Penalty", -2.0, 2.0, 0.0, 0.1)

    st.divider()
    st.markdown("**Azure OpenAI 환경변수**")
    st.code("AZURE_OAI_ENDPOINT\nAZURE_OAI_DEPLOYMENT\nAZURE_OAI_KEY", language="bash")

    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Azure OpenAI 클라이언트 초기화 ──────────────────────────────────────────
endpoint = os.getenv("AZURE_OAI_ENDPOINT", "")
api_key = os.getenv("AZURE_OAI_KEY", "")
deployment = os.getenv("AZURE_OAI_DEPLOYMENT", "")

if not endpoint or not api_key or not deployment:
    st.error("❌ 환경변수가 설정되지 않았습니다. AZURE_OAI_ENDPOINT / AZURE_OAI_KEY / AZURE_OAI_DEPLOYMENT 를 확인하세요.")
    st.stop()

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version="2025-01-01-preview",
)

SYSTEM_PROMPT = {
    "role": "system",
    "content": [{"type": "text", "text": "너는 컴퓨터 전문가야"}],
}

# ── 세션 상태 초기화 ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []   # (role, text) 튜플 리스트

# ── 헤더 ────────────────────────────────────────────────────────────────────
st.title("💻 컴퓨터 전문가 AI")
st.caption("Azure OpenAI 기반 챗봇 · 사이드바에서 파라미터를 조정하세요")
st.divider()

# ── 기존 대화 출력 ───────────────────────────────────────────────────────────
for role, text in st.session_state.messages:
    avatar = "🙋" if role == "user" else "💬"
    with st.chat_message(role, avatar=avatar):
        st.markdown(text)

# ── 사용자 입력 ──────────────────────────────────────────────────────────────
user_input = st.chat_input("메시지를 입력하세요...")

if user_input:
    # 화면에 유저 메시지 출력
    with st.chat_message("user", avatar="🙋"):
        st.markdown(user_input)

    st.session_state.messages.append(("user", user_input))

    # ── API에 보낼 메시지 구성 (시스템 + 슬라이딩 윈도우) ───────────────────
    message_limit = message_cnt * 2          # 보관할 최대 메시지 수

    history = st.session_state.messages      # 전체 기록
    windowed = history[-message_limit:] if len(history) > message_limit else history

    chat_payload = [SYSTEM_PROMPT] + [
        {
            "role": role,
            "content": [{"type": "text", "text": text}],
        }
        for role, text in windowed
    ]

    # ── AI 응답 (스트리밍) ───────────────────────────────────────────────────
    with st.chat_message("assistant", avatar="💬"):
        placeholder = st.empty()
        full_response = ""

        try:
            stream = client.chat.completions.create(
                model=deployment,
                messages=chat_payload,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=None,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    full_response += delta.content
                    placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"❌ 오류 발생: {e}"
            placeholder.error(full_response)

    st.session_state.messages.append(("assistant", full_response))
