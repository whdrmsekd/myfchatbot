import os
import time
import re
import streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# ── [고정] 페이지 설정 ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="종근당의 챗봇",
    page_icon="💻",
    layout="centered",
)

# ── 사이드바: 파라미터 설정 ──────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 설정")

    # Assistants API 특성상 상용 파라미터 일부 조정 및 초기화 버튼 배치
    max_tokens = st.slider("최대 토큰 (max_tokens)", 100, 2000, 1000)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
    top_p = st.slider("Top-p", 0.0, 1.0, 0.95, 0.01)

    st.divider()
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        if "thread_id" in st.session_state:
            del st.session_state["thread_id"]
        st.rerun()

# ── Azure OpenAI 클라이언트 초기화 ──────────────────────────────────────────
endpoint = os.getenv("AZURE_OAI_ENDPOINT", "")
api_key = os.getenv("AZURE_OAI_KEY", "")
deployment = os.getenv("AZURE_OAI_DEPLOYMENT", "")

if not endpoint or not api_key or not deployment:
    st.error("❌ 환경변수가 설정되지 않았습니다. AZURE_OAI_ENDPOINT / AZURE_OAI_KEY / AZURE_OAI_DEPLOYMENT 를 확인하세요.")
    st.stop()

# Assistants API를 사용하기 위해 대화 전용 api_version 설정 (혹은 2024-05-01-preview 이상 권장)
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version="2024-05-01-preview",
)

# ── Assistants 및 Thread 초기화 ─────────────────────────────────────────────
# 1. 관리형 Assistant 동적 생성 혹은 세션 보존
if "assistant_id" not in st.session_state:
    try:
        assistant = client.beta.assistants.create(
            name="종근당 데이터 분석 전문가",
            instructions="너는 데이터 분석과 통계, 그래프 시각화를 담당하는 스마트 전문가야. 그래프 작성 요청을 받으면 파이썬 code_interpreter 도구를 사용하여 정교한 차트를 그리고 시각화 결과를 제공해줘.",
            model=deployment,
            tools=[{"type": "code_interpreter"}] # Code Interpreter 활성화
        )
        st.session_state.assistant_id = assistant.id
    except Exception as e:
        st.error(f"Assistant 생성 실패. 모델/배포명을 확인하세요: {e}")
        st.stop()

# 2. 개별 유저용 대화 Thread 생성
if "thread_id" not in st.session_state:
    thread = client.beta.threads.create()
    st.session_state.thread_id = thread.id

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── [고정] 헤더 ──────────────────────────────────────────────────────────────
st.title("💻 종근당의 챗봇")
st.caption("Azure OpenAI 기반 챗봇")
st.divider()

# ── 기존 대화 출력 ───────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🙋" if msg["role"] == "user" else "💬"):
        if msg.get("content"):
            st.markdown(msg["content"])
        if msg.get("images"):
            for img_data in msg["images"]:
                st.image(img_data)

# ── 사용자 입력 및 처리 ──────────────────────────────────────────────────────
user_input = st.chat_input("메시지를 입력하세요...")

if user_input:
    with st.chat_message("user", avatar="🙋"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    # Thread에 유저 메시지 추가
    client.beta.threads.messages.create(
        thread_id=st.session_state.thread_id,
        role="user",
        content=user_input
    )

    # 실행(Run) 생성 및 스트리밍 처리
    with st.chat_message("assistant", avatar="💬"):
        placeholder = st.empty()
        full_response = ""
        image_data_list = []

        try:
            # Assistants API 스트리밍 실행
            with client.beta.threads.runs.create_and_stream(
                thread_id=st.session_state.thread_id,
                assistant_id=st.session_state.assistant_id,
                temperature=temperature,
                top_p=top_p
            ) as stream:
                for event in stream:
                    # 텍스트 실시간 출력 처리
                    if event.event == 'thread.message.delta':
                        for delta in event.data.delta.content:
                            if delta.type == 'text' and delta.text.value:
                                full_response += delta.text.value
                                placeholder.markdown(full_response + "▌")
            
            # 스트리밍 완료 후 커서 처리 정돈
            placeholder.markdown(full_response if full_response else "분석을 완료했습니다.")

            # 코드 인터프리터가 생성한 결과물(파일/이미지)이 있는지 Thread 최종 메시지 확인
            messages = client.beta.threads.messages.list(thread_id=st.session_state.thread_id)
            last_message = messages.data[0] # 가장 최근 어시스턴트 메시지

            for content_block in last_message.content:
                # 모델이 내부 가상환경에서 그려낸 차트 파일 이미지 추출
                if content_block.type == 'image_file':
                    file_id = content_block.image_file.file_id
                    # 파일 다운로드 인터페이스 호출
                    image_data = client.files.content(file_id).read()
                    image_data_list.append(image_data)
                    # 화면에 차트 렌더링
                    st.image(image_data)

            # 세션 대화기록 백업
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "images": image_data_list
            })

        except Exception as e:
            st.error(f"❌ 오류 발생: {e}")
