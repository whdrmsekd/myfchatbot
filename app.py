import os
import time
import streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# ── [고정] 페이지 설정 ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="종근당 통합 AI 에이전트",
    page_icon="📊",
    layout="centered",
)

# ── 환경변수 로드 ──────────────────────────────────────────────────────────
endpoint = os.getenv("AZURE_OAI_ENDPOINT") or os.getenv("ENDPOINT_URL", "")
api_key = os.getenv("AZURE_OAI_KEY") or os.getenv("AZURE_OPENAI_API_KEY", "")
deployment = os.getenv("AZURE_OAI_DEPLOYMENT") or os.getenv("DEPLOYMENT_NAME", "")

if not endpoint or not api_key or not deployment:
    st.error("❌ OpenAI 환경변수가 설정되지 않았습니다. .env 파일을 확인하세요.")
    st.stop()

# 💡 Azure에서 가장 안정적으로 서포트하는 API 버전으로 고정
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version="2024-08-01-preview", 
)

# ── 세션 상태 및 비동기 스레드 관리 ──────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit 재실행 시 중복 생성을 막기 위해 session_state 검증을 강화합니다.
if "thread_id" not in st.session_state:
    try:
        new_thread = client.beta.threads.create()
        st.session_state.thread_id = new_thread.id
    except Exception as e:
        st.error(f"스레드 생성 실패: {e}")
        st.stop()

if "assistant_id" not in st.session_state:
    try:
        # 💡 에러의 주원인이 되는 복잡한 RAG 구조를 걷어내고, 그래프/수식 연산 장치만 깔끔하게 탑재
        assistant = client.beta.assistants.create(
            name="종근당 데이터 분석가",
            instructions=(
                "너는 종근당의 [데이터 분석 및 수식 계산 전문가]이다. "
                "사용자가 수식 계산, 통계, 데이터 요약 및 그래프(차트) 생성을 요청하면 "
                "반드시 code_interpreter 도구를 사용하여 파이썬 코드를 실행하고 시각화 결과물 이미지나 계산값을 도출하라."
            ),
            model=deployment,
            tools=[{"type": "code_interpreter"}]
        )
        st.session_state.assistant_id = assistant.id
    except Exception as e:
        st.error(f"❌ 에이전트 생성 중 BadRequestError 발생: {e}\n리소스의 Assistants 가용 여부를 확인하세요.")
        st.stop()

# ── 사이드바 ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 관리")
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        try:
            new_thread = client.beta.threads.create()
            st.session_state.thread_id = new_thread.id
        except:
            pass
        st.rerun()

# ── [고정] 헤더 ──────────────────────────────────────────────────────────────
st.title("종근당의 데이터 분석 챗봇")
st.caption("Azure Assistants API 기반 [파이썬 데이터 계산 + 실시간 그래프 시각화]")
st.divider()

# ── 기존 대화 출력 (화면 유지용) ──────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🙋" if msg["role"] == "user" else "💬"):
        if msg.get("content"):
            st.markdown(msg["content"])
        if msg.get("image_data"):
            st.image(msg["image_data"], caption="에이전트가 생성한 그래프")

# ── 사용자 입력 및 에이전트 구동 ─────────────────────────────────────────────
user_input = st.chat_input("사칙연산 수식이나 '데이터로 그래프 그려줘'라고 요청하세요...")

if user_input:
    # 1. 유저 입력 출력 및 세션 저장
    with st.chat_message("user", avatar="🙋"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        # 2. 클라우드 대화방에 메시지 추가
        client.beta.threads.messages.create(
            thread_id=st.session_state.thread_id,
            role="user",
            content=user_input
        )

        # 3. 에이전트 러닝(Run) 시작
        with st.chat_message("assistant", avatar="💬"):
            status_placeholder = st.empty()
            status_placeholder.markdown("🏃 파이썬 코드를 구동하여 연산을 수행 중입니다...")

            run = client.beta.threads.runs.create(
                thread_id=st.session_state.thread_id,
                assistant_id=st.session_state.assistant_id
            )

            # 4. 폴링(Polling) 루프: 완료될 때까지 상태 대기
            while run.status in ["queued", "in_progress"]:
                time.sleep(1)
                run = client.beta.threads.runs.retrieve(
                    thread_id=st.session_state.thread_id,
                    run_id=run.id
                )

            # 5. 실행 완료 후 결과 데이터 분석 및 화면 출력
            if run.status == "completed":
                status_placeholder.empty() 
                
                messages = client.beta.threads.messages.list(thread_id=st.session_state.thread_id)
                latest_message = messages.data[0]

                response_text = ""
                image_bytes_list = []

                for content_block in latest_message.content:
                    if content_block.type == "text":
                        response_text += content_block.text.value
                    elif content_block.type == "image_file":
                        file_id = content_block.image_file.file_id
                        image_data = client.files.content(file_id).read()
                        image_bytes_list.append(image_data)

                # 화면에 결과 텍스트 출력
                if response_text:
                    st.markdown(response_text)
                
                # 화면에 생성된 그래프 이미지 출력
                for img_bytes in image_bytes_list:
                    st.image(img_bytes, caption="에이전트가 생성한 그래프")

                # 세션 이력에 통합 저장
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "image_data": image_bytes_list[0] if image_bytes_list else None
                })
                
            else:
                status_placeholder.error(f"❌ 에이전트 수행 실패. 상태: {run.status}")

    except Exception as e:
        st.error(f"❌ 통신 오류 발생: {e}")
