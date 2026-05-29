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

search_endpoint = os.getenv("SEARCH_ENDPOINT", "")
search_key = os.getenv("SEARCH_KEY", "")
search_index = os.getenv("SEARCH_INDEX_NAME", "rag-10ai017safety")

if not endpoint or not api_key or not deployment:
    st.error("❌ OpenAI 환경변수가 설정되지 않았습니다. .env 파일을 확인하세요.")
    st.stop()

# Assistants 클라이언트 초기화 (v2 규격 사용)
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version="2024-05-01-preview", 
)

# ── 세션 상태 및 비동기 스레드 관리 ──────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    # 클라우드 상에 고유 대화방(Thread) 생성
    new_thread = client.beta.threads.create()
    st.session_state.thread_id = new_thread.id

if "assistant_id" not in st.session_state:
    with st.spinner("🤖 종근당 분석 에이전트 엔진 최적화 중..."):
        # RAG용 Search 도구와 데이터 분석용 Code Interpreter 도구를 법적 규격에 맞게 듀얼 장착
        assistant = client.beta.assistants.create(
            name="종근당 통합 전문가",
            instructions=(
                "너는 종근당의 [데이터 분석 및 안전규정 통합 에이전트]이다. "
                "1. 사용자가 수식 계산, 통계, 데이터 요약 및 그래프(차트) 생성을 요청하면 "
                "반드시 code_interpreter 도구를 사용하여 파이썬 코드를 실행하고 시각화 결과물 이미지나 계산값을 도출하라. "
                "2. 안전 지침이나 사내 규정 검색을 원하면 azure_search 도구를 사용해 사실 기반으로 답변하라."
            ),
            model=deployment,
            tools=[
                {"type": "code_interpreter"},
                {
                    "type": "azure_search",
                    "azure_search": {
                        "endpoint": search_endpoint,
                        "key": search_key,
                        "index_name": search_index
                    }
                }
            ]
        )
        st.session_state.assistant_id = assistant.id

# ── 사이드바 ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 관리")
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        # 새 대화방 할당
        new_thread = client.beta.threads.create()
        st.session_state.thread_id = new_thread.id
        st.rerun()

# ── [고정] 헤더 ──────────────────────────────────────────────────────────────
st.title("종근당의 통합 AI 챗봇")
st.caption("Assistants API 기반 [RAG 문서 검색 + 파이썬 데이터 계산 + 그래프 시각화]")
st.divider()

# ── 기존 대화 출력 (화면 유지용) ──────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🙋" if msg["role"] == "user" else "💬"):
        if msg.get("content"):
            st.markdown(msg["content"])
        if msg.get("image_data"):
            st.image(msg["image_data"], caption="에이전트가 생성한 그래프")

# ── 사용자 입력 및 에이전트 구동 ─────────────────────────────────────────────
user_input = st.chat_input("규정 검색, 사칙연산, 또는 '표 데이터로 그래프 그려줘'라고 요청하세요...")

if user_input:
    # 1. 유저 입력 출력 및 세션 저장
    with st.chat_message("user", avatar="🙋"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 2. 클라우드 대화방에 메시지 추가
    client.beta.threads.messages.create(
        thread_id=st.session_state.thread_id,
        role="user",
        content=user_input
    )

    # 3. 에이전트 러닝(Run) 시작
    with st.chat_message("assistant", avatar="💬"):
        status_placeholder = st.empty()
        status_placeholder.markdown("🏃 에이전트가 도구를 선택하고 연산을 수행 중입니다...")

        run = client.beta.threads.runs.create(
            thread_id=st.session_state.thread_id,
            assistant_id=st.session_state.assistant_id
        )

        # 4. 폴링(Polling) 루프: 완료될 때까지 상태 대기 (코드 실행 및 RAG 완료 시점 추적)
        while run.status in ["queued", "in_progress"]:
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(
                thread_id=st.session_state.thread_id,
                run_id=run.id
            )

        # 5. 실행 완료 후 결과 데이터 분석 및 화면 출력
        if run.status == "completed":
            status_placeholder.empty() # 대기 문구 제거
            
            # 대화방의 최신 메시지 가져오기
            messages = client.beta.threads.messages.list(thread_id=st.session_state.thread_id)
            latest_message = messages.data[0]

            response_text = ""
            image_bytes_list = []

            # AI가 반환한 컨텐츠 배열 파싱 (텍스트 글자와 그래프 이미지를 정밀 구별)
            for content_block in latest_message.content:
                if content_block.type == "text":
                    response_text += content_block.text.value
                elif content_block.type == "image_file":
                    # AI가 가상 머신 안에서 그린 그래프 파일 ID 추출
                    file_id = content_block.image_file.file_id
                    # 파일 API를 통해 실제 이미지 바이너리 획득
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
            status_placeholder.error(f"❌ 에이전트 수행 중 문제가 발생했습니다. 상태: {run.status}")
