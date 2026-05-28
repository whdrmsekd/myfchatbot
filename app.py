import os
import json
import streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# ── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="컴퓨터 및 연산 전문가 AI",
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

# ── 사칙연산 실제 파이썬 함수 및 툴 스펙 정의 ─────────────────────────────────
def add_numbers(num1, num2):
    return json.dumps({"operation": "add", "result": num1 + num2})

def subtract_numbers(num1, num2):
    return json.dumps({"operation": "subtract", "result": num1 - num2})

def multiply_numbers(num1, num2):
    return json.dumps({"operation": "multiply", "result": num1 * num2})

def divide_numbers(num1, num2):
    if num2 == 0:
        return json.dumps({"operation": "divide", "error": "Cannot divide by zero."})
    return json.dumps({"operation": "divide", "result": num1 / num2})

tools = [
    {"type": "function", "function": {"name": "add_numbers", "description": "두 숫자의 합을 계산합니다.", "parameters": {"type": "object", "properties": {"num1": {"type": "number"}, "num2": {"type": "number"}}, "required": ["num1", "num2"]}}},
    {"type": "function", "function": {"name": "subtract_numbers", "description": "첫 번째 숫자에서 두 번째 숫자를 뺍니다.", "parameters": {"type": "object", "properties": {"num1": {"type": "number"}, "num2": {"type": "number"}}, "required": ["num1", "num2"]}}},
    {"type": "function", "function": {"name": "multiply_numbers", "description": "두 숫자의 곱을 계산합니다.", "parameters": {"type": "object", "properties": {"num1": {"type": "number"}, "num2": {"type": "number"}}, "required": ["num1", "num2"]}}},
    {"type": "function", "function": {"name": "divide_numbers", "description": "첫 번째 숫자를 두 번째 숫자로 나눕니다.", "parameters": {"type": "object", "properties": {"num1": {"type": "number"}, "num2": {"type": "number"}}, "required": ["num1", "num2"]}}}
]

SYSTEM_PROMPT = {
    "role": "system",
    "content": "너는 컴퓨터 및 복잡한 수식 연산 전문가야. 사칙연산 요청이 들어오면 제공된 도구를 사용해서 정확히 계산해줘.",
}

# ── 세션 상태 초기화 ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []   # OpenAI API 규격에 맞는 딕셔너리 리스트로 통합 변경

# ── 헤더 ────────────────────────────────────────────────────────────────────
st.title("💻 컴퓨터 및 연산 전문가 AI")
st.caption("Azure OpenAI 기반 챗봇 · 사칙연산 기능 툴 콜링 지원")
st.divider()

# ── 기존 대화 출력 ───────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"] and msg.get("content"):
        avatar = "🙋" if msg["role"] == "user" else "💬"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

# ── 사용자 입력 ──────────────────────────────────────────────────────────────
user_input = st.chat_input("메시지를 입력하세요...")

if user_input:
    with st.chat_message("user", avatar="🙋"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    # ── API에 보낼 메시지 구성 (시스템 + 슬라이딩 윈도우) ───────────────────
    message_limit = message_cnt * 2
    history = st.session_state.messages
    windowed = history[-message_limit:] if len(history) > message_limit else history

    chat_payload = [SYSTEM_PROMPT] + windowed

    # ── AI 응답 처리 (스트리밍 + 툴 콜링 대응) ─────────────────────────────────
    with st.chat_message("assistant", avatar="💬"):
        placeholder = st.empty()
        full_response = ""
        tool_calls_chunks = {} # 스트리밍되는 툴 콜 조각들을 모을 그릇

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
                tools=tools,
                tool_choice="auto",
                stream=True,
            )

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                
                # 1. 일반 텍스트 응답 스트리밍
                if delta and delta.content:
                    full_response += delta.content
                    placeholder.markdown(full_response + "▌")
                
                # 2. 툴 콜링 데이터 스트리밍 수집
                if delta and delta.tool_calls:
                    for tool_chunk in delta.tool_calls:
                        idx = tool_chunk.index
                        if idx not in tool_calls_chunks:
                            tool_calls_chunks[idx] = {
                                "id": tool_chunk.id,
                                "name": tool_chunk.function.name,
                                "arguments": ""
                            }
                        if tool_chunk.function.arguments:
                            tool_calls_chunks[idx]["arguments"] += tool_chunk.function.arguments

            placeholder.markdown(full_response if full_response else "🛠️ 계산 도구를 가져오는 중...")

            # 3. 수집된 툴 콜링이 있다면 실행 시점
            if tool_calls_chunks:
                # 모델의 tool_calls 요청 명세를 메시지 내역에 추가
                built_tool_calls = [
                    {
                        "id": chunk_data["id"],
                        "type": "function",
                        "function": {"name": chunk_data["name"], "arguments": chunk_data["arguments"]}
                    }
                    for chunk_data in tool_calls_chunks.values()
                ]
                
                assistant_message = {"role": "assistant", "content": full_response or None, "tool_calls": built_tool_calls}
                st.session_state.messages.append(assistant_message)
                chat_payload.append(assistant_message)

                # 각 툴을 순서대로 실행하고 결과를 페이로드에 추가
                for tool in built_tool_calls:
                    func_name = tool["function"]["name"]
                    func_args = json.loads(tool["function"]["arguments"])
                    num1 = func_args.get("num1")
                    num2 = func_args.get("num2")

                    if func_name == "add_numbers":
                        result_content = add_numbers(num1, num2)
                    elif func_name == "subtract_numbers":
                        result_content = subtract_numbers(num1, num2)
                    elif func_name == "multiply_numbers":
                        result_content = multiply_numbers(num1, num2)
                    elif func_name == "divide_numbers":
                        result_content = divide_numbers(num1, num2)
                    else:
                        result_content = json.dumps({"error": "Unknown function"})

                    tool_response_message = {
                        "role": "tool",
                        "tool_call_id": tool["id"],
                        "name": func_name,
                        "content": result_content
                    }
                    st.session_state.messages.append(tool_response_message)
                    chat_payload.append(tool_response_message)

                # 4. 툴 연산 결과를 바탕으로 두 번째 대답 스트리밍 (최종 답변 생성)
                final_placeholder = st.empty()
                final_response = ""

                second_stream = client.chat.completions.create(
                    model=deployment,
                    messages=chat_payload,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                )

                for chunk in second_stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        final_response += delta.content
                        final_placeholder.markdown(final_response + "▌")

                final_placeholder.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
            
            else:
                # 툴 콜링이 발생하지 않은 일반 대화인 경우
                st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"❌ 오류 발생: {e}")
