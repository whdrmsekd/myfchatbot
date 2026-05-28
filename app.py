import os
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# ── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="종근당의 챗봇",
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

# ── 데이터셋 정의 (날씨 및 시간) ──────────────────────────────────────────────
WEATHER_DATA = {
    "tokyo": {"temperature": "10", "unit": "celsius"},
    "san francisco": {"temperature": "72", "unit": "fahrenheit"},
    "paris": {"temperature": "22", "unit": "celsius"},
    "seoul": {"temperature": "30", "unit": "celsius"},
    "cheongju": {"temperature": "24", "unit": "celsius"}
}

TIMEZONE_DATA = {
    "tokyo": "Asia/Tokyo",
    "san francisco": "America/Los_Angeles",
    "paris": "Europe/Paris",
    "seoul": "Asia/Seoul",
    "cheongju": "Asia/Seoul"
}

# ── 실행 대상 파이썬 기능(Tools) 정의 ─────────────────────────────────────────
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

def get_current_weather(location, unit=None):
    location_lower = location.lower()
    for key in WEATHER_DATA:
        if key in location_lower:
            weather = WEATHER_DATA[key]
            return json.dumps({
                "location": location,
                "temperature": weather["temperature"],
                "unit": unit if unit else weather["unit"]
            })
    return json.dumps({"location": location, "temperature": "unknown"})

def get_current_time(location):
    location_lower = location.lower()
    for key, timezone in TIMEZONE_DATA.items():
        if key in location_lower:
            current_time = datetime.now(ZoneInfo(timezone)).strftime("%Y-%m-%d %I:%M %p")
            return json.dumps({"location": location, "current_time": current_time})
    return json.dumps({"location": location, "current_time": "unknown"})

# ── 툴 스펙 명세서 리스트 구성 ────────────────────────────────────────────────
tools = [
    {"type": "function", "function": {"name": "add_numbers", "description": "두 숫자의 합을 계산합니다.", "parameters": {"type": "object", "properties": {"num1": {"type": "number"}, "num2": {"type": "number"}}, "required": ["num1", "num2"]}}},
    {"type": "function", "function": {"name": "subtract_numbers", "description": "첫 번째 숫자에서 두 번째 숫자를 뺍니다.", "parameters": {"type": "object", "properties": {"num1": {"type": "number"}, "num2": {"type": "number"}}, "required": ["num1", "num2"]}}},
    {"type": "function", "function": {"name": "multiply_numbers", "description": "두 숫자의 곱을 계산합니다.", "parameters": {"type": "object", "properties": {"num1": {"type": "number"}, "num2": {"type": "number"}}, "required": ["num1", "num2"]}}},
    {"type": "function", "function": {"name": "divide_numbers", "description": "첫 번째 숫자를 두 번째 숫자로 나눕니다.", "parameters": {"type": "object", "properties": {"num1": {"type": "number"}, "num2": {"type": "number"}}, "required": ["num1", "num2"]}}},
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "지정된 도시의 현재 날씨와 기온 정보를 가져옵니다. 지원 도시: Seoul, Tokyo, Paris, San Francisco, Cheongju",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "도시 이름 (예: Seoul 또는 서울)"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "지정된 도시의 현재 날짜와 시간(연, 월, 일, 시, 분) 정보를 가져옵니다. 지원 도시: Seoul, Tokyo, Paris, San Francisco, Cheongju",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "도시 이름 (예: Tokyo 또는 도쿄)"}
                },
                "required": ["location"]
            }
        }
    }
]

SYSTEM_PROMPT = {
    "role": "system",
    "content": "너는 컴퓨터, 연산, 날씨 및 시간 정보를 안내하는 올라운더 전문가야. 사용자의 요청에 따라 적절한 도구를 호출해서 정확한 정보를 실시간으로 답변해줘.",
}

# ── 세션 상태 초기화 ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── 헤더 ────────────────────────────────────────────────────────────────────
st.title("💻 종근당의 챗봇")
st.caption("Azure OpenAI 기반 챗봇")
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

    # ── API용 메시지 버퍼 구성 (시스템 프롬프트 + 슬라이딩 윈도우) ──────────────
    message_limit = message_cnt * 2
    history = st.session_state.messages
    windowed = history[-message_limit:] if len(history) > message_limit else history

    chat_payload = [SYSTEM_PROMPT] + windowed

    # ── AI 응답 처리 (스트리밍 및 멀티 툴 콜링 대응) ───────────────────────────
    with st.chat_message("assistant", avatar="💬"):
        placeholder = st.empty()
        full_response = ""
        tool_calls_chunks = {}

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
                
                # 1. 텍스트 출력 스트리밍
                if delta and delta.content:
                    full_response += delta.content
                    placeholder.markdown(full_response + "▌")
                
                # 2. 실시간 툴 콜 데이터 조각 수집
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

            placeholder.markdown(full_response if full_response else "🛠️ 필요한 정보를 가져오는 중...")

            # 3. 툴 콜 실행 및 피드백 처리
            if tool_calls_chunks:
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

                for tool in built_tool_calls:
                    func_name = tool["function"]["name"]
                    func_args = json.loads(tool["function"]["arguments"])
                    
                    # 함수 종류에 따른 인자 분기 처리
                    if func_name in ["add_numbers", "subtract_numbers", "multiply_numbers", "divide_numbers"]:
                        num1 = func_args.get("num1")
                        num2 = func_args.get("num2")
                        if func_name == "add_numbers": result_content = add_numbers(num1, num2)
                        elif func_name == "subtract_numbers": result_content = subtract_numbers(num1, num2)
                        elif func_name == "multiply_numbers": result_content = multiply_numbers(num1, num2)
                        elif func_name == "divide_numbers": result_content = divide_numbers(num1, num2)
                    
                    elif func_name == "get_current_weather":
                        result_content = get_current_weather(func_args.get("location"), func_args.get("unit"))
                        
                    elif func_name == "get_current_time":
                        result_content = get_current_time(func_args.get("location"))
                        
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

                # 4. 외부 도구 데이터 취합 후 최종 답변 스트리밍
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
                st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"❌ 오류 발생: {e}")
