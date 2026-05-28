import os
import json
import time
import re
from datetime import datetime
from zoneinfo import ZoneInfo
import streamlit as st
import pandas as pd
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
def add_numbers(num1, num2): return json.dumps({"operation": "add", "result": num1 + num2})
def subtract_numbers(num1, num2): return json.dumps({"operation": "subtract", "result": num1 - num2})
def multiply_numbers(num1, num2): return json.dumps({"operation": "multiply", "result": num1 * num2})
def divide_numbers(num1, num2):
    if num2 == 0: return json.dumps({"operation": "divide", "error": "Cannot divide by zero."})
    return json.dumps({"operation": "divide", "result": num1 / num2})

def get_current_weather(location, unit=None):
    location_lower = location.lower()
    for key in WEATHER_DATA:
        if key in location_lower:
            weather = WEATHER_DATA[key]
            return json.dumps({"location": location, "temperature": weather["temperature"], "unit": unit if unit else weather["unit"]})
    return json.dumps({"location": location, "temperature": "unknown"})

def get_current_time(location):
    location_lower = location.lower()
    for key, timezone in TIMEZONE_DATA.items():
        if key in location_lower:
            current_time = datetime.now(ZoneInfo(timezone)).strftime("%Y-%m-%d %I:%M %p")
            return json.dumps({"location": location, "current_time": current_time})
    return json.dumps({"location": location, "current_time": "unknown"})

# 🛠️ [수정] X축 텍스트 정렬 꼬임 현상 방지 로직 보강
def draw_line_chart(title, values, labels=None):
    try:
        if labels and len(labels) == len(values):
            # 만약 라벨이 모두 숫자로만 구성되어 있다면 숫자형 데이터 타입으로 변환 시도
            # (문자열 '11'이 '101'보다 앞에 오게 만드는 사전순 정렬 오류 방지)
            try:
                numeric_labels = [float(str(l).replace(",", "")) for l in labels]
                # 소수점 뒤가 0으로 끝나면 깔끔하게 정수로 치환
                labels = [int(n) if n.is_integer() else n for n in numeric_labels]
            except ValueError:
                # 숫자로 변환이 불가능한 순수 텍스트(예: '1월', '2월')라면 
                # 데이터가 들어온 순서 고유값을 보존하기 위해 범주형 변환 처리
                labels = pd.Categorical(labels, categories=labels, ordered=True)
                
            df = pd.DataFrame({"값 (Value)": values}, index=labels)
            # 인덱스(X축) 기준으로 정렬하여 차트 뒤틀림 완전 차단
            df = df.sort_index()
        else:
            df = pd.DataFrame({"값 (Value)": values})
            
        st.subheader(f"📊 {title}")
        st.line_chart(df)
        return json.dumps({"status": "success", "message": f"'{title}' 그래프를 화면에 성공적으로 렌더링했습니다. 이미지 주소나 base64 코드를 유저에게 절대 직접 출력하지 마세요."})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ── 툴 스펙 명세서 리스트 구성 ────────────────────────────────────────────────
tools = [
    {"type": "function", "function": {"name": "add_numbers", "description": "두 숫자의 합을 계산합니다.", "parameters": {"type": "object", "properties": {"num1": {"type": "number"}, "num2": {"type": "number"}}, "required": ["num1", "num2"]}}},
    {"type": "function", "function": {"name": "subtract_numbers", "description": "첫 번째 숫자에서 두 번째 숫자를 뺍니다.", "parameters": {"type": "object", "properties": {"num1": {"type": "number"}, "num2": {"type": "number"}}, "required": ["num1", "num2"]}}},
    {"type": "function", "function": {"name": "multiply_numbers", "description": "두 숫자의 곱을 계산합니다.", "parameters": {"type": "object", "properties": {"num1": {"type": "number"}, "num2": {"type": "number"}}, "required": ["num1", "num2"]}}},
    {"type": "function", "function": {"name": "divide_numbers", "description": "첫 번째 숫자를 두 번째 숫자로 나눕니다.", "parameters": {"type": "object", "properties": {"num1": {"type": "number"}, "num2": {"type": "number"}}, "required": ["num1", "num2"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "지정된 도시의 현재 날씨 기온 정보를 가져옵니다. 지원도시: Seoul, Tokyo, Paris, San Francisco, Cheongju", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "지정된 도시의 현재 날짜와 시간 정보를 가져옵니다. 지원도시: Seoul, Tokyo, Paris, San Francisco, Cheongju", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {
        "type": "function",
        "function": {
            "name": "draw_line_chart",
            "description": "수치형 데이터를 기반으로 선 그래프를 화면에 그려줍니다. 함수 응답에 이미지 코드가 포함되더라도 최종 답변에 base64 텍스트 데이터를 절대 포함하여 출력하지 마십시오.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "그래프의 제목"},
                    "values": {"type": "array", "items": {"type": "number"}, "description": "숫자 배열 데이터"},
                    "labels": {"type": "array", "items": {"type": "string"}, "description": "X축 라벨 배열 (반드시 데이터 순서 혹은 오름차순 순서 규칙을 명확히 정의해서 전달하세요)"}
                },
                "required": ["title", "values"]
            }
        }
    }
]

SYSTEM_PROMPT = {
    "role": "system",
    "content": "너는 데이터 분석, 연산, 날씨 조회를 담당하는 스마트 전문가야. 'draw_line_chart'를 통해 그래프를 그렸다면 이미지 데이터나 (data:image/png;base64...) 형태의 텍스트 주소는 최종 응답에 절대 포함하지 말고, 그래프를 화면에 성공적으로 그렸다는 요약 안내 멘트만 단정하게 완성해줘.",
}

# ── 세션 상태 초기화 ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── [고정] 헤더 ──────────────────────────────────────────────────────────────
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

    message_limit = message_cnt * 2
    history = st.session_state.messages
    windowed = history[-message_limit:] if len(history) > message_limit else history

    chat_payload = [SYSTEM_PROMPT] + windowed

    # ── AI 응답 처리 ─────────────────────────────────────────────────────────
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
                
                if delta and delta.content:
                    full_response += delta.content
                    clean_stream = re.sub(r'!\[.*?\]\(data:image\/.*?base64,.*?\)', '', full_response, flags=re.DOTALL)
                    clean_stream = re.sub(r'\(data:image\/.*?base64,.*?\)', '', clean_stream, flags=re.DOTALL)
                    placeholder.markdown(clean_stream + "▌")
                
                if delta and delta.tool_calls:
                    for tool_chunk in delta.tool_calls:
                        idx = tool_chunk.index
                        if idx not in tool_calls_chunks:
                            tool_calls_chunks[idx] = {"id": tool_chunk.id, "name": tool_chunk.function.name, "arguments": ""}
                        if tool_chunk.function.arguments:
                            tool_calls_chunks[idx]["arguments"] += tool_chunk.function.arguments

            full_response = re.sub(r'!\[.*?\]\(data:image\/.*?base64,.*?\)', '', full_response, flags=re.DOTALL)
            full_response = re.sub(r'\(data:image\/.*?base64,.*?\)', '', full_response, flags=re.DOTALL)
            placeholder.markdown(full_response if full_response else "🛠️ 도구를 가동 중입니다...")

            if tool_calls_chunks:
                built_tool_calls = [
                    {"id": chunk_data["id"], "type": "function", "function": {"name": chunk_data["name"], "arguments": chunk_data["arguments"]}}
                    for chunk_data in tool_calls_chunks.values()
                ]
                
                assistant_message = {"role": "assistant", "content": full_response or None, "tool_calls": built_tool_calls}
                st.session_state.messages.append(assistant_message)
                chat_payload.append(assistant_message)

                for tool in built_tool_calls:
                    func_name = tool["function"]["name"]
                    func_args = json.loads(tool["function"]["arguments"])
                    
                    if func_name in ["add_numbers", "subtract_numbers", "multiply_numbers", "divide_numbers"]:
                        num1, num2 = func_args.get("num1"), func_args.get("num2")
                        if func_name == "add_numbers": result_content = add_numbers(num1, num2)
                        elif func_name == "subtract_numbers": result_content = subtract_numbers(num1, num2)
                        elif func_name == "multiply_numbers": result_content = multiply_numbers(num1, num2)
                        elif func_name == "divide_numbers": result_content = divide_numbers(num1, num2)
                    
                    elif func_name == "get_current_weather":
                        result_content = get_current_weather(func_args.get("location"))
                        
                    elif func_name == "get_current_time":
                        result_content = get_current_time(func_args.get("location"))
                    
                    elif func_name == "draw_line_chart":
                        result_content = draw_line_chart(
                            title=func_args.get("title"),
                            values=func_args.get("values"),
                            labels=func_args.get("labels")
                        )
                        
                    else:
                        result_content = json.dumps({"error": "Unknown function"})

                    tool_response_message = {"role": "tool", "tool_call_id": tool["id"], "name": func_name, "content": result_content}
                    st.session_state.messages.append(tool_response_message)
                    chat_payload.append(tool_response_message)

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
                    if not chunk.choices: continue
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        final_response += delta.content
                        clean_final = re.sub(r'!\[.*?\]\(data:image\/.*?base64,.*?\)', '', final_response, flags=re.DOTALL)
                        clean_final = re.sub(r'\(data:image\/.*?base64,.*?\)', '', clean_final, flags=re.DOTALL)
                        final_placeholder.markdown(clean_final + "▌")

                final_response = re.sub(r'!\[.*?\]\(data:image\/.*?base64,.*?\)', '', final_response, flags=re.DOTALL)
                final_response = re.sub(r'\(data:image\/.*?base64,.*?\)', '', final_response, flags=re.DOTALL)
                
                final_placeholder.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
            
            else:
                st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"❌ 오류 발생: {e}")
