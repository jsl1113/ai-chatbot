import streamlit as st

from dotenv import load_dotenv
from llm import get_ai_response

load_dotenv() # 환경변수 불러오기 

# 페이지 컴픽 
st.set_page_config(page_title="소득세 챗봇", page_icon="👻")

# 타이틀 설정
st.title("👻소득세 챗봇👻") # h1 태그로 생성됨 
st.caption("소득세에 관련된 모든 것을 답해드려요!")


# 입력된 채팅들을 Session State 에 저장 
# message_list 에 채팅한 값들을 계속 넣어줌 
if 'message_list' not in st.session_state:
    st.session_state.message_list = []

# UI 변경사항 업데이트 
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# name 은 user, assistant, ai, human 등 
# 입력될 때마다, 전체 코드가 다시 싹 돈다 
if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용들을 말씀해주세요."):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role" : "user", "content" : user_question})

    with st.spinner("답변을 생성하는 중입니다"):
        ai_response = get_ai_response(user_question)

        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
            st.session_state.message_list.append({"role" : "ai", "content" : ai_message})
