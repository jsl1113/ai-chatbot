import streamlit as st
import os 

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain import hub
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


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


def get_ai_message(user_memsage):
    embedding = OpenAIEmbeddings(model='text-embedding-3-large') 
    index_name = 'tax-markdown-index'
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)

    llm = ChatOpenAI(model='gpt-4o')
    prompt = hub.pull("rlm/rag-prompt")
    retriever = database.as_retriever(search_kwargs={'k': 7})

    # QA chain 만들기 
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever, 
        chain_type_kwargs={"prompt": prompt}
    )

    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다. 
        그런 경우에는 {{question}}
        사전: {dictionary}
        질문: {{question}}
    """)

    # query -> 직장인 -> 거주자 chain 추가 
    dictionary_chain = prompt | llm | StrOutputParser()
    tax_chain = {"query": dictionary_chain} | qa_chain
    ai_message = tax_chain.invoke({"question": user_memsage})
    return ai_message["result"]


# name 은 user, assistant, ai, human 등 
# 입력될 때마다, 전체 코드가 다시 싹 돈다 
if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용들을 말씀해주세요."):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role" : "user", "content" : user_question})

    with st.spinner("답변을 생성하는 중입니다"):
        ai_message = get_ai_message(user_question)

        with st.chat_message("ai"):
            st.write(ai_message)
        st.session_state.message_list.append({"role" : "ai", "content" : ai_message})
