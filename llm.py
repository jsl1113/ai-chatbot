import os 

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


def get_retriever():
    embedding = OpenAIEmbeddings(model='text-embedding-3-large') 
    index_name = 'tax-markdown-index'
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    retriever = database.as_retriever(search_kwargs={'k': 7})
    return retriever


def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)
    return llm


def get_dictionary_chain():
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다. 
        그런 경우에는 {{question}}
        사전: {dictionary}
        질문: {{question}}
    """)
    # query -> 직장인 -> 거주자 chain 추가 
    dictionary_chain = prompt | get_llm() | StrOutputParser()
    return dictionary_chain


def get_qa_chain():
    prompt = hub.pull("rlm/rag-prompt")

    # QA chain 만들기 
    qa_chain = RetrievalQA.from_chain_type(
        llm=get_llm(), 
        retriever=get_retriever(), 
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain


def get_ai_message(user_memsage):
    tax_chain = {"query": get_dictionary_chain()} | get_qa_chain()
    ai_message = tax_chain.invoke({"question": user_memsage})
    return ai_message["result"]