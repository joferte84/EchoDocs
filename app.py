import streamlit as st
import os

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

st.set_page_config(page_title='Pregunta a tu PDF')
st.header("Pregunta a tu PDF")
OPENAI_API_KEY = st.text_input('OpenAI API Key', type='password')

if 'qa_history' not in st.session_state:
    st.session_state['qa_history'] = []
    
if 'pdf_files' not in st.session_state:
    st.session_state['pdf_files'] = []

new_pdf_objs = st.file_uploader("Carga tu documento", type="pdf", accept_multiple_files=True)

if new_pdf_objs:
    st.session_state['pdf_files'].extend(new_pdf_objs)

user_question = st.text_input("Haz una pregunta sobre tus PDFs:")

@st.cache_resource
def create_embeddings(pdf_files):
    all_chunks = []
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() if page.extract_text() else ""

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)

    knowledge_base = FAISS.from_texts(all_chunks, embeddings_model)
    return knowledge_base

if st.session_state['pdf_files'] and user_question:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    knowledge_base = create_embeddings(st.session_state['pdf_files'])
    docs = knowledge_base.similarity_search(user_question, 3)
    llm = ChatOpenAI(model_name='gpt-3.5-turbo')
    chain = load_qa_chain(llm, chain_type="stuff")
    respuesta = chain.run(input_documents=docs, question=user_question)
    
    st.write(respuesta)  
    
    st.session_state['qa_history'].append((user_question, respuesta))  
    
    st.sidebar.write("Historial de Preguntas:") 
    for i, (q, a) in enumerate(reversed(st.session_state['qa_history']), start=1):
        with st.sidebar.expander(f"Pregunta {i}: {q}"):
            st.write(a)
