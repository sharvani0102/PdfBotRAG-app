import streamlit as st 
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv

load_dotenv()

## load the GROQ and OPEN AI API key 
#st.write(st.secrets["OPENAI_API_KEY"])
#st.write(os.environ["OPENAI_API_KEY"]==st.secrets["OPENAI_API_KEY"])
#st.write(st.secrets["groq_api_key"]["my_key"])
#os.environ['OPENAI_API_KEY']=st.secrets["OPEN_AI_API_KEY"]
groq_api_key= st.secrets["groq_api_key"]["my_key"]

st.title("Chat With Your Pdfs")

llm = ChatGroq(groq_api_key = groq_api_key,
               model_name = "Llama3-8b-8192")

promptTemplate = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context. 
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    
    """
)

def vector_embeddings():
    
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        #st.session_state.embeddings = HuggingFaceInstructEmbeddings(model_name = 'dunzhang/stella_en_1.5B_v5')
        st.session_state.loader = PyPDFDirectoryLoader("./data")
        st.session_state.documents = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap = 200 )
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.documents[:20])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

prompt1 = st.text_input("Ask me Anything");

st.text("This app contains uploaded documents for rules of the board games Monopoly and Catan")
if st.button("Process Games Rule Documents"):
    vector_embeddings()
    st.write("vector store db is ready")
    

import time  

if prompt1:
    document_chain = create_stuff_documents_chain(llm,promptTemplate)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
   
    start = time.process_time()
    response = retrieval_chain.invoke({'input' : prompt1})
    print("response time : ", time.process_time()-start)
    st.write(response['answer'])
    
    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")


