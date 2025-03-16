import os
import streamlit as st
import pickle
import time
import langchain
import google.generativeai as genai
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

articles=[]

for i in range(3):
    url=st.sidebar.text_input(f"URL {i+1}")
    articles.append(url)

process_urls_clicked=st.sidebar.button("Process Urls")
file_path="faiss_store_gemini.pkl"

main_placeholder=st.empty()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

if process_urls_clicked:
    loaders= UnstructuredURLLoader(urls=articles)
    main_placeholder.text("Data Loading...Started...✅✅✅") 
    data=loaders.load()

    #splitting data
    text_splitter=RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', '.', ','],
    chunk_size=1000,)
    main_placeholder.text("Text Splitter...Started...✅✅✅") 
    docs=text_splitter.split_documents(data)

    #creating embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")

    vectorindex_openai=FAISS.from_documents(docs, embeddings)
    #serialize 
    pkl=vectorindex_openai.serialize_to_bytes()
    
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)


    # Saving the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(pkl, f)

#Input query field
query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            pkl=pickle.load(f)
            
            # Deserialize the FAISS index and create a retrieval question-answering chain
            vectorIndex=FAISS.deserialize_from_bytes(embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document"),
            serialized=pkl, allow_dangerous_deserialization=True )
            chain=RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorIndex.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            st.header("Answer")
            st.write(result["answer"]) #displays the answer
            sources=result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                source_list=sources.split("\n")
                for s in source_list:
                    st.write(s) #displays the source