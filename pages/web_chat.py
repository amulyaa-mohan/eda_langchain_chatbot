# pages/web_chat.py — FIXED: output_key + clean memory handling
import os
import requests
import validators
import streamlit as st
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_huggingface import HuggingFaceEmbeddings
from src.config.settings import Settings

# ------------------------------------------------------------------
# 1. LLM & EMBEDDINGS
# ------------------------------------------------------------------
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=Settings().GOOGLE_API_KEY,
        temperature=0
    )

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ------------------------------------------------------------------
# 2. SCRAPE USING JINA.AI (returns clean Markdown)
# ------------------------------------------------------------------
def scrape_website(url: str) -> str:
    try:
        resp = requests.get(f"https://r.jina.ai/{url}")
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        st.error(f"Failed to scrape {url}: {e}")
        return ""

# ------------------------------------------------------------------
# 3. VECTOR DB (cached)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Analyzing websites...", ttl=3600)
def setup_vectordb(websites):
    docs = []
    for url in websites:
        txt = scrape_website(url)
        if txt:
            docs.append(Document(page_content=txt, metadata={"source": url}))
    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    return DocArrayInMemorySearch.from_documents(splits, get_embeddings())

# ------------------------------------------------------------------
# 4. MAIN APP
# ------------------------------------------------------------------
def run():
    st.title("Web Intelligence Bot")
    st.caption("Enter website URLs → Ask anything about their content")

    # ---------- Sidebar ----------
    if "websites" not in st.session_state:
        st.session_state.websites = []

    with st.sidebar:
        st.header("Add Websites")
        url_input = st.text_input("Enter URL", placeholder="https://example.com")
        if st.button("Add Website"):
            if url_input and validators.url(url_input):
                if url_input not in st.session_state.websites:
                    st.session_state.websites.append(url_input)
                    st.success(f"Added: {url_input}")
                else:
                    st.info("URL already added.")
            else:
                st.error("Invalid URL!")

        if st.button("Clear All"):
            st.session_state.websites = []

        if st.session_state.websites:
            st.write("**Active URLs:**")
            for u in st.session_state.websites:
                st.write(f"- {u}")

    # ---------- No URLs ----------
    if not st.session_state.websites:
        st.info("Add at least one website URL to start.")
        return

    # ---------- Build DB ----------
    vectordb = setup_vectordb(st.session_state.websites)
    if not vectordb:
        st.error("Failed to process websites.")
        return

    # ---------- Retrieval + Memory ----------
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",          # <-- THIS FIXES THE ERROR
        return_messages=True
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    # ---------- Chat ----------
    # Initialise chat history for this tool
    if "web_messages" not in st.session_state:
        st.session_state.web_messages = []

    for msg in st.session_state.web_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about the websites…"):
        st.session_state.web_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching websites…"):
                result = qa_chain.invoke({"question": prompt})
                answer = result["answer"]
                st.markdown(answer)

                # ---- Sources ----
                if sources := result.get("source_documents"):
                    with st.expander(f"Sources ({len(sources)})"):
                        for i, doc in enumerate(sources[:3], 1):
                            url = doc.metadata["source"]
                            st.caption(f"**Source {i}**: [{url}]({url})")

        st.session_state.web_messages.append({"role": "assistant", "content": answer})