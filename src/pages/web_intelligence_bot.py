
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import requests
from bs4 import BeautifulSoup
import validators
import streamlit as st
from typing import List
import time
import random

from langchain_google_genai import GoogleGenerativeAI


from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.chains import RetrievalQA

from src.config.settings import Settings

def get_llm():
    return GoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=Settings().GOOGLE_API_KEY,
        temperature=0,
        convert_system_message_to_human=True
    )

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def scrape_website(url: str) -> str:
    if not url:
        return ""
    url = url.strip()

    if "sec.gov" in url.lower() and (url.endswith(".htm") or url.endswith(".html")):
        try:
            base = url.rsplit('/', 1)[0]
            filename = url.rsplit('/', 1)[1]
            stem = filename.split('.')[0]
            txt_url = f"{base}/{stem}.txt"
            r = requests.get(txt_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200 and len(r.text) > 200:
                return r.text
        except Exception:
            pass

    try:
        clean = url.replace("https://", "").replace("http://", "")
        jina_url = f"https://r.jina.ai/https://{clean}"
        r = requests.get(jina_url, timeout=30)
        if r.status_code == 200 and len(r.text.strip()) > 50:
            return r.text
    except Exception:
        pass

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }

    for _ in range(3):
        try:
            time.sleep(random.uniform(0.8, 1.6))
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
                    tag.decompose()
                text = soup.get_text(" ", strip=True)
                if len(text) > 50:
                    return text
        except Exception:
            pass

    return ""

@st.cache_resource(show_spinner="Indexing documents...", ttl=3600)
def setup_vectordb(websites: List[str]):
    docs = []
    for url in websites:
        try:
            text = scrape_website(url)
            if not text:
                st.warning(f"No text extracted from {url}")
                continue
            docs.append(Document(page_content=text, metadata={"source": url}))
        except Exception as e:
            st.warning(f"Failed to process {url}: {e}")

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    try:
        vectordb = DocArrayInMemorySearch.from_documents(chunks, get_embeddings())
    except TypeError:
        vectordb = DocArrayInMemorySearch.from_documents(chunks, embedding=get_embeddings())

    return vectordb

def extract_docs_from_vectordb(vectordb) -> List[Document]:
    if not vectordb:
        return []

    options = [
        "_documents",
        "docs",
        "_data",
        "_docstore",
    ]

    for attr in options:
        if hasattr(vectordb, attr):
            try:
                val = getattr(vectordb, attr)
                if isinstance(val, list):
                    return val
            except:
                pass

    try:
        retr = vectordb.as_retriever()
        sample = retr.get_relevant_documents("the")
        if sample:
            return sample
    except:
        pass

    return []

def run():
    st.set_page_config(page_title="Web Intelligence Bot", layout="wide")
    st.title("🔎 Web Intelligence Bot — SEC-grade RAG (Option 2)")
    st.caption("Add URLs → Index → Ask questions")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []   

    with st.sidebar:
        st.header("Index URLs")
        url_input = st.text_input(
            "URL to index",
            value="https://www.sec.gov/Archives/edgar/data/1318605/000162828024002390/tsla-20231231.htm"
        )

        if st.button("Add & Index"):
            if not validators.url(url_input):
                st.error("Invalid URL")
            else:
                st.session_state.setdefault("urls", [])
                if url_input not in st.session_state["urls"]:
                    st.session_state["urls"].append(url_input)
                    st.success("Added. Rebuilding index...")
                    st.session_state.pop("vectordb", None)
                else:
                    st.info("URL already added")

        if st.button("Clear URLs"):
            st.session_state["urls"] = []
            st.session_state.pop("vectordb", None)

        if st.session_state.get("urls"):
            st.markdown("**Indexed URLs:**")
            for u in st.session_state["urls"]:
                st.write(f"- {u}")

    urls = st.session_state.get("urls", [])
    if not urls:
        st.info("Add URLs to begin.")
        return

    vectordb = setup_vectordb(urls)
    if not vectordb:
        st.error("Indexing returned no documents.")
        return

    docs_list = extract_docs_from_vectordb(vectordb)
    st.success(f"Indexed {len(docs_list)} document chunks successfully!")

    with st.expander("Debug — First chunk"):
        if docs_list:
            st.code(docs_list[0].page_content[:1600])
        else:
            st.write("No docs")


    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    qa = RetrievalQA.from_chain_type(
        llm=get_llm(),
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    st.subheader("💬 Chat History")
    if st.session_state.chat_history:
        for item in st.session_state.chat_history:
            st.markdown(f"**You:** {item['q']}")
            st.markdown(f"**Bot:** {item['a']}")
            st.markdown("---")
    else:
        st.caption("No chat yet.")

    st.subheader("Ask a factual question")
    query = st.text_input("Your question")

    if query:
        with st.spinner("Thinking..."):
            try:
                result = qa({"query": query})
            except:
                try:
                    result = qa.run(query)
                except:
                    result = qa.invoke({"query": query})

        answer = ""
        sources = []

        if isinstance(result, dict):
            answer = result.get("result") or result.get("answer") or ""
            sources = result.get("source_documents", [])
        elif isinstance(result, str):
            answer = result

       
        if not answer:
            answer = "No answer found."

        st.session_state.chat_history.append({"q": query, "a": answer})

        st.markdown("### ✅ Answer")
        st.write(answer)

        if sources:
            with st.expander(f"Sources ({len(sources)})"):
                for i, d in enumerate(sources[:6], start=1):
                    src = getattr(d, "metadata", {}).get("source", "Unknown")
                    st.caption(f"{i}. {src}")


if __name__ == "__main__":
    run()
