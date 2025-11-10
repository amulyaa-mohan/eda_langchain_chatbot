# pages/web_chat.py — NEW: Chat with Website (using BeautifulSoup for scraping)
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from src.config.settings import Settings
from bs4 import BeautifulSoup
import requests
import torch  # For GPU if available
from pathlib import Path


# ------------------------------------------------------------------
# 1. CACHED EMBEDDINGS (runs only once)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI brain… (first time only)")
def get_embeddings():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": device}
        )
    except:
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embeddings()

# ------------------------------------------------------------------
# 2. LLM
# ------------------------------------------------------------------
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # Supported in 2025
        google_api_key=Settings().GOOGLE_API_KEY,
        temperature=0
    )


try:
    PROMPT_TPL = Path("src/prompts/web_prompt.txt").read_text(encoding="utf-8")
except FileNotFoundError:
    st.error("Create `src/prompts/web_prompt.txt`")
    st.stop()

# ------------------------------------------------------------------
# 4. MAIN APP
# ------------------------------------------------------------------
def run():
    st.title("Web Intelligence Bot")
    st.caption("Enter a URL → Ask anything about the website")

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "url" not in st.session_state:
        st.session_state.url = None

    url = st.text_input("Enter Website URL", placeholder="https://example.com")

    # ------------------------------------------------------------------
    # SCRAPE & INDEX WEBSITE (general, fast)
    # ------------------------------------------------------------------
    if url and url != st.session_state.url:
        with st.spinner("Scraping website… (less than 5 seconds)"):
            try:
                # Fetch content with BeautifulSoup
                response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                # Extract clean text (remove scripts, styles, etc.)
                for script in soup(["script", "style"]):
                    script.extract()
                text = soup.get_text(separator="\n").strip()

                # Create document with metadata
                docs = [Document(page_content=text, metadata={"source": url, "page": "Full Page"})]

                # Chunking – 800 chars → fast
                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                chunks = splitter.split_documents(docs)

                # Optional: Add dynamic summary from title/meta
                title = soup.title.string if soup.title else "Website"
                summary = Document(
                    page_content=f"Website title: {title}. Summary: {text[:500].replace('\n', ' ')}",
                    metadata={"page": "Summary"}
                )
                chunks = [summary] + chunks

                # Build FAISS
                st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
                st.session_state.url = url
                st.session_state.messages = []

                st.success("Ready! Ask anything about the website.")

            except Exception as e:
                st.error(f"Error scraping URL: {e}")

    # ------------------------------------------------------------------
    # QA CHAIN
    # ------------------------------------------------------------------
    if st.session_state.vectorstore:
        qa = RetrievalQA.from_chain_type(
            llm=get_llm(),
            chain_type="stuff",
            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PromptTemplate.from_template(PROMPT_TPL)},
        )

        # Chat
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about the website…"):
            # ----- GREETING -----
            if prompt.strip().lower() in ["hello", "hi", "hey"]:
                reply = "Hello! How can I help you with the website?"
            else:
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking…"):
                        result = qa.invoke({"query": prompt})
                        reply = result["result"]
                        st.write(reply)

                        # ----- SHOW SOURCES -----
                        if sources := result.get("source_documents"):
                            with st.expander(f"Sources ({len(sources)})"):
                                for doc in sources[:3]:
                                    page = doc.metadata.get("page", "Summary")
                                    snippet = doc.page_content.replace("\n", " ")[:200]
                                    st.caption(f"**Page {page}**: {snippet}…")

            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": reply})
