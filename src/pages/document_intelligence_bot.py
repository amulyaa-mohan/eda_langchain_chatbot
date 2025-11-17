import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], 'classes')]

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from src.config.settings import Settings
from pathlib import Path
import re

@st.cache_resource(show_spinner="Loading AI brain… (first time only)")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embeddings()

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=Settings().GOOGLE_API_KEY,
        temperature=0
    )

PROMPT_TPL = """
You are a precise document analyst. Use **all** the supplied context to answer the question.
If a fact is present in any chunk (even partially), include it. Only say "I don't know" if *nothing* in the context relates.

--- ETHICAL GUARDRAIL ---
If the question is about harming anyone, illegal activity, or anything unethical, respond **exactly** with:
"I cannot help with that. Promoting violence or illegal actions is against my guidelines."

--- INSTRUCTIONS ---
1. Extract numbers, dates, quotes, checkbox states (☐ = No, ☒ = Yes), company names, etc.
2. Keep the answer **1–2 short paragraphs** (max 4 sentences). Be factual.
3. End with a citation:  
   Source: "exact quote…" (Page X)   or   (Summary)

Context:
{context}

Question: {question}

Answer:
"""
PROMPT = PromptTemplate(template=PROMPT_TPL, input_variables=["context", "question"])

def run():
    st.title("Document Intelligence Bot")
    st.caption("Ask anything from PDFs — with sources")

    debug = st.sidebar.checkbox("Debug Mode")

    with st.sidebar:
        st.header("Document Source")
        source = st.radio("Choose", ["Upload Document", "Use Sample 10-K"])

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "file_name" not in st.session_state:
        st.session_state.file_name = None

    if source == "Upload Document":
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded_file and uploaded_file.name != st.session_state.file_name:
            with st.spinner("Indexing document…"):
                try:
                    loader = PyPDFLoader(uploaded_file)
                    docs = loader.load()
                    
                    
                    summary_facts = []
                    for i in range(min(5, len(docs))):
                        page = docs[i].page_content
                        m = re.search(r'(FORM\s+10-K|ANNUAL\s+REPORT).*?', page, re.I)
                        if m: summary_facts.append(m.group().strip())
                        nums = re.findall(r'\$[\d,.,]+|\d{4}\s+shares?', page)
                        summary_facts.extend(nums[:3])
                        checks = re.findall(r'(☐|☒)\s+(Yes|No)', page)
                        summary_facts.extend([f"{c[0]} {c[1]}" for c in checks[:3]])
                    
                    summary_text = "Key facts: " + " | ".join(set(summary_facts[:8]))
                    summary_doc = Document(page_content=summary_text, metadata={"page": "Summary"})
                    
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = [summary_doc] + splitter.split_documents(docs)
                    
                    st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
                    st.session_state.file_name = uploaded_file.name
                    st.session_state.messages = []
                    st.success(f"Loaded: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error: {e}")

    else:
        sample_path = Path(__file__).parent.parent.parent / "data" / "nextnav_10k_2022.pdf"
        
        if not sample_path.exists():
            st.error("Sample PDF not found! Place 'nextnav_10k_2022.pdf' in data/ folder")
            st.stop()
        
        if st.session_state.file_name != "nextnav_10k_2022.pdf":
            with st.spinner("Loading NextNav Inc. 10-K (2022)…"):
                try:
                    loader = PyPDFLoader(str(sample_path))
                    docs = loader.load()
                    
                    summary_facts = []
                    for i in range(min(5, len(docs))):
                        page = docs[i].page_content
                        m = re.search(r'(FORM\s+10-K|ANNUAL\s+REPORT).*?', page, re.I)
                        if m: summary_facts.append(m.group().strip())
                        nums = re.findall(r'\$[\d,.,]+|\d{4}\s+shares?', page)
                        summary_facts.extend(nums[:3])
                        checks = re.findall(r'(☐|☒)\s+(Yes|No)', page)
                        summary_facts.extend([f"{c[0]} {c[1]}" for c in checks[:3]])
                    
                    summary_text = "Key facts: " + " | ".join(set(summary_facts[:8]))
                    summary_doc = Document(page_content=summary_text, metadata={"page": "Summary"})
                    
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = [summary_doc] + splitter.split_documents(docs)
                    
                    st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
                    st.session_state.file_name = "nextnav_10k_2022.pdf"
                    st.session_state.messages = []
                    st.success("NextNav Inc. 10-K (2022) loaded!")
                except Exception as e:
                    st.error(f"Error loading sample: {e}")

    if st.session_state.vectorstore:
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 12, "fetch_k": 30, "lambda_mult": 0.5}
        )
        qa = RetrievalQA.from_chain_type(
            llm=get_llm(),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about the document…"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        result = qa.invoke({"query": prompt})
                        reply = result["result"]
                        st.markdown(reply)

                        sources = result.get("source_documents", [])
                        if sources:
                            with st.expander(f"Sources ({len(sources)})"):
                                for doc in sources[:3]:
                                    page = doc.metadata.get("page", "Summary")
                                    snippet = doc.page_content.replace("\n", " ")[:200]
                                    st.caption(f"**Page {page}**: {snippet}…")

                        if debug:
                            scored = st.session_state.vectorstore.similarity_search_with_score(prompt, k=12)
                            with st.expander("Debug: Top chunks"):
                                for i, (doc, score) in enumerate(scored[:5]):
                                    page = doc.metadata.get("page", "Summary")
                                    st.caption(f"Chunk {i+1} | Page {page} | Score: {score:.3f}")

                    except Exception as e:
                        st.error(f"Error: {e}")

            st.session_state.messages.append({"role": "assistant", "content": reply})

run()