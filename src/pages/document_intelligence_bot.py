
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))  # Path fix

import torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], 'classes')]

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from src.config.settings import Settings
import tempfile
from pathlib import Path
import re   

st.write("Debug:loaded successfully.")  

@st.cache_resource(show_spinner="Loading AI brain… (first time only)")
def get_embeddings():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": device}
        )
    except Exception as e:
        st.error(f"Embeddings load error: {e}")
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  

embeddings = get_embeddings()

def get_llm():
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=Settings().GOOGLE_API_KEY,
            temperature=0
        )
    except Exception as e:
        st.error(f"LLM load error: {e}")
        return None

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
    st.caption("Ask anything – from the document")

    debug = st.sidebar.checkbox("Debug Mode (show chunks & scores)")


    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "file_name" not in st.session_state:
        st.session_state.file_name = None

    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "txt", "docx"])


    if uploaded_file and uploaded_file.name != st.session_state.file_name:
        with st.spinner("Reading & indexing…"):
            try:
                tmp_path = Path(tempfile.gettempdir()) / uploaded_file.name
                tmp_path.write_bytes(uploaded_file.getvalue())

                if uploaded_file.name.lower().endswith(".pdf"):
                    loader = PyPDFLoader(str(tmp_path))
                elif uploaded_file.name.lower().endswith(".txt"):
                    loader = TextLoader(str(tmp_path), encoding="utf-8")
                else:
                    loader = Docx2txtLoader(str(tmp_path))

                docs = loader.load()

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)

                summary_facts = []
                for i in range(min(3, len(docs))):
                    page = docs[i].page_content
                    m = re.search(r'(FORM\s+10-K|ANNUAL\s+REPORT).*?(NextNav|ACRES).*?', page, re.I)
                    if m:
                        summary_facts.append(m.group().strip())
                    nums = re.findall(r'\$[\d,]+\.?\d*|\d{4}\s+shares?', page)
                    summary_facts.extend(nums[:2])
                    checks = re.findall(r'(☐|☒)\s+(Yes|No)', page)
                    summary_facts.extend([f"{c[0]} {c[1]}" for c in checks[:2]])

                summary_text = "Key facts: " + " | ".join(set(summary_facts[:6]))
                summary = Document(page_content=summary_text, metadata={"page": "Enhanced Summary"})
                chunks = [summary] + chunks

                st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
                st.session_state.file_name = uploaded_file.name
                st.session_state.messages = []

                st.success("Ready! Strong retrieval active.")
            except Exception as e:
                st.error(f"Indexing error: {e}")
            finally:
                tmp_path.unlink(missing_ok=True)

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
            if prompt.strip().lower() in ["hello", "hi", "hey"]:
                reply = "Hello! How can I help you with the document?"
            else:
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("Searching…"):
                        try:
                            result = qa.invoke({"query": prompt})
                            reply = result["result"]
                            st.write(reply)
                        except Exception as e:
                            reply = f"Error: {str(e)}"
                            st.error(reply)

                        
                        sources = result.get("source_documents", [])
                        if debug and sources:
                            
                            scored = st.session_state.vectorstore.similarity_search_with_score(prompt, k=12)
                            with st.expander(f"Debug: Retrieved {len(sources)} chunks"):
                                for i, (doc, score) in enumerate(scored[:5]):
                                    page = doc.metadata.get("page", "Summary")
                                    snippet = doc.page_content.replace("\n", " ")[:150]
                                    st.caption(f"**Chunk {i+1} (Page {page}, Score: {score:.3f})**: {snippet}…")

                        if sources:
                            with st.expander(f"Sources ({len(sources)})"):
                                for doc in sources[:3]:
                                    page = doc.metadata.get("page", "Summary")
                                    snippet = doc.page_content.replace("\n", " ")[:200]
                                    st.caption(f"**Page {page}**: {snippet}…")

            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": reply})

run()

