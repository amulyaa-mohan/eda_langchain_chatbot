# Updated src/app.py — Add Web Intelligence Bot to sidebar
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.set_page_config(page_title="Smart Data Insights", layout="wide")
st.title("Smart Data Insights Assistant")
st.caption("AI-Powered Analytics for Databases & Documents")

page = st.sidebar.selectbox("Choose Tool", 
    ("Structured Data Explorer", "Document Intelligence Bot", "Web Intelligence Bot"))

if page == "Structured Data Explorer":
    import pages.sql_chat as sql_chat
    sql_chat.run()
elif page == "Document Intelligence Bot":
    import pages.rag_chat as rag_chat
    rag_chat.run()
elif page == "Web Intelligence Bot":
    import pages.web_chat as web_chat
    web_chat.run()
