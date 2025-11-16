import streamlit as st

st.set_page_config(
    page_title="Smart Data Insights Assistant",
    page_icon='💬',
    layout='wide'
)

st.header("Smart Data Insights Assistant")
st.write("""
[![view source code ](https://img.shields.io/badge/GitHub%20Repository-gray?logo=github)](https://github.com/amulyaa-mohan/eda_langchain_chatbot)
""")  # Add your LinkedIn or other badges if desired

st.write("""
Welcome to the Smart Data Insights Assistant! This app leverages LangChain, CrewAI, and other AI tools to provide insights from structured data, documents, and the web.

Key Features:
- **Structured Query Bot**: Analyze CSVs with SQL queries and visualizations.
- **Document Intelligence Bot**: Query PDFs using Retrieval-Augmented Generation (RAG).
- **Web Intelligence Bot**: Scrape and query website content.

Select a bot from the sidebar to get started. For more details, check the GitHub repo above.
""")


