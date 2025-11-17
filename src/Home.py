
import streamlit as st

st.set_page_config(
    page_title="Smart Data Insights Assistant 🤖",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded"  
)


st.title("AI Smart Data Insights Assistant")
st.markdown("### Turn Data, Documents & the Web into Actionable Intelligence")

st.info("""
Welcome! This app uses **LangChain + CrewAI + Gemini** to answer your questions in plain English — from CSVs, PDFs, or live websites.
""")

st.markdown("### Available Tools")

col1, col2, col3 = st.columns(3)

with col1:
    st.success("Structured Query Bot")
    st.markdown("""
    - Use Olist dataset or upload CSV  
    - Ask: “Top 5 cities by revenue?”  
    - Get: SQL + Interactive Chart
    """)

with col2:
    st.success("Document Intelligence Bot")
    st.markdown("""
    - Upload PDFs (reports, books, papers)  
    - Ask: “What are the key risks?”  
    - Get: Answers with source pages
    """)

with col3:
    st.success("Web Intelligence Bot")
    st.markdown("""
    - Paste any website URL  
    - Ask: “Tesla revenue in 2023?”  
    - Get: Real-time scraped insights
    """)

st.markdown("### Try These Questions")
st.code("""
• Show monthly sales trend in 2018
• Summarize the CEO's message from this report
• What is Tesla's cash position from latest 10-K?
""", language="text")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    Made with <strong>Streamlit</strong>  
    Powered by Gemini • LangChain • CrewAI  
    <br>
    <a href="https://github.com/amulyaa-mohan/exploratory_data_analysis_chatbot" target="_blank">
        View Source Code
    </a>
</div>
""", unsafe_allow_html=True)