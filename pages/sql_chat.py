# pages/sql_chat.py
import streamlit as st
import pandas as pd
from crewai import Task, Crew, Process
from agents.sql_agent import create_sql_agent
from agents.viz_agent import create_viz_agent
from db.connector import get_db
from utils.helpers import extract_sql, auto_chart
from pathlib import Path

# Fixed prompt path
PROMPT_TPL = Path("src/prompts/sql_prompt.txt").read_text()

def run():
    # Unique session state for SQL page
    session_key = "sql_messages"
    if session_key not in st.session_state:
        st.session_state[session_key] = []

    messages = st.session_state[session_key]
    db = get_db()

    with st.expander("Database Schema", expanded=False):
        st.code(db.get_table_info()[:2000], language="sql")

    # Display chat history
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input with unique key
    if prompt := st.chat_input(
        "Ask about sales, customers, trends…",
        key="sql_chat_input"
    ):
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Running SQL → Viz agents…"):
                sql_agent = create_sql_agent()
                viz_agent = create_viz_agent()

                sql_task = Task(
                    description=PROMPT_TPL.format(
                        schema=db.get_table_info()[:1500],
                        question=prompt,
                    ),
                    agent=sql_agent,
                    expected_output="SQL query only",
                )

                viz_task = Task(
                    description="Create Plotly chart code from the previous SQL result.",
                    agent=viz_agent,
                    expected_output="Python code that builds a Plotly figure.",
                )

                crew = Crew(
                    agents=[sql_agent, viz_agent],
                    tasks=[sql_task, viz_task],
                    process=Process.sequential,
                    verbose=False,
                )
                crew.kickoff()

                sql_result = sql_task.output.raw_output
                viz_result = viz_task.output.raw_output

            
            raw_sql = extract_sql(sql_result)
            if raw_sql:
                raw_sql = raw_sql.replace('%', '%%') 
            else:
                st.error("Failed to extract SQL query from agent response.")
                st.code(sql_result, language="text")
                return

            st.subheader("Generated SQL")
            st.code(raw_sql, language="sql")

            try:
                df = pd.read_sql(raw_sql, db._engine)
                st.subheader("Result")
                st.dataframe(df.head(20))

                st.subheader("Visualization")
                fig = auto_chart(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No chart generated.")

                st.subheader("Viz Agent Code")
                st.code(viz_result, language="python")

            except Exception as e:
                st.error(f"SQL Execution Error: {e}")
                st.code(raw_sql, language="sql")