import sys, os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import tempfile
from pathlib import Path
from sqlalchemy import create_engine, text

from crewai import Task, Crew, Process
from agents.sql_agent import create_sql_agent
from agents.viz_agent import create_viz_agent
from db.connector import get_db
from utils.helpers import extract_sql, auto_chart


prompt_path = Path(os.path.dirname(__file__), '..', 'prompts', 'sql_prompt.txt')
PROMPT_TPL = prompt_path.read_text()

def patch_sql_for_sqlite(sql: str) -> str:
    """Fix MySQL functions when running on SQLite."""
    if not sql:
        return sql
    sql = sql.replace("YEAR(", "CAST(STRFTIME('%Y', ")
    sql = sql.replace(") AS sales_year", ") AS INTEGER) AS sales_year")
    return sql


def safe_get_task_output_text(task):
    for attr in ["raw_output", "text", "result"]:
        val = getattr(task.output, attr, None)
        if isinstance(val, str) and val.strip():
            return val
    if task.output:
        return str(task.output)
    return ""


def safe_extract_sql(text):
    if not text:
        return None
    try:
        sql = extract_sql(text)
        if sql:
            return sql
    except:
        pass
    low = text.lower()
    if "select " in low:
        return text[low.find("select "):]
    return None

def run():
    st.title("Structured Query Bot 🤖")

    if "sql_messages" not in st.session_state:
        st.session_state.sql_messages = []

    messages = st.session_state.sql_messages

    st.sidebar.header("Dataset Mode")

    dataset_mode = st.sidebar.radio(
        "Choose dataset mode",
        ["Local Olist DB", "Upload CSVs (SQLite)"],
        index=0
    )

    engine = None
    schema_for_prompt = "Schema unavailable."

    if dataset_mode == "Local Olist DB":
        try:
            db = get_db()
            engine = getattr(db, "_engine", None)
            schema_for_prompt = db.get_table_info()[:2000]

            st.sidebar.success("Connected to Local MySQL Olist database.")
            with st.sidebar.expander("Olist Schema"):
                st.code(schema_for_prompt)

        except Exception as e:
            st.sidebar.error(f"MySQL Connection Error: {e}")


    else:
        uploaded_files = st.sidebar.file_uploader(
            "Upload multiple CSV tables",
            type=["csv"],
            accept_multiple_files=True
        )

        if uploaded_files:
            temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
            engine = create_engine(f"sqlite:///{temp_db.name}")

            schema_text = ""
            for f in uploaded_files:
                df = pd.read_csv(f)
                tname = os.path.splitext(f.name)[0].lower().replace(" ", "_")
                df.to_sql(tname, engine, if_exists="replace", index=False)

                schema_text += f"\nTable `{tname}` → {list(df.columns)}"

                st.sidebar.success(f"Loaded `{tname}` ({df.shape[0]} rows)")

            schema_for_prompt = schema_text or "Schema unavailable."

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])


    if user_q := st.chat_input("Ask about the data (e.g. 'top sales year')"):

        st.chat_message("user").markdown(user_q)
        messages.append({"role": "user", "content": user_q})

        with st.chat_message("assistant"):
            with st.spinner("Analyzing…"):


                sql_agent = create_sql_agent()
                viz_agent = create_viz_agent()

                sql_task = Task(
                    description=PROMPT_TPL.format(
                        schema=schema_for_prompt,
                        question=user_q,
                    ),
                    agent=sql_agent,
                    expected_output="SQL query only"
                )

                viz_task = Task(
                    description="Write python Plotly code to visualize the query output.",
                    agent=viz_agent,
                    expected_output="python code"
                )

                crew = Crew(
                    agents=[sql_agent, viz_agent],
                    tasks=[sql_task, viz_task],
                    process=Process.sequential
                )

                crew.kickoff()

                sql_text = safe_get_task_output_text(sql_task)
                viz_text = safe_get_task_output_text(viz_task)

        sql_clean = safe_extract_sql(sql_text)

        if not sql_clean:
            st.error("❌ Could not extract SQL")
            st.code(sql_text)
            return

        if dataset_mode == "Upload CSVs (SQLite)":
            sql_clean = patch_sql_for_sqlite(sql_clean)

        st.subheader("Generated SQL")
        st.code(sql_clean, language="sql")

        if not engine:
            st.error("No database engine found.")
            return

        try:
            df = pd.read_sql(text(sql_clean), engine)
            st.subheader("Query Results")
            st.dataframe(df.head(50))

            st.subheader("Visualization")
            st.plotly_chart(auto_chart(df), use_container_width=True)

            st.subheader("Visualization Agent Code")
            st.code(viz_text, language="python")

            messages.append({"role": "assistant", "content": "Query executed successfully."})

        except Exception as e:
            st.error(f"SQL Execution Error: {e}")
            return


if __name__ == "__main__":
    run()
