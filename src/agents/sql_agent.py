# src/agents/sql_agent.py

from crewai import Agent
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config.settings import Settings

settings = Settings()

_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=settings.GOOGLE_API_KEY,
    temperature=0,
    google_api_version="v1",
    streaming=False,
    convert_system_message_to_human=True,
)

def create_sql_agent() -> Agent:
    return Agent(
        role="Senior SQL Data Engineer",
        goal="Generate safe, correct SQL from natural language.",
        backstory="Expert in SQL, schemas, and query optimization.",
        llm=_llm,
        verbose=True,
        allow_delegation=False,
    )
