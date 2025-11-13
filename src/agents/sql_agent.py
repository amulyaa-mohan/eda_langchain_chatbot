from crewai import Agent
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config.settings import Settings

settings = Settings()
_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  
    google_api_key=settings.GOOGLE_API_KEY,
    temperature=0,
    google_api_version="v1"  
)

def create_sql_agent() -> Agent:
    return Agent(
        role="Senior MySQL Data Engineer",
        goal="Generate safe, correct SQL from natural language",
        backstory="You know the Olist schema perfectly.",
        llm=_llm,
        verbose=True,
        allow_delegation=False,
    )