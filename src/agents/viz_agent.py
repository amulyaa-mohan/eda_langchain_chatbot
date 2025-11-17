from crewai import Agent
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config.settings import Settings

settings = Settings()

def create_viz_agent() -> Agent:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0,
        google_api_version="v1",
        streaming=False,                       
        convert_system_message_to_human=True,  
    )

    return Agent(
        role="Data Visualization Specialist",
        goal="Pick the best Plotly chart and output the exact Python code",
        backstory="Bar for rankings, line for trends, pie for proportions.",
        llm=llm,                               
        verbose=True,
        allow_delegation=False,
    )
