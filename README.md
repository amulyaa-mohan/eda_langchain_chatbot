<h1><b>Exploratory Data Analysis Chatbot 🤖</b></h1> 
A multi-tool AI chatbot for exploratory data analysis (EDA) using LangChain and CrewAI. <br> This mini-project allows users to interact with structured data, documents, and web content through natural language queries, generating SQL queries, visualizations, and insights.
<hr>
<h3><b>Description</b></h3>
This project is an interactive chatbot system for EDA tasks. It includes three specialized bots:
<ol>
<li><b>Document Intelligence Bot:</b> Analyzes uploaded documents (PDFs, CSVs) using Retrieval-Augmented Generation (RAG).</li>
<li><b>Structured Query Bot:</b> Queries structured databases (e.g., Olist e-commerce dataset or uploaded CSV) with SQL and generates Plotly visualizations.</li>
<li><b>Web Intelligence Bot:</b> Scrapes websites and answers questions using RAG on the content.</li>
<li>Built as a Streamlit multi-page app for easy deployment and use.</li>
</ol>
<hr>
<h3><b>Tech Stack</b></h3>
<ol>
<li><b>Frontend/UI:</b> Streamlit</li>
<li><b>AI Framework:</b> LangChain (for RAG, agents, embeddings)</li>
<li><b>Agent Orchestration:</b> CrewAI (for SQL and visualization agents)</li>
<li><b>LLM:</b> Google Gemini (via langchain-google-genai)</li>
<li><b>Embeddings:</b> HuggingFace (all-MiniLM-L6-v2 model) </li>
<li><b>Database:</b> SQLite (for in-memory/temp DBs), MySQL (for local datasets)</li>
<li><b>Visualization:</b> Plotly</li>
<li><b>Scraping:</b> Requests, BeautifulSoup</li>
<li><b>Other:</b> Pandas (data handling), SQLAlchemy (DB connections)</li>
</ol>
<hr>
<h3><b>Features</b></h3>
<ol>
<li><b>Multi-Page Streamlit Interface:</b> Easy navigation between home and bots.</li>
<li><b>Local Dataset Support:</b> Pre-loaded Olist e-commerce dataset for structured queries.</li>
<li><b>CSV Upload:</b> Upload your own CSV for custom structured queries.</li>
<li><b>Natural Language Queries:</b> Ask questions in plain English – get SQL, data tables, and visualizations.</li>
<li><b>Web Scraping:</b> Add URLs, scrape content, ask questions with sources.</li>
<li><b>Document Analysis:</b> Upload PDFs/CSVs, query content with RAG.</li>
<li><b>Error Handling:</b> Graceful handling of invalid URLs, DB errors, and no-content scenarios.</li>
<li><b>Debug Views:</b> Schema expanders and status messages for transparency.</li>
<h3>Screenshots</h3><hr>

*Home page with navigation.*

![Home Page](https://raw.githubusercontent.com/amulyaa-Mohan/exploratory_data_analysis_chatbot/main/Screenshots/home.png)

*Structured Query Bot with Olist dataset and visualization.*

![Structured Query Bot](https://raw.githubusercontent.com/amulyaa-Mohan/exploratory_data_analysis_chatbot/main/Screenshots/structured_query_bot.png)

*Document Intelligence Bot answering from uploaded PDF.*

![Document Intelligence Bot](https://raw.githubusercontent.com/amulyaa-Mohan/exploratory_data_analysis_chatbot/main/Screenshots/document_intelligence_bot.png)

*Web Intelligence Bot querying SEC filing.*

![Web Intelligence Bot](https://raw.githubusercontent.com/amulyaa-Mohan/exploratory_data_analysis_chatbot/main/Screenshots/web_intelligence_bot.png)

### Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/amulyaa-Mohan/exploratory_data_analysis_chatbot.git
   cd exploratory_data_analysis_chatbot

2. **Create & activate virtual environment**
    ```bash
   python -m venv venv
    Windows
    venv\Scripts\activate
    macOS / Linux
    source venv/bin/activate
3. Install dependencies
   ```bash
    pip install -r requirements.txt
5. Set your Google API Key
6. Create a .env file in the project root
7. (Optional) Load Olist Dataset
   Download from Kaggle Olist Dataset
   Extract all CSV files into the data/ folder

8. **Run the app**
    ```bash
    streamlit run src/Home.py

License
MIT License — see LICENSE for details.

**Thank you for using the Exploratory Data Analysis Chatbot!<br>
By @amulyaa-Mohan**
