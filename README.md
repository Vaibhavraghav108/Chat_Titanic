# Titanic Dataset Chat Agent

A chatbot that answers questions about the Titanic dataset using AI. Built with FastAPI, LangChain, and Streamlit.

Ask statistical questions ("What percentage were male?") or request visualizations ("Show a histogram of ages") — it analyzes the data and responds.

## Tech Stack

- **Backend:** FastAPI + LangChain (ReAct agent)
- **Frontend:** Streamlit
- **LLM Support:** OpenAI, Google Gemini, or HuggingFace
- **Data Analysis:** Pandas, Matplotlib, Seaborn

## Project Structure

```
├── backend/
│   ├── main.py            # FastAPI server
│   ├── agent.py           # LangChain agent setup
│   ├── tools.py           # query_data and create_visualization tools
│   ├── data_loader.py     # Loads and cleans titanic.csv
│   └── static/plots/      # Generated plots saved here
├── frontend/
│   └── app.py             # Streamlit chat UI
├── data/
│   └── titanic.csv
├── .env                   # Your API keys
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

### 1. Create virtual environment

```bash
python -m venv .myenv

# Windows
.myenv\Scripts\activate

# Mac/Linux
source .myenv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up API key

Copy `.env.example` to `.env` and add at least one API key:

```bash
copy .env.example .env
```

The app picks the first valid key it finds:
1. OpenAI
2. Google Gemini
3. HuggingFace

### 4. Run the backend

```bash
cd backend
uvicorn main:app --reload
```

Runs at http://localhost:8000

### 5. Run the frontend (new terminal)

```bash
cd frontend
streamlit run app.py
```

Opens at http://localhost:8501

## Example Questions

- What percentage of passengers were male?
- Show a histogram of passenger ages
- What was the average ticket fare?
- How many passengers embarked from each port?
- What was the survival rate by passenger class?
- Show a bar chart of survival by gender
