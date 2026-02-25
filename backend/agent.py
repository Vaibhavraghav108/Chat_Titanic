import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from functools import lru_cache
from tools import query_data, create_visualization
from data_loader import get_schema_info
from dotenv import load_dotenv
load_dotenv()


def is_valid_key(key):
    """Check if an API key is actually set (not empty or placeholder)."""
    if not key:
        return False
    placeholders = ["your-", "put-your", "enter-your", "sk-xxx", "placeholder"]
    return not any(key.lower().startswith(p) for p in placeholders)


def get_llm():
    # Try OpenAI first
    openai_key = os.getenv("OPENAI_API_KEY")
    if is_valid_key(openai_key):
        from langchain_openai import ChatOpenAI
        print("Using OpenAI as LLM provider")
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            api_key=openai_key,
        )

    # Try Google Gemini next
    google_key = os.getenv("GOOGLE_API_KEY")
    if is_valid_key(google_key):
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = os.getenv("GOOGLE_MODEL", "gemini-3-flash-preview")
        print(f"Using Google Gemini ({model}) as LLM provider")
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            google_api_key=google_key,
        )
    
    # Try HuggingFace as last option
    hf_key = os.getenv("HUGGINGFACE_API_KEY")
    if is_valid_key(hf_key):
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        repo_id = os.getenv("HUGGINGFACE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
        print(f"Using HuggingFace ({repo_id}) as LLM provider")
        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            huggingfacehub_api_token=hf_key,
            temperature=0.1,
            max_new_tokens=1024,
        )
        return ChatHuggingFace(llm=llm)

    raise RuntimeError(
        "No API key found! Please set OPENAI_API_KEY, GOOGLE_API_KEY, "
        "or HUGGINGFACE_API_KEY in your .env file."
    )


def build_prompt():

    schema = get_schema_info()
    template = """You are a data analyst assistant that answers questions about the Titanic dataset.

{schema}

You have access to these tools:
{tools}

Tool names: {tool_names}

Rules:
1. For data/statistics questions -> use query_data tool
2. For charts/plots/visualizations -> use create_visualization tool
3. If user wants both stats and a chart, do query_data first then create_visualization
4. Always give a clear Final Answer with numbers rounded to 2 decimal places
5. Use the exact column names from the schema above

You must follow this format:

Question: the user question
Thought: think about what to do
Action: one of [{tool_names}]
Action Input: python code to run
Observation: tool output
... (repeat if needed)
Thought: I now know the final answer
Final Answer: your answer to the user

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

    return PromptTemplate(
        input_variables=["input", "agent_scratchpad"],
        partial_variables={"schema": schema},
        template=template,
    )


def setup_agent():
    llm = get_llm()
    tools = [query_data, create_visualization]
    prompt = build_prompt()

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
        return_intermediate_steps=True,
    )
    print("Agent ready!")
    return executor


@lru_cache(maxsize=1)
def get_agent():
    """Returns the agent, creating it on first call (cached via lru_cache)."""
    return setup_agent()
