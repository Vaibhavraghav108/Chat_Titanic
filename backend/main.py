
import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import get_agent

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(title="Titanic Chat Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    plot_path: str | None = None


@app.on_event("startup")
def startup():
    get_agent()
    print("Server ready!")


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        agent = get_agent()
        result = agent.invoke({"input": question})

        answer = result.get("output", "Sorry, I couldn't generate an answer.")
        plot_path = None

        for step in result.get("intermediate_steps", []):
            action, observation = step
            if action.tool == "create_visualization" and "saved successfully" in str(observation):
                match = re.search(r"(plot_[a-z0-9]+\.png)", str(observation))
                if match:
                    plot_path = f"/static/plots/{match.group(1)}"

        return ChatResponse(answer=answer, plot_path=plot_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
