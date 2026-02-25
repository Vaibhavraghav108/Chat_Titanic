import streamlit as st
import requests

BACKEND = "http://localhost:8000"

st.set_page_config(page_title="Titanic Dataset Explorer", page_icon="🚢", layout="wide")

st.title("Titanic Dataset Explorer")
st.markdown("Ask questions about the Titanic passengers - get answers and visualizations!")

# sidebar
with st.sidebar:
    st.header("Example Questions")
    examples = [
        "What percentage of passengers were male?",
        "Show a histogram of passenger ages",
        "What was the average ticket fare?",
        "How many passengers embarked from each port?",
        "What was the survival rate by passenger class?",
        "Show a bar chart of survival by gender",
    ]
    for q in examples:
        if st.button(q, key=q, use_container_width=True):
            st.session_state["pending_question"] = q

# chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(msg["question"])
    with st.chat_message("assistant"):
        st.write(msg["answer"])
        if msg.get("plot_url"):
            st.image(msg["plot_url"])

# user input
question = st.chat_input("Ask something about the Titanic dataset...")

if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{BACKEND}/chat",
                    json={"question": question},
                    timeout=120,
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer received.")
                    plot_path = data.get("plot_path")

                    st.write(answer)

                    plot_url = None
                    if plot_path:
                        plot_url = f"{BACKEND}{plot_path}"
                        st.image(plot_url)

                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer,
                        "plot_url": plot_url,
                    })
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

            except requests.exceptions.ConnectionError:
                st.error("Can't connect to backend. Make sure the FastAPI server is running!")
            except requests.exceptions.Timeout:
                st.error("Request timed out. Try a simpler question.")
            except Exception as e:
                st.error(f"Something went wrong: {e}")
