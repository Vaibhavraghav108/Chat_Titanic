import os
import random
import string
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from langchain.tools import tool
from data_loader import load_titanic_data

# folder where plots get saved
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def random_filename():
    chars = random.choices(string.ascii_lowercase + string.digits, k=8)
    return "plot_" + "".join(chars) + ".png"

@tool
def query_data(python_code: str) -> str:
    """Run pandas code on the Titanic dataframe. Use variable `result` to store output."""
    df = load_titanic_data()
    local_vars = {}

    try:
        exec(python_code, {"df": df, "pd": pd}, local_vars)

        if "result" in local_vars:
            return str(local_vars["result"])
        return "Code ran but no output. Put your answer in `result`."

    except Exception as e:
        return f"Error: {e}"


@tool
def create_visualization(python_code: str) -> str:
    """Create a matplotlib/seaborn chart. Do NOT call plt.show()."""
    df = load_titanic_data()
    plt.close("all")

    try:
        exec(python_code, {"df": df, "pd": pd, "plt": plt, "sns": sns}, {})

        filename = random_filename()
        filepath = os.path.join(PLOTS_DIR, filename)
        plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close("all")

        return f"Visualization saved successfully: {filename}"

    except Exception as e:
        plt.close("all")
        return f"Error creating plot: {e}"
