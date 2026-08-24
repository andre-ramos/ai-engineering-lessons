# This tutorial guides you through creating an AI-powered data analysis agent
# that can interpret and answer questions about a dataset using natural language.
# It combines language models with data manipulation tools to enable intuitive
# data exploration using Pandas and NumPy.

from langchain_experimental.agents.agent_toolkits import (
    create_pandas_dataframe_agent,
)
from langchain_openai import ChatOpenAI

import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Set a random seed for reproducibility
np.random.seed(42)

# Generate sample data
n_rows = 1000

# Generate dates
start_date = datetime(2022, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(n_rows)]

# Define data categories
makes = [
    "Toyota",
    "Honda",
    "Ford",
    "Chevrolet",
    "Nissan",
    "BMW",
    "Mercedes",
    "Audi",
    "Hyundai",
    "Kia",
]

models = [
    "Sedan",
    "SUV",
    "Truck",
    "Hatchback",
    "Coupe",
    "Van",
]

colors = [
    "Red",
    "Blue",
    "Black",
    "White",
    "Silver",
    "Gray",
    "Green",
]

# Create the dataset
data = {
    "Date": dates,
    "Make": np.random.choice(makes, n_rows),
    "Model": np.random.choice(models, n_rows),
    "Color": np.random.choice(colors, n_rows),
    "Year": np.random.randint(2015, 2023, n_rows),
    "Price": np.random.uniform(20000, 80000, n_rows).round(2),
    "Mileage": np.random.uniform(0, 100000, n_rows).round(0),
    "EngineSize": np.random.choice(
        [1.6, 2.0, 2.5, 3.0, 3.5, 4.0],
        n_rows,
    ),
    "FuelEfficiency": np.random.uniform(20, 40, n_rows).round(1),
    "SalesPerson": np.random.choice(
        ["Alice", "Bob", "Charlie", "David", "Eva"],
        n_rows,
    ),
}

# Create DataFrame and sort by date
df = pd.DataFrame(data).sort_values("Date")

print("\nFirst few rows of the generated data:")
print(df.head())

print("\nDataFrame info:")
df.info()

print("\nSummary statistics:")
print(df.describe())


# Creating Data Analysis Agent

agent = create_pandas_dataframe_agent(ChatOpenAI(model="gpt-4o", temperature=0),
                                      df,
                                      verbose=True,
                                      allow_dangerous_code=True,
                                      agent_type = "tool-calling")

print("Data Analysis Agent is ready.")


def ask_agent(question):
    response = agent.invoke({
        "input": question,
        "agent_scratchpad": f"Human: {question}\nAI: To answer this question, I need to use Python to analyze the dataframe. I'll use the python_repl_ast tool.\n\nAction: python_repl_ast\nAction Input: ",
    })
    print(f"Question: {question}")
    print(f"Answer: {response}")
    print("---")


# Example questions
ask_agent("What are the column names in this dataset?")
ask_agent("How many rows are in this dataset?")
ask_agent("What is the average price of cars sold?")