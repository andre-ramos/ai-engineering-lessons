# LangGraph is a framework for creating applications using graph-based workflows. 
# Each node represents a function or computational step, and edges define the flow between these nodes based on certain conditions.


#In this tutorial, we'll demonstrate the power of LangGraph by building a multi-step text analysis pipeline. Our use case will focus on processing a given text through three key stages:

#Text Classification: We'll categorize the input text into predefined categories (e.g., News, Blog, Research, or Other).
#Entity Extraction: We'll identify and extract key entities such as persons, organizations, and locations from the text.
#Text Summarization: Finally, we'll generate a concise summary of the input text.

import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from IPython.display import display, Image

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# State Class to hold the text, classification, entities, and summary
class State(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)



def classification_node(state: State):
    ''' Classify the text into one of the categories: News, Blog, Research, or Other '''
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Classify the following text into one of the categories: News, Blog, Research, or Other.\n\nText:{text}\n\nCategory:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    classification = llm.invoke([message]).content.strip()
    return {"classification": classification}


def entity_extraction_node(state: State):
    ''' Extract all the entities (Person, Organization, Location) from the text '''
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Extract all the entities (Person, Organization, Location) from the following text. Provide the result as a comma-separated list.\n\nText:{text}\n\nEntities:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    entities = llm.invoke([message]).content.strip().split(", ")
    return {"entities": entities}


def summarization_node(state: State):
    ''' Summarize the text in one short sentence '''
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in one short sentence.\n\nText:{text}\n\nSummary:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    summary = llm.invoke([message]).content.strip()
    return {"summary": summary}


workflow = StateGraph(State)

workflow.add_node("classification", classification_node)
workflow.add_node("entity_extraction", entity_extraction_node)
workflow.add_node("summarization", summarization_node)

workflow.set_entry_point("classification")
# Add edges to the graph to define the flow of execution
workflow.add_edge("classification", "entity_extraction")
workflow.add_edge("entity_extraction", "summarization")
workflow.add_edge("summarization", END)

app = workflow.compile()


sample_text = """
LONDON -- A group of masked men blocked the entrance to Britain’s busiest ferry port on Saturday. Police said they were “engaging with the protesters."
The Port of Dover said all traffic in and out “is stopped due to an ongoing public order incident.”
Footage on social media showed several dozen men in black, their faces covered, standing across a major road near the port.
It was not immediately clear what they were demanding.
The National Highways agency said the protests caused a 4-mile (6.4-kilometer) traffic jam, causing delays of up to 45 minutes.
"""

state_input = {"text": sample_text}
result = app.invoke(state_input)

print("Classification:", result["classification"])
print("\nEntities:", result["entities"])
print("\nSummary:", result["summary"])