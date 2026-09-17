import os
from crewai import Agent, Task, Crew
from markdown import markdown
from crewai_tools import WebsiteSearchTool


from dotenv import load_dotenv

# Load environment variables before importing deepeval/langchain
load_dotenv()


### Sequential Multi-agent Architecture Overview

# * Each agent has a specialized responsibility:
#   * Receipt Interpreter: extracts structured grocery data.
#   * Expiration Estimator: enriches items with estimated shelf life using external information.
#   * Grocery Tracker: manages and validates the current inventory.
#   * Recipe Recommender: uses the remaining inventory to suggest recipes.
# * Tasks are connected through explicit **context passing**, so the output of one stage becomes the input of the next.
# * The Grocery Tracker introduces a **human-in-the-loop step**, allowing the inventory to be reviewed before recipe generation.
# * External tools are isolated to the agents that need them, such as the expiration agent's website search.
# * Delegation is disabled, keeping agent responsibilities clearly separated and making the workflow easier to understand and debug.

# Overall, this is closer to a **deterministic agent workflow with LLM-powered steps** than an open-ended autonomous multi-agent system.



# Load the markdown receipt file
with open('data/grocery_receipt.md', 'r') as f:
    receipt_markdown = markdown(f.read())

# Today's date for reference
today = "2026-09-17"
print("Receipt loaded successfully!")

# Agents

receipt_interpreter_agent = Agent(
    role="Receipt Markdown Interpreter",
    goal=(
        "Accurately extract items, their counts, and weights with units from a given receipt in markdown format. "
        "Provide structured data to support the grocery management system."
    ),
    backstory=(
        "As a key member of the grocery management crew for the household, your mission is to meticulously extract "
        "details such as item names, quantities, and weights from receipt markdown files. Your role is vital for the "
        "grocery tracker agent, which monitors the household's inventory levels."
    ),
    personality=(
        "Diligent, detail-oriented, and efficient. The Receipt Markdown Interpreter is committed to providing accurate "
        "and structured information to support effective grocery management. It is particularly focused on clarity and precision."
    ),
    allow_delegation=False,
    verbose=True
)

# Use website earch tool to search the website "www.stilltasty.com"
expiration_date_search_web_tool = WebsiteSearchTool(website='https://www.stilltasty.com/')

expiration_date_search_agent = Agent(
    role="Expiration Date Estimation Specialist",
    goal=(
        "Accurately estimate the expiration dates of items extracted by the Receipt Markdown Interpreter Agent. "
        "Utilize online sources to determine typical shelf life when refrigerated and add the estimated number of days to the purchase date."
    ),
    backstory=(
        "As the Expiration Date Estimation Specialist, your role is to ensure the household's groceries are consumed before expiration. "
        "You use your access to online resources to search for the best estimates on how long each item typically lasts when stored properly."
    ),
    personality=(
        "Meticulous, resourceful, and reliable. This agent ensures the household maintains a well-stocked but efficiently used inventory, minimizing waste."
    ),
    allow_delegation=False,
    verbose=True,
    tools=[expiration_date_search_web_tool]
)


grocery_tracker_agent = Agent(
    role="Grocery Inventory Tracker",
    goal=(
        "Accurately track the remaining groceries based on user consumption input. "
        "Subtract consumed items from the grocery list obtained from the Expiration Date Estimation Specialist and update the inventory. "
        "Provide the user with an updated list of what's left, along with corresponding expiration dates."
    ),
    backstory=(
        "As the household's Grocery Inventory Tracker, your responsibility is to ensure that groceries are accurately tracked based on user input. "
        "You need to understand the user's input on what they've consumed, update the inventory list, and remind them of what's left and the expiration dates. "
        "Your role is crucial in helping the household avoid waste and ensure timely consumption of perishable items."
    ),
    personality=(
        "Helpful, detail-oriented, and responsive. This agent is focused on ensuring the household has an up-to-date inventory, minimizing waste, and helping users stay organized."
    ),
    allow_delegation=False,
    verbose=True
)

recipe_web_tool = WebsiteSearchTool(website='https://www.americastestkitchen.com/recipes')

# Optimized Grocery Recipe Recommendation Agent
rest_grocery_recipe_agent = Agent(
    role="Grocery Recipe Recommendation Specialist",
    goal=(
        "Provide recipe recommendations using the remaining groceries in the inventory. "
        "Avoid using items with a count of 0 and prioritize recipes that maximize the use of available ingredients. "
        "If ingredients are insufficient, suggest restocking recommendations."
    ),
    backstory=(
        "As a Grocery Recipe Recommendation Specialist, your mission is to help the household make the most out of their remaining groceries. "
        "Your role is to search the web for easy, delicious recipes that utilize available ingredients while minimizing waste. "
        "Ensure that the recipes are simple to follow and use as many of the remaining ingredients as possible."
    ),
    personality=(
        "Creative, resourceful, and efficient. This agent is dedicated to helping the household create enjoyable meals with what they have on hand."
    ),
    allow_delegation=False,
    verbose=True,
    tools=[recipe_web_tool],
    human_input=True
)

# Tasks

read_receipt_task = Task(
    agent=receipt_interpreter_agent,
    description=(
        f"Analyze the receipt markdown file provided: {receipt_markdown}. "
        "Extract information on items purchased, their counts, weights, and units. "
        f"Additionally, extract today's date information which is provided here: {today}. "
        "Ensure all item names are converted into clear, human-readable text."
    ),
    expected_output="""
    {
        "items": [
            {
                "item_name": "string - Human-readable name of the item",
                "count": "integer - Number of units purchased",
                "unit": "string - Unit of measurement (e.g., kg, lbs, pcs)"
            }
        ],
        "date_of_purchase": "string - Date in YYYY-MM-DD format"
    }
    """
)


expiration_date_search_task = Task(
    agent=expiration_date_search_agent,
    description=(
        "Using the list of items extracted by the Receipt Markdown Interpreter Agent, search online to find the typical shelf life of each item when refrigerated. "
        "Add this information to the date of purchase to estimate the expiration date for each item."
        "Ensure that the output includes the item name, count, unit, and estimated expiration date."
    ),
    expected_output="""
    {
        "items": [
            {
                "item_name": "string - Human-readable name of the item",
                "count": "integer - Number of units purchased",
                "unit": "string - Unit of measurement (e.g., kg, lbs, pcs)",
                "expiration_date": "string - Estimated expiration date in YYYY-MM-DD format"
            }
        ]
    }
    """,
    context=[read_receipt_task]
)

grocery_tracking_task = Task(
    agent=grocery_tracker_agent,
    description=(
        "Using the grocery list with expiration dates provided by the Expiration Date Estimation Specialist, "
        "update the inventory based on user input about items they have consumed. "
        "Subtract the consumed quantities from the inventory list and provide a summary of what items are left, including their expiration dates. "
        "Ensure that the updated list is returned in JSON format."
    ),
    expected_output="""
    {
        "items": [
            {
                "item_name": "string - Human-readable name of the item",
                "count": "integer - Updated number of units remaining",
                "unit": "string - Unit of measurement (e.g., kg, lbs, pcs)",
                "expiration_date": "string - Estimated expiration date in YYYY-MM-DD format"
            }
        ]
    }
    """,
    context=[expiration_date_search_task],
    human_input=True,
    output_file = "data/grocery_tracker.json"
)

recipe_recommendation_task = Task(
    agent=rest_grocery_recipe_agent,
    description=(
        "Using the updated grocery list provided by the Grocery Inventory Tracker, "
        "search online for recipes that utilize the available ingredients. "
        "Only include items with a count greater than zero. If no suitable recipe can be found, provide restocking recommendations. "
        "Ensure that the output includes recipe names, ingredients, instructions, and the source website."
    ),
    expected_output="""
    {
        "recipes": [
            {
                "recipe_name": "string - Name of the recipe",
                "ingredients": [
                    {
                        "item_name": "string - Ingredient name",
                        "quantity": "string - Quantity required",
                        "unit": "string - Measurement unit (e.g., kg, pcs, tbsp)"
                    }
                ],
                "steps": [
                    "string - Step-by-step instructions for the recipe"
                ],
                "source": "string - Website URL for the recipe"
            }
        ],
        "restock_recommendations": [
            {
                "item_name": "string - Name of the item to restock",
                "quantity_needed": "integer - Suggested quantity to purchase",
                "unit": "string - Measurement unit (e.g., kg, pcs)"
            }
        ]
    }
    """,
    context=[grocery_tracking_task],
    output_file = "data/recipe_recommendation.json"
)

# Running

# Create a crew with the agent and task, assigning the tasks to the agents in the order they should be executed
crew = Crew(agents=[receipt_interpreter_agent, 
                    expiration_date_search_agent, 
                    grocery_tracker_agent, 
                    rest_grocery_recipe_agent], 
            tasks=[read_receipt_task, 
                   expiration_date_search_task, 
                   grocery_tracking_task, 
                   recipe_recommendation_task],
            verbose=True)

# Kick off the crew and type ENTER for each input (human input is true)
result = crew.kickoff()