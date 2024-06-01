from dotenv import load_dotenv
import os
from langchain_openai import AzureOpenAI  # Updated import
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.callbacks import get_openai_callback
from langchain.agents import Tool, AgentType
from langchain.chains import LLMChain
# New imports for agent creation
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent  # Assuming this as an example

# Load environment variables
load_dotenv()

# Correctly retrieving API key from .env
API_KEY = os.getenv('OPENAI_API_KEY')
# Setup Azure OpenAI LLM
llm = AzureOpenAI(
    temperature=0,
    openai_api_key=API_KEY,
    azure_endpoint="",  # Updated parameter
    deployment_name="",
    api_version=""
)

# Connect to the SQLite database
db = SQLDatabase.from_uri("sqlite:///C:/sqlite/db/Chinook.db")

# Create a SQLDatabaseChain to handle SQL queries (Check if any updates are needed here)
db_chain = SQLDatabaseChain.from_llm(llm=llm, db=db, verbose=True)

# Define the tools that your agent will use for interacting with the database
tools = [
    Tool(
        name="Chinook DB",
        func=db_chain.run,
        description="Useful for when you need to answer questions about Chinook. Input should be in the form of a question containing full context"
    )
]

template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: {input}
Thought:{agent_scratchpad}
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
"""

custom_prompt_template = PromptTemplate.from_template(template)

# Initialize the agent with the defined tools and a default prompt
agent = create_react_agent(tools=tools, llm=llm, prompt=custom_prompt_template)

# Function to handle database queries
def handle_db_queries(query):
    input_dict = {"input": query}  # Adjust "input" if a different key is expected
    return agent.invoke(input_dict)

