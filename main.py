#llama-index
#pypdf
#python-dotenv

from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
#from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from pdf import canada_engine


load_dotenv()

output_file = open('output.txt', 'w')

#Create df from given csv file
population_path = os.path.join("data", "population.csv")
population_df = pd.read_csv(population_path)

#Wrapping the df and giving us an interface to ask questions about the data
population_query_engine = PandasQueryEngine(
    df=population_df, verbose=False, instruction_str=instruction_str
)

#Creating and updating a prompt
population_query_engine.update_prompts({"pandas_prompt": new_prompt})
#population_query_engine.query("What is the population of canada") 

# Define tools for the agent to use, each tool wraps a query engine
tools = [
    #tool,
    QueryEngineTool(query_engine=population_query_engine, 
                    metadata=ToolMetadata(
                        name="population_data",
                        description="this gives information about the world population and demographics"
                    ),
    ),
    QueryEngineTool(query_engine=canada_engine, 
                    metadata=ToolMetadata(
                        name="canada_data",
                        description="this gives detailed info about British Columbia strata property act"
                    ),
    ),
]

# Initialize the LLM 
llm = OpenAI(model="gpt-3.5-turbo")

# Create a ReActAgent from the defined tools and LLM
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

while(prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    output_file.write(str(result) + '\n')
    output_file.flush()
    

output_file.close()