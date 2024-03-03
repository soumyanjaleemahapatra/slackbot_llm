import os
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain_community.utilities import GoogleSerperAPIWrapper, SerpAPIWrapper
from langchain.prompts import PromptTemplate
from typing import Dict, List
from pydantic import BaseModel

# Define Pydantic models for representing language model output
class Generation(BaseModel):
    text: str
    generation_info: str = None

class TokenUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int

class LlmOutput(BaseModel):
    token_usage: TokenUsage
    model_name: str

class Output(BaseModel):
    generations: List[List[Generation]]
    llm_output: LlmOutput

# Initialize the GPT-3.5 Turbo language model
llm_gpt3_turbo = OpenAI(temperature=0, model_name="gpt-3.5-turbo", request_timeout=30, max_retries=2, verbose=True)

# Utility function to get a simple response from the language model
async def get_simple_response(input: str) -> Output:
    return await llm_gpt3_turbo.agenerate([input])

# Utility function to determine if the input requests "smart mode"
async def is_asking_for_smart_mode(input: str) -> bool:
    # Define a prompt template to identify smart mode requests
    prompt = PromptTemplate(
        input_variables=["input"],
        template="""Determine if the following input contains explicit requests like increased intelligence, extra thinking, gpt4, expensiveness, slowness, etc.
        If so, return "smart_mode: yes". If not, return "smart_mode: no".
        Examples:
        <!begin_input> Hey Chatterbot, I am gonna need you to think real hard about this one! No need to be creative since I'm just gonna talk about code. <!end_input>
        smart_mode: yes
        <!begin_input> Hey Chatterbot, let's brainstorm some funny song titles! <!end_input>
        smart_mode: no
        <!begin_input> Help me code. <!end_input>
        smart_mode: no
        <!begin_input> {input} <!end_input>
        """
    )
    # Format the query with the input and get the model response
    query = prompt.format(input=input)
    print("About to ask GPT-3.5 Turbo about:", query)
    
    try:
        response: Output = await get_simple_response(query)
        response = response.generations[0][0].text
        response = response.split("smart_mode: ")[1].strip().lower()
        return response == "yes"
    except Exception as e:
        print("Error in is_asking_for_smart_mode", e)
        return False

# Utility function to get the recommended temperature for the language model's response
async def get_recommended_temperature(input: str, default_temperature=0.3) -> float:
    # Define a prompt template to ask for the appropriate temperature
    prompt = PromptTemplate(
        input_variables=["input", "default_temperature"],
        template="""Please indicate the appropriate temperature for the LLM to respond to the following message, using a scale from 0.00 to 1.00.
        For tasks that require maximum precision, such as coding, please use a temperature of 0.
        For tasks that require more creativity, such as generating imaginative responses, use a temperature of 0.7-1.0.
        If an explicit temperature/creativity is requested, use that.
        If the appropriate temperature is unclear, please use a default of {default_temperature}.
        Please note that the temperature should be selected based solely on the nature of the task and should not be influenced by the complexity or sophistication of the message.
        Examples:
        <!begin_input> Get as creative as possible for this one! <!end_input>
        temperature: 1.00
        <!begin_input> Tell me a bedtime story about a dinosaur! <!end_input>
        temperature: 0.80
        <!begin_input> Let's write some code. (Be really smart please) <!end_input>
        temperature: 0.00
        <!begin_input> Temperature:88% Model: Super duper smart! <!end_input>
        temperature: 0.88
        <!begin_input> How are you doing today? <!end_input>
        temperature: {default_temperature}
        ###
        <!begin_input>: {input} <!end_input>
        """
    )
    # Format the query with the input and default temperature, then get the model response
    query = prompt.format(default_temperature=default_temperature, input=input)
    print("About to ask GPT-3.5 Turbo about:", query)
    
    try:
        response: Output = await get_simple_response(query)
        response = response.generations[0][0].text
        print("Response:", response)
        response = response.split("temperature: ")[1].strip().lower()
        # Try to parse the response as a float
        try:
            return float(response)
        except:
            return default_temperature
    except Exception as e:
        print("Error in get_recommended_temperature", e)
        return default_temperature

# Utility function to get a conversational agent
def get_conversational_agent(model_name="gpt-3.5-turbo"):
    # Initialize memory for conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # Initialize a tool for searching information about dinosaurs
    tools = [
        Tool(
            name="dinosaurs",
            func=GoogleSerperAPIWrapper().run,
            description="(Useful to learn about dinosaurs)."
        ),
        Tool(
            name="search",
            func=SerpAPIWrapper().run,
            description="(ONLY if your confidence in your answer is below 0.2, use this tool to search for information)"
        ),
    ]
    # Initialize the GPT-3.5 Turbo language model for conversation
    llm = ChatOpenAI(temperature=0, model=model_name, verbose=True, request_timeout=30, max_retries=0)
    # Initialize a conversational agent chain
    agent_chain = initialize_agent(tools, llm, agent="chat-conversational-react-description", verbose=True, memory=memory, request_timeout=30)
    return agent_chain
