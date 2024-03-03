import asyncio
from langchain.agents import Agent, Tool, initialize_agent
from langchain.chains import ConversationChain
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               MessagesPlaceholder, SystemMessagePromptTemplate)
from langchain_community.utilities import GoogleSerperAPIWrapper, SerpAPIWrapper
from .conversation_utils import is_asking_for_smart_mode, get_recommended_temperature
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from slack_sdk import WebClient
from .AsyncStreamingSlackCallbackHandler import AsyncStreamingSlackCallbackHandler

DEFAULT_MODEL = "gpt-3.5-turbo"
UPGRADE_MODEL = "gpt-4"
DEFAULT_TEMPERATURE = 0.3

class ConversationAI:
    def __init__(self, bot_name: str, slack_client: WebClient, existing_thread_history=None, model_name: str = None):
        # Initialization
        self.bot_name, self.slack_client, self.existing_thread_history = bot_name, slack_client, existing_thread_history
        self.model_name, self.agent, self.model_temperature = None, None, None
        self.lock = asyncio.Lock()

    async def create_agent(self, sender_user_info, initial_message):
        # Creating a conversation agent
        sender_profile = sender_user_info["profile"]

        # Asynchronously check if smart mode is requested and get recommended temperature
        smart_mode, recommended_temperature = await asyncio.gather(
            is_asking_for_smart_mode(initial_message),
            get_recommended_temperature(initial_message, DEFAULT_TEMPERATURE)
        )

        # Determine model and temperature based on user input
        self.model_name = UPGRADE_MODEL if smart_mode else DEFAULT_MODEL
        self.model_temperature = max(0.0, min(recommended_temperature or DEFAULT_TEMPERATURE, 1.0))

        print(f"Will use model: {self.model_name}")
        print(f"Will use temperature: {self.model_temperature}")

        model_facts = f"You are based on the OpenAI model {self.model_name}. Your 'creativity temperature' is set to {self.model_temperature}."

        # Define conversation prompts
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(f"""The following is a Slack chat thread between users and you, a Slack bot named {self.bot_name}.
You are funny and smart, and you are here to help.
...
"""),
            HumanMessagePromptTemplate.from_template(f"""Here is some information about me. Do not respond to this directly, but feel free to incorporate it into your responses:
...
"""),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        self.callbackHandler = AsyncStreamingSlackCallbackHandler(self.slack_client)
        llm = ChatOpenAI(model_name=self.model_name, temperature=self.model_temperature, request_timeout=60, max_retries=3,
                         streaming=True, verbose=True, callback_manager=AsyncCallbackManager([self.callbackHandler]))

        memory = ConversationBufferMemory(return_messages=True)
        existing_thread_history = self.existing_thread_history

        # Populate memory with existing thread history
        if existing_thread_history:
            for message in existing_thread_history:
                sender_name, message_content = list(message.items())[0]
                getattr(memory.chat_memory, f"add_{'ai' if sender_name == 'bot' else 'user'}_message")(message_content)

        self.memory, self.agent = memory, ConversationChain(memory=memory, prompt=prompt, llm=llm, verbose=True)
        return self.agent

    async def get_or_create_agent(self, sender_user_info, message):
        # Get or create a conversation agent
        if self.agent is None:
            self.agent = await self.create_agent(sender_user_info, message)
        return self.agent

    async def respond(self, sender_user_info, channel_id: str, thread_ts: str, message_being_responded_to_ts: str, message: str):
        # Generate a response
        async with self.lock:
            agent = await self.get_or_create_agent(sender_user_info, message)
            print("Starting response...")
            await self.callbackHandler.start_new_response(channel_id, thread_ts)
            response = await self.agent.apredict(input=message)
            return response


def get_conversational_agent(model_name="gpt-3.5-turbo"):
    # Get a conversational agent
    search, memory = GoogleSerperAPIWrapper(), ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    search, tools = SerpAPIWrapper(), [Tool(name="search", func=search.run, description="...")]
    llm = ChatOpenAI(temperature=0, model=model_name, verbose=True, request_timeout=30, max_retries=0)
    agent_chain = initialize_agent(tools, llm, agent="chat-conversational-react-description", verbose=True, memory=memory, request_timeout=30)
    return agent_chain
