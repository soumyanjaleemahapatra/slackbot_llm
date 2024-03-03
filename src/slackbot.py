import logging
import os
import re
from typing import List
from langchain_community.llms import OpenAI
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.errors import SlackApiError
from .conversationAI import ConversationAI

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get('SLACK_APP_TOKEN')
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")

class Slackbot:
    ## Bot initialization
    def __init__(self, slack_app: AsyncApp):
        # Dictionary to track threads in which the bot is participating
        self.threads_bot_is_participating_in = {}
        
        # Reference to the Slack app instance
        self.app = slack_app
        
        # Reference to the Slack client associated with the app
        self.client = self.app.client
        
        # Cache to store mappings from user IDs to user names
        self.id_to_name_cache = {}
        
        # Cache to store user information based on user IDs
        self.user_id_to_info_cache = {}

    ## Bot Startup process
    async def start(self):
        # Log a message indicating the start of the bot setup
        logger.info("Looking up bot user_id. (If this fails, something is wrong with the auth)")
        
        # Authenticate and fetch information about the bot user
        response = await self.app.client.auth_test()

        # Extract the bot user_id from the authentication response
        self.bot_user_id = response["user_id"]

        # Fetch and store the bot user name using the user_id
        self.bot_user_name = await self.get_username_for_user_id(self.bot_user_id)
        
        # Log the bot user_id and user name for reference
        logger.info("Bot user id: "+ self.bot_user_id)
        logger.info("Bot user name: "+ self.bot_user_name)

        # Start the bot using AsyncSocketModeHandler
        await AsyncSocketModeHandler(app, SLACK_APP_TOKEN).start_async()

    ## Message handling
    async def on_message(self, event, say):
        # Extract relevant information from the received message event
        message_ts = event['ts']
        thread_ts = event.get('thread_ts', message_ts)

        try:
            # Check if the bot sent the message, if yes, ignore it
            if event.get('user', None) == self.bot_user_id:
                logger.debug("Not handling message event since the bot sent the message.")
                return

            # Determine if the bot should start participating in the thread
            start_participating_if_not_already = False
            channel_id = event['channel']
            channel_type = event.get('channel_type', None)

            # Check if the message is in a direct message (im) or mentions the bot in a channel
            if channel_type and channel_type == "im":
                start_participating_if_not_already = True
            elif self.is_bot_mentioned(event['text']):
                start_participating_if_not_already = True

            # If the bot should start participating, add AI to the thread
            if start_participating_if_not_already:
                await self.add_ai_to_thread(channel_id, thread_ts, message_ts)

            # Check if the bot is already participating in the thread
            if self.is_ai_participating_in_thread(thread_ts, message_ts):
                user_id = event['user']
                text = event['text']

                # Confirm the message is received by adding a reaction
                await self.confirm_message_received(channel_id, thread_ts, message_ts, user_id)

                # Respond to the message using the conversation AI
                await self.respond_to_message(channel_id, thread_ts, message_ts, user_id, text)

        except Exception as e:
            # Handle any exceptions that might occur during message processing
            response = f":exclamation::exclamation::exclamation: Error: {e}"
            logger.exception(response)
            await say(text=response, thread_ts=thread_ts)

    ## Thread management (Adding a new ConversationAI instance to the threads the bot is participating in) 
    async def add_ai_to_thread(self, channel_id, thread_ts, message_ts):
        # Check if the bot is already participating in the thread
        if thread_ts in self.threads_bot_is_participating_in:
            return

        processed_message_history = None
        
        # Check if this message is the first in the thread
        if not Slackbot.is_parent_thread_message(message_ts, thread_ts):
            logger.debug("It looks like I am not the first message in the thread. I should get the full thread history from Slack and add it to my memory.")
            
            # Retrieve the full thread history from Slack
            thread_history = await client.conversations_replies(channel=channel_id, ts=thread_ts)
            processed_message_history = []

            # Iterate through the thread history
            message_history = thread_history.data['messages']
            
            # Exclude the last message (the one the bot is responding to)
            message_history = message_history[:-1]

            # Process each message in the history
            for message in message_history:
                text = message['text']
                text = await self.translate_mentions_to_names(text)
                user_id = message['user']
                user_name = await self.get_username_for_user_id(user_id)

                # Add messages to the processed history
                if (user_id == self.bot_user_id):
                    processed_message_history.append({"bot": text})
                else:
                    # Get the username for this user_id:
                    processed_message_history.append({f"{user_name}": text})

         # Create a new ConversationAI instance with the processed history
        ai = ConversationAI(self.bot_user_name, self.client, processed_message_history)
        
        # Add the ConversationAI instance to the threads the bot is participating in
        self.threads_bot_is_participating_in[thread_ts] = ai

    def is_ai_participating_in_thread(self, thread_ts, message_ts):
        # Check if the bot is participating in the specified thread
        if thread_ts in self.threads_bot_is_participating_in:
            return True
        return False

    def is_bot_mentioned(self, text):
        # Check if the bot is mentioned in the given text
        return f"<@{self.bot_user_id}>" in text

    async def confirm_message_received(self, channel, thread_ts, message_ts, user_id_of_sender):
        # React to the message with a thinking face emoji:
        try:
            # Use the reactions_add method from the Slack SDK to add a "thinking_face" reaction
            await self.client.reactions_add(channel=channel, name="thinking_face", timestamp=message_ts)
        except Exception as e:
            # Handle exceptions that may occur during the reaction addition
            logger.exception(e)

    async def respond_to_message(self, channel_id, thread_ts, message_ts, user_id, text):
        try:
            # Attempt to respond to the received message using the conversation AI
            conversation_ai: ConversationAI = self.threads_bot_is_participating_in.get(thread_ts, None)
        
            # Check if a ConversationAI instance is found for the given thread_ts
            if conversation_ai is None:
                raise Exception("No AI found for thread_ts")

            # Translate mentions in the message text to user names
            text = await self.translate_mentions_to_names(text)

            # Fetch information about the sender of the message
            sender_user_info = await self.get_user_info_for_user_id(user_id)

            # Use the conversation AI to generate a response
            response = await conversation_ai.respond(sender_user_info, channel_id, thread_ts, message_ts, text)

            # If the response is None, indicate that the bot won't respond with an emoji
            if response is None:
                await self.confirm_wont_respond_to_message(channel_id, thread_ts, message_ts, user_id)

            # We don't respond here since the bot is streaming responses

        except Exception as e:
            # Handle exceptions that may occur during response generation
            response = f":exclamation::exclamation::exclamation: Error: {e}"

            # Log the error details in red to the console
            logger.exception(response)

            # Reply to Slack with the error message
            await self.reply_to_slack(channel_id, thread_ts, message_ts, response)

    async def translate_mentions_to_names(self, text):
        # Replace every @mention of a user id with their actual name:
        # First, use a regex to find @mentions that look like <@U123456789>:
        matches = re.findall(r"<@(U[A-Z0-9]+)>", text)
        for match in matches:
            mention_string = f"<@{match}>"
            # Get the actual name for the user ID
            mention_name = await self.get_username_for_user_id(match)
            if mention_name is not None:
                # Replace the mention in the text with @username
                text = text.replace(mention_string, "@" + mention_name)

        return text

    async def get_username_for_user_id(self, user_id):
        # Fetch user information using the provided user ID
        user_info = await self.get_user_info_for_user_id(user_id)

        # Extract the profile information from the user details
        profile = user_info['profile']

        # Check if the user is a bot
        if (user_info['is_bot']):
            # If the user is a bot, use the real name from the profile
            ret_val = profile['real_name']
        else:
            # If the user is not a bot, use the display name from the profile
            ret_val = profile['display_name']

        # Return the determined username (real_name for bots, display_name for others)
        return ret_val
    
    async def get_user_info_for_user_id(self, user_id):
        # Check if user information is already in the cache
        user_info = self.user_id_to_info_cache.get(user_id, None)
        if user_info is not None:
            # If cached, return the cached user information
            return user_info
        
        # If not cached, make an API call to Slack API to get user information
        user_info_response = await self.app.client.users_info(user=user_id)
        user_info = user_info_response['user']
        
        # Log the user information for debugging purposes
        logger.debug(user_info)

        # Cache the user information for future use
        self.user_id_to_info_cache[user_id] = user_info

        # Return the retrieved user information
        return user_info

    @staticmethod
    def is_parent_thread_message(message_ts, thread_ts):
        # Check if the given message timestamp is the same as the thread timestamp
        return message_ts == thread_ts

    async def confirm_wont_respond_to_message(self, channel, thread_ts, message_ts, user_id_of_sender):
        # React to the message with a speak_no_evil emoji:
        try:
            # Attempt to add a speak_no_evil emoji reaction to the message
            await self.client.reactions_add(channel=channel, name="speak_no_evil", timestamp=message_ts)
        except Exception as e:
            # If an exception occurs during the reaction addition, log the exception
            logger.exception(e)

    ## Handling user joining a channel
    async def on_member_joined_channel(self, event_data): 
        # Extract user ID and channel ID from the event data
        user_id = event_data["user"]
        channel_id = event_data["channel"]

        # Fetch user information and username using the user ID
        user_info = await self.get_user_info_for_user_id(user_id)
        username = await self.get_username_for_user_id(user_id)

        # Extract profile information for the user (if available)
        profile = user_info.get("profile", {})

        # Initialize an OpenAI instance for creative message generation
        llm_gpt3_turbo = OpenAI(temperature=1, model_name="gpt-3.5-turbo", request_timeout=30, max_retries=5, verbose=True)

        # TODO: Consider extracting the welcome message template into a YAML file for better maintainability
        # Generate a welcome message using GPT-3.5 Turbo based on a template
        welcome_message = (await llm_gpt3_turbo.agenerate([f"""
        You are a funny and creative slackbot {self.bot_user_name}
        Someone just joined a Slack channel you are a member of, and you want to welcome them creatively and in a way that will make them feel special.
        You are VERY EXCITED about someone joining the channel, and you want to convey that!
        Their username is {username}, but when you mention their username, you should say "<@{user_id}>" instead.
        Their title is: {profile.get("title")}
        Their current status: "{profile.get("status_emoji")} {profile.get("status_text")}"
        Write a Slack message, formatted in Slack markdown, that encourages everyone to welcome them to the channel excitedly.
        Use emojis. Maybe write a song. Maybe a poem.
        
        Afterwards, tell the user that you look forward to "chatting" with them, and mention that they can just mention <@{self.bot_user_id}> whenever they want to talk.
        """])).generations[0][0].text

        # Check if a welcome message was successfully generated
        if welcome_message:
            try:
                # Send the welcome message to the user in the channel
                await self.client.chat_postMessage(channel=channel_id, text=welcome_message)
            except Exception as e:
                # Log an exception if there's an error sending the welcome message
                logger.exception("Error sending welcome message")

    async def upload_snippets(self, channel_id: str, thread_ts: str, response: str) -> str:
        # Find all code snippets in the response using a regular expression
        matches: List[str] = re.findall(r"```(.*?)```", response, re.DOTALL)
    
        # Initialize a counter for snippet numbering
        counter: int = 1
    
        # Iterate over each code snippet found
        for match in matches:
            # Remove leading and trailing whitespaces from the snippet
            match = match.strip()
            
            # Extract the first line of the snippet
            first_line: str = match.splitlines()[0]
            
            # Extract the first word from the first line (often representing the programming language)
            first_word: str = first_line.split()[0]
            
            # Determine the file extension based on the detected programming language
            extension: str = "txt"
            if first_word == "python":
                extension = "py"
            elif first_word in ["javascript", "typescript"]:
                extension = "js"
            elif first_word == "bash":
                extension = "sh"
            
            # If the extension is not determined yet, use the first word or default to "txt"
            if not extension:
                if first_word:
                    extension = first_word
                else:
                    extension = "txt"
            
            # Upload the snippet as a file to Slack using the files_upload API method
            file_response = await self.client.files_upload(channels=channel_id, content=match, filename=f"snippet_{counter}.{extension}", thread_ts=thread_ts)
            
            # Retrieve the file ID from the API response
            file_id: str = file_response["file"]["id"]
            
            # Modify the response by appending a link to the uploaded file
            response += "\n" + f"<https://slack.com/files/{self.bot_user_id}/{file_id}|code.{extension}>"
            
            # Increment the counter for the next snippet
            counter += 1
    
        # Return the modified response
        return response
    async def reply_to_slack(self, channel_id, thread_ts, message_ts, response):
        # Check if the response consists of only a single emoji (enclosed in colons)
        slack_emoji_regex = r"^:[a-z0-9_+-]+:$"
        if re.match(slack_emoji_regex, response.strip()):
            try:
                # Extract the emoji name by removing colons
                emoji_name = response.strip().replace(":", "")
                logger.info("Responding with single emoji: " + emoji_name)
                
                # React to the message with the identified emoji
                await self.client.reactions_add(channel=channel_id, name=emoji_name, timestamp=message_ts)
            except Exception as e:
                # Log any exception that occurs during the reaction addition
                logger.exception(e)
            return
        else:
            # If the response is not a single emoji, post a regular message to the Slack channel
            await self.client.chat_postMessage(channel=channel_id, text=response, thread_ts=thread_ts)


app = AsyncApp(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)
client = app.client
slack_bot = Slackbot(app)

## Event handlers
# Event handler for the "message" event
@app.event("message")
async def on_message(payload, say):
    # Log that the message is being processed
    logger.info("Processing message...")
    
    # Call the on_message method of the SlackBot instance to handle the message
    await slack_bot.on_message(payload, say)


# Event handler for the "member_joined_channel" event
@app.event("member_joined_channel")
async def handle_member_joined_channel(event_data):
    # Log that the member_joined_channel event is being processed along with event_data
    logger.info("Processing member_joined_channel event", event_data)
    
    # Call the on_member_joined_channel method of the SlackBot instance to handle the event
    await slack_bot.on_member_joined_channel(event_data)


# Event handler for the "reaction_added" event
@app.event('reaction_added')
async def on_reaction_added(payload):
    # Log that the reaction_added event is being ignored
    logger.info("Ignoring reaction_added")


# Event handler for the "reaction_removed" event
@app.event('reaction_removed')
async def on_reaction_removed(payload):
    # Log that the reaction_removed event is being ignored
    logger.info("Ignoring reaction_removed")


# Event handler for the "app_mention" event
@app.event('app_mention')
async def on_app_mention(payload, say):
    # Log that the app_mention event is being ignored in favor of handling it via the message handler
    logger.info("Ignoring app_mention in favor of handling it via the message handler...")


