import json
from langchain.schema.messages import HumanMessage, AIMessage
from datetime import datetime
import yaml

# load_config is a function that loads the configuration from the config.yaml file
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

# save_chat_history_json is a function that saves the chat history in a json file
def save_chat_history_json(chat_history, file_path):
    with open(file_path, "w") as f:
        json_data = [message.dict() for message in chat_history]
        json.dump(json_data, f)

# load_chat_history_json is a function that loads the chat history from a json file
def load_chat_history_json(file_path):
    with open(file_path, "r") as f:
        json_data = json.load(f)
        messages = [HumanMessage(**message) if message["type"] == "human" else AIMessage(**message) for message in json_data]
        return messages

# get_timestamp is a function that returns the current timestamp
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# get_avatar is a function that returns the avatar of the sender
def get_avatar(sender_type):
    if sender_type == "human":
        return "chat_icons/user_image.png"
    else:
       return "chat_icons/bot_image.png"