from langchain_community.chat_loaders.whatsapp import WhatsAppChatLoader
from typing import List
from langchain_openai import AzureOpenAI
from dotenv import load_dotenv
import os
from langchain_community.chat_loaders.base import ChatSession
from langchain_community.chat_loaders.utils import (
    map_ai_messages,
    merge_chat_runs,
)
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
def load_and_process_chat_history(file_path):
    """
    Load and process the chat history from a .txt file.
    """
    chat_history = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if ']' in line and ':' in line:
                    timestamp, message = line.split(']', 1)
                    timestamp = timestamp[1:].strip()
                    sender, content = message.split(':', 1)
                    chat_message = {
                        'timestamp': timestamp,
                        'sender': sender.strip(),
                        'content': content.strip()
                    }
                    chat_history.append(chat_message)
                else:
                    print(f"Skipping line due to unexpected format: {line.strip()}")
        return chat_history
    except Exception as e:
        print(f"Error reading or processing chat history: {e}")
        return []



def find_answer_in_chat_history(question, chat_history):
    """
    Find an answer to the given question in the chat history.

    Parameters:
    - question (str): The user's question.
    - chat_history (List[Dict[str, str]]): The chat history.

    Returns:
    - str: A response based on the chat history.
    """
    # Simple search for a relevant response in the chat history
    for message in chat_history:
        if question.lower() in message['content'].lower():
            return f"Found something related: {message['content']}"
    return "Sorry, I couldn't find anything related in our chat history."

def get_chat_responses(question: str) -> List[str]:
    loader = WhatsAppChatLoader(
        path="_chat.txt",
    )
    raw_messages = loader.lazy_load()
    merged_messages = merge_chat_runs(raw_messages)
    messages: List[ChatSession] = list(
        map_ai_messages(merged_messages, sender="Prince Choudhary")
    )

    messages[0]["messages"].append({"role": "user", "content": question, "timestamp": "now"})

    responses = []

    for chunk in llm.stream(messages[0]["messages"]):
        responses.append(chunk)  # Directly append the response

    return responses
