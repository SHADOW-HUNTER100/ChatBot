import os
from dotenv import load_dotenv
import chainlit as cl
import requests
import json
from typing import Dict, List, Any

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# Available models with descriptions
AVAILABLE_MODELS = {
    "mistralai/Mistral-7B-Instruct-v0.2": "Mistral 7B (Balanced)",
    "openchat/openchat-7b": "OpenChat 7B (General)",
    "neversleep/llama-3.1-lumimaid-70b": "Llama 3.1 Lumimaid 70B (Advanced)",
    "microsoft/phi-3-medium-128k-instruct": "Phi-3 Medium (Efficient)",
    "google/gemma-7b-it": "Gemma 7B (Lightweight)"
}

# Default model and system message
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
system_message = "You are a helpful AI assistant. Answer questions clearly and concisely. If you don't know the answer, say so."

# Headers for the API request
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
    "HTTP-Referer": "http://localhost:8000",  # Corrected spelling to 'Referer'
    "X-Title": "Agentic AI Chatbot"  # Required by OpenRouter
}

# Event: On chat start
@cl.on_chat_start
async def start():
    # Initialize conversation history with system message
    history = [
        {"role": "system", "content": system_message}
    ]
    cl.user_session.set("history", history)

    # Send welcome message
    await cl.Message(
        content="Welcome To ChatBot🔥\n\nI'm your AI assistant. How can I help you today?"
    ).send()

# Event: On message element (file upload and model switching)
@cl.on_message
async def on_message(message: cl.Message):
    # Get current settings from session
    history = cl.user_session.get("history") or [{"role": "system", "content": system_message}]
    current_model = cl.user_session.get("current_model", model_name)

    # Check if user wants to switch model (command format: "model: model_name")
    user_message = message.content.strip()
    if user_message.lower().startswith("model:"):
        requested_model = user_message[7:].strip()  # Get the model name after "model:"

        # Check if the requested model exists in our available models
        model_found = False
        for model_id, model_display_name in AVAILABLE_MODELS.items():
            if requested_model.lower() in model_id.lower() or requested_model.lower() in model_display_name.lower():
                cl.user_session.set("current_model", model_id)
                await cl.Message(content=f"Model changed to: {model_display_name} ({model_id})").send()
                model_found = True
                break

        if not model_found:
            available_models_list = ", ".join([f"{name} ({model_id})" for model_id, name in AVAILABLE_MODELS.items()])
            await cl.Message(content=f"Model '{requested_model}' not found. Available models: {available_models_list}").send()

        return  # Don't process the message as a regular chat message

    # Handle file uploads
    if message.elements:
        for element in message.elements:
            if element.mime.startswith("text/"):
                # Read text file content
                with open(element.path, "r", encoding="utf-8") as file:
                    file_content = file.read()

                # Add file content to the conversation
                file_msg = f"User uploaded a text file named '{element.name}'. Here's its content:\n\n{file_content}\n\nPlease analyze this content and respond appropriately."
                history.append({"role": "user", "content": file_msg})

                # Send confirmation message
                await cl.Message(content=f"File '{element.name}' uploaded and processed. I've analyzed its content and am ready to discuss it.").send()
            else:
                # For other file types, just acknowledge the upload
                await cl.Message(content=f"File '{element.name}' uploaded. (Note: I can only analyze text files directly)").send()

    # Add user message to history if it's not empty (for non-file messages)
    if message.content.strip() and not user_message.lower().startswith("model:"):
        history.append({"role": "user", "content": message.content})

    # Limit conversation history to prevent token overflow (keep system message + last 10 exchanges)
    if len(history) > 21:  # System message + 10 user/assistant pairs
        history = [history[0]] + history[-20:]  # Keep system message + last 20 messages
        cl.user_session.set("history", history)

    payload = {
        "model": current_model,
        "messages": history,  # Corrected key: "messages" not "message"
        "temperature": 0.7,
        "max_tokens": 1000
    }

    # Show a loading indicator while processing
    msg = cl.Message(content="⏳ Thinking...")
    await msg.send()

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",  # Corrected URL
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        res = response.json()

        bot_msg = res["choices"][0]["message"]["content"]
        history.append({"role": "assistant", "content": bot_msg})

        # Update the message with the actual response
        msg.content = bot_msg
        await msg.update()

    except requests.exceptions.HTTPError as e:
        if response is not None and response.status_code == 401:
            msg.content = "❌ Error: Unauthorized - Please check your API key in the .env file"
            await msg.update()
        elif response is not None and response.status_code == 403:
            msg.content = "❌ Error: Access forbidden - Your API key may not have access to this model"
            await msg.update()
        elif response is not None and response.status_code == 429:
            msg.content = "❌ Error: Rate limit exceeded - Too many requests"
            await msg.update()
        else:
            msg.content = f"❌ HTTP Error: {str(e)}"
            await msg.update()
    except requests.exceptions.RequestException as e:
        msg.content = f"❌ Request error: {str(e)}"
        await msg.update()
    except Exception as e:
        msg.content = f"❌ An error occurred: {str(e)}"
        await msg.update()