# utils/conversation.py

import requests
import argparse
import os

API_TOKEN = "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"
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

def get_response(prompt, history=None, model_name="HuggingFaceH4/zephyr-7b-beta", api_key=API_TOKEN, history_length=3):
    """
    Get a response from the model using conversation history
    
    Args:
        prompt: The current user prompt
        history: List of previous (prompt, response) tuples
        model_name: Name of the model to use
        api_key: API key for authentication
        history_length: Number of previous exchanges to include in context
        
    Returns:
        The model's response
    """
    # TODO: Implement the contextual response function
    # Initialize history if None
    if history is None:
        history = []
        
    # Just chuck everything together in the prompt
    context_prompt = ""
    for past_prompt, past_response in history[-history_length:]:
        context_prompt += f"User: {past_prompt}\nResponse: {past_response}\n"
    context_prompt += f"User: {prompt}\nResponse:"
    # Get a response from the API
    # Return the response
    api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": context_prompt}

    # Get a response from the API
    # Return the response
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        output = response.json()
        if isinstance(output, list) and 'generated_text' in output[0]:
            return output[0]['generated_text'].strip()
        elif isinstance(output, dict) and 'generated_text' in output:
            return output['generated_text'].strip()
        return str(output)
    except requests.exceptions.RequestException as e:
        return f"[ERROR] API request failed: {e}"

def run_chat(model_name, api_key, history_length):
    """Run an interactive chat session with context"""
    print("Welcome to the Contextual LLM Chat! Type 'exit' to quit.")
    
    # Initialize conversation history
    history = []
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
            
        # Get response using conversation history
        response = get_response(
            prompt=user_input,
            history=history,
            model_name=model_name,
            api_key=api_key,
            history_length=history_length
        )
        # Update history
        history.append((user_input, response))
        # Print the response
        print(f"LLM: {response}\n")
        
def main():
    parser = argparse.ArgumentParser(description="Chat with an LLM using conversation history")
    # TODO: Add arguments to the parser

    parser = argparse.ArgumentParser(description="Chat with an LLM using conversation history")
    parser.add_argument("--model_name", type=str, default="HuggingFaceH4/zephyr-7b-beta", help="Hugging Face model name")
    parser.add_argument("--api_key", type=str, default="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", help="Hugging Face API token")
    parser.add_argument("--history_length", type=int, default=3, help="Number of previous questions and answers to include in context")

    args = parser.parse_args()
    
    # TODO: Run the chat function with parsed arguments
    run_chat(model_name=args.model_name, api_key=args.api_key, history_length=args.history_length)
    
if __name__ == "__main__":
    main()