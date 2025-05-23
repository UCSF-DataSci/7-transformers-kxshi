import requests
import argparse
import os

API_TOKEN = "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"
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

def get_response(prompt, model_name="HuggingFaceH4/zephyr-7b-beta", api_key=API_TOKEN):
    """
    Get a response from the model
    
    Args:
        prompt: The prompt to send to the model
        model_name: Name of the model to use
        api_key: API key for authentication (optional for some models)
        
    Returns:
        The model's response
    """
    # Set up the API URL and headers
    api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {
    "Authorization": f"Bearer {api_key}"
    }
    # Create a payload with the prompt
    payload = {"inputs": prompt}
    # Send the payload to the API
    # Extract and return the generated text from the response
    # Handle any errors that might occur
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

def run_chat(model_name, api_key):
    """Run an interactive chat session"""
    print("Welcome to the Simple LLM Chat! Type 'exit' to quit.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
            
        response = get_response(user_input, model_name=model_name, api_key=api_key)
        print(f"Response: {response}")
        
def main():
    parser = argparse.ArgumentParser(description="Chat with an LLM")
    
    parser.add_argument("--model_name", type=str, default="HuggingFaceH4/zephyr-7b-beta", help="Name of the model to use")
    parser.add_argument("--api_key", type=str, default=API_TOKEN, help="API key for authentication")

    args = parser.parse_args()
    
    run_chat(model_name=args.model_name, api_key=args.api_key)
    
if __name__ == "__main__":
    main()