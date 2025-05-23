from transformers import pipeline
import argparse

def get_response(prompt, model_name):
    """
    Run the model locally using transformers.pipeline

    Args:
        prompt: The input text to send to the model
        model_name: Name of the Hugging Face model to load

    Returns:
        The generated text from the model, or error message
    """
    try:
        # Force use of PyTorch framework to avoid TF model loading errors
        chat = pipeline("text2text-generation", model=model_name, framework="pt")
        response = chat(prompt, max_length=100, do_sample=False)
        return response[0]["generated_text"]
    except Exception as e:
        return f"Error: {str(e)}"

def run_chat(model_name):
    """
    Run an interactive chat session
    """
    print("Local LLM Chat (no API). Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        response = get_response(user_input, model_name=model_name)
        print(f"Bot: {response}")

def main():
    parser = argparse.ArgumentParser(description="Chat with a local LLM")
    
    # Add model name argument
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/flan-t5-small",
        help="Name of the Hugging Face model to use locally"
    )

    args = parser.parse_args()
    run_chat(model_name=args.model_name)

if __name__ == "__main__":
    main()