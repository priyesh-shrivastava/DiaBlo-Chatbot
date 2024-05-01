from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pre-trained model tokenizer (vocabulary)
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')

# Load pre-trained model (weights)
model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')

# Function to handle chatbot response
def chatbot_response(text, chat_history_ids):
    # Tokenize new input sentence
    new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')

    # Append tokens to chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids else new_user_input_ids

    # Generate response given the user input
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    return response, chat_history_ids

# Example conversation
chat_history_ids = None
user_input = "Hello, who are you?"
response, chat_history_ids = chatbot_response(user_input, chat_history_ids)
print("Bot: " + response)

# Continue the conversation
user_input = "What's your purpose?"
response, chat_history_ids = chatbot_response(user_input, chat_history_ids)
print("Bot: " + response)
