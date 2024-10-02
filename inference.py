import torch
from config import get_config
from loadmodel import load_model
from tokenizer import get_tokenizer
from model import TCLM

config = get_config()
tokenizer = get_tokenizer(config)

user_token_id = tokenizer.token_to_id('[USER]')
bot_token_id = tokenizer.token_to_id('[BOT]')

# Initialize the model
model = TCLM(
    vocab_size=tokenizer.get_vocab_size(),
    seq_len=config['seq_len'],
    d_model=config['d_model'],
    N=config['n_layers'],
    h=config['head'],
    dropout=config['dropout'],
    d_ff=config['d_ff']
)

optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
model, _, _, _ = load_model(config, config['device'], model, optimizer)

conversation_history = []

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit" or user_input == "":
        break

    conversation_history.append(user_input)
    input_sequence = [user_token_id]
    user_input_ids = tokenizer.encode(user_input).ids
    input_sequence += user_input_ids

    # If conversation history exceeds a certain length, truncate it
    if len(conversation_history) > 5:  # Adjust the number based on context length needed
        conversation_history = conversation_history[-5:]

    # Append previous turns to the input sequence
    for i in range(len(conversation_history)):
        if i % 2 == 0:  # User turn
            input_sequence += [user_token_id] + tokenizer.encode(conversation_history[i]).ids
        else:  # Bot turn
            input_sequence += [bot_token_id] + tokenizer.encode(conversation_history[i]).ids

    # Convert to tensor and send to device
    try:
        input_tensor = torch.tensor([input_sequence]).to(config['device'])
    except Exception as e:
        print(f"Error while converting to tensor: {e}")
        continue

    # Generate response from the model
    generated_sequence = model.generate(
        input_tensor,
        max_new_tokens=100,
        seq_len=config['seq_len'],
        temperature=config['temperature'],
        top_k=config['top_k']
    )

    predicted_text = tokenizer.decode(generated_sequence[0].cpu().numpy())
    conversation_history.append(predicted_text)
    print(conversation_history)

    print("Omnira:", predicted_text)
