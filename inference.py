import torch
from config import get_config
from loadmodel import load_model
from datapipeline import get_ds
from model import TCLM

# Load configuration, tokenizer, and model
config = get_config()
_, _, tokenizer = get_ds(config)

model = TCLM(vocab_size=tokenizer.get_vocab_size(),
             seq_len=config['seq_len'],
             d_model=config['d_model'],
             N=config['n_layers'],
             h=config['head'],
             dropout=config['dropout'],
             d_ff=config['d_ff'])

optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
model, _, _, _ = load_model(config, config['device'], model, optimizer)
 
# Inference loop
while True:
    user_text = input("User: ")

    if user_text.lower() in ["exit", ""]:
        break

    # Add [SOS], [USER], and [BOT] tokens to input text
    input_text = f"[SOS] [USER] {user_text} [BOT]"

    # Tokenize the input text
    idx = tokenizer.encode(input_text).ids
    idx = torch.tensor([idx]).to(config['device'])

    # Generate the bot's response
    generated_sequence = model.generate(idx, max_new_tokens=100, seq_len=config['seq_len'], temperature=config['temperature'], top_k=config['top_k'])

    # Decode and print the generated text
    predicted_text = tokenizer.decode(generated_sequence[0].cpu().numpy())
    
    # Find the position of the [BOT] token and print tokens after it
    # bot_token_index = predicted_text.find(user_text[-1])
    # response_tokens = predicted_text[bot_token_index:].strip()

    print("Omnira:", predicted_text)