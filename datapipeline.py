import torch
from torch.utils.data import Dataset, random_split, DataLoader
import json
from tokenizer import get_tokenizer

class MultiTurnChatDataset(Dataset):

    def __init__(self, ds, tokenizer, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer = tokenizer
        self.pad_token = torch.tensor([tokenizer.token_to_id("[PAD]")], dtype=torch.int64)
        self.user_token = torch.tensor([tokenizer.token_to_id("[USER]")], dtype=torch.int64)
        self.bot_token = torch.tensor([tokenizer.token_to_id("[BOT]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        conversation = self.ds[idx]['conversation']

        # Concatenate multi-turn dialogue, adding <USER> and <BOT> tokens for each turn
        conversation_tokens = []
        for i, turn in enumerate(conversation):
            if i % 2 == 0:  # Even index: User turn
                conversation_tokens += [self.user_token] + self.tokenizer.encode(turn).ids
            else:  # Odd index: Bot turn
                conversation_tokens += [self.bot_token] + self.tokenizer.encode(turn).ids

        # Ensure the total length doesn't exceed the sequence length
        num_padding_tokens = self.seq_len - len(conversation_tokens)
        if num_padding_tokens < 0:
            raise ValueError("Multi-turn conversation is too long for the sequence length")

        # Pad the conversation tokens
        input_tokens = conversation_tokens + [self.pad_token] * num_padding_tokens

        # Convert to torch tensors
        inputs = torch.tensor(input_tokens, dtype=torch.int64)

        assert inputs.size(0) == self.seq_len

        # The model's task is to predict the next bot response, so the labels are shifted
        labels = torch.tensor(input_tokens[1:] + [self.pad_token], dtype=torch.int64)  # Shifted by 1 for language modeling

        return {
            "input_ids": inputs,
            "labels": labels,
        }

def get_ds(config):
    with open(config['dataset_file_path'], 'r', encoding='utf-8') as f:
        ds_raw = json.load(f)

    tokenizer = get_tokenizer(config)

    # Split the dataset into train and validation sets
    train_ds_size = int(0.99 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # Create datasets and dataloaders
    train_ds = MultiTurnChatDataset(train_ds_raw, tokenizer, config['seq_len'])
    val_ds = MultiTurnChatDataset(val_ds_raw, tokenizer, config['seq_len'])

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer