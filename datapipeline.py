import torch
from torch.utils.data import Dataset, random_split, DataLoader
import json
from tokenizer import get_tokenizer

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer = tokenizer


        self.sos_token = torch.tensor([tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['input']
        tgt_text = src_target_pair['output']

        input_tokens = self.tokenizer.encode(src_text).ids
        output_tokens = self.tokenizer.encode(tgt_text).ids

        inp_num_padding_tokens = self.seq_len - len(input_tokens) - 2
        lbl_num_padding_tokens = self.seq_len - len(output_tokens) - 2

        if inp_num_padding_tokens < 0 or lbl_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        inputs = torch.cat(
            [
                self.sos_token,
                torch.tensor(input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * inp_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        labels = torch.cat(
            [
                self.sos_token,
                torch.tensor(output_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * lbl_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        assert inputs.size(0) == self.seq_len
        assert labels.size(0) == self.seq_len

        return {
            "input_ids": inputs,
            "labels": labels
        }

def get_ds(config):
    with open(config['dataset_file_path'], 'r', encoding='utf-8') as f:
        ds_raw = json.load(f)

    tokenizer = get_tokenizer(config, ds_raw)

    train_ds_size = int(0.99 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])


    train_ds = BilingualDataset(train_ds_raw, tokenizer, config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer, config['seq_len'])

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer