from pathlib import Path
from tokenizers import Tokenizer

def get_tokenizer(config, ds):
    tokenizer_path = Path(config['tokenizer_file_path'])
    if tokenizer_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        print("tokenizer file not found!")
    return tokenizer