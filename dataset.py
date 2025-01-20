import numpy as np
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from datasets import load_dataset

def build_corpus(size):
    ds = load_dataset("wikimedia/wikipedia", "20231101.en",split='train',streaming=False)
    rng = np.random.default_rng(seed=42)
    idxs = rng.choice(ds.num_rows, size, replace=False)
    ds = ds.select(idxs)

    corpus = []
    tokenizer = RegexpTokenizer(r'\w+')
    for x in tqdm(ds):
        x = x['text']
        tokens = tokenizer.tokenize(x.lower())
        corpus.append(tokens)
    return corpus
