import numpy as np
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from datasets import load_dataset
from nltk.stem import PorterStemmer
import random
import pickle
import os

PICKLE_FILEPATH = 'data/corpus/{size_corpus}-corpus.pkl'

def save_corpus(size_corpus, corpus, word_to_idx, idx_to_word, word_count,):
    os.makedirs('data/corpus/', exist_ok=True)
    filepath = PICKLE_FILEPATH.format(size_corpus=size_corpus)
    with open(filepath, "wb") as f:
        pickle.dump({
            "corpus": corpus,
            "word_to_idx": word_to_idx,
            "idx_to_word": idx_to_word,
            "word_count": word_count,
        }, f)

def load_corpus(size_corpus):
    filepath = PICKLE_FILEPATH.format(size_corpus=size_corpus)
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data["corpus"], data["word_to_idx"], data["idx_to_word"], data['word_count']

def build_corpus(size, 
                 return_fields=['corpus'],
                 load=True,
                 save=False, 
                 seed=42):
    
    pickle_file = PICKLE_FILEPATH.format(size_corpus=size)
    if os.path.exists(pickle_file) and load and not save:
        print('-'*40)
        print('LOADING corpus')
        print('-'*40)
        corpus, word2idx, idx2word, word_count = load_corpus(size)
    
    else:
        if not os.path.exists(pickle_file):
            print(f'!!! Filepath {pickle_file} does not exist!')
            
        print('-'*40)
        print('SAVING corpus')
        print('-'*40)
        random.seed(seed)
        ds = load_dataset("wikimedia/wikipedia", "20231101.en",split='train',streaming=False)
        rng = np.random.default_rng(seed=seed)
        idxs = rng.choice(ds.num_rows, size, replace=False)
        ds = ds.select(idxs)
        
        word_to_idx={'[UNK]':0}
        word_counts={}
        corpus = []
        stemmer = PorterStemmer()
        tokenizer = RegexpTokenizer(r'[\w-]+')
        for x in tqdm(ds):
            x = x['text']
            line=[]
            for word in tokenizer.tokenize(x.lower()):
                word = stemmer.stem(word)
                word_to_idx[word] = word_to_idx.get(word,len(word_to_idx))
                word_counts[word] = word_counts.get(word,0)+1
                line.append(word)
            corpus.append(line)
        idx_to_word = {v:k for k,v in word_to_idx.items()}
        
        save_corpus(size, corpus, word_to_idx, idx_to_word, word_counts)
        
    
    return_dict = {}
    
    if 'corpus' in return_fields:
        return_dict['corpus'] = corpus
    
    if 'word2idx' in return_fields:
        return_dict['word2idx'] = word2idx
        
    if 'idx2word' in return_fields:
        return_dict['idx2word'] = idx2word
        
    if 'word_count' in return_fields:
        return_dict['word_count'] = word_count
        
    return return_dict