

import os
os.environ['HF_HOME'] = '/scratch/marcelolocatelli/cache/'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/marcelolocatelli/cache/'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from nltk.tokenize import word_tokenize,RegexpTokenizer
from nltk.stem import PorterStemmer
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import Sampler
import random
import pickle

from datasets import load_dataset

SEED=42

ds = load_dataset("wikimedia/wikipedia", "20231101.en",split='train',streaming=False)
rng = np.random.default_rng(seed=SEED)
idxs = rng.choice(ds.num_rows, size=100000, replace=False)
ds = ds.select(idxs)
torch.manual_seed(SEED)
random.seed(SEED)

class W2VDataset:
    def __init__(self, data_source, window_size, batch_size=10000):
        """
        Custom batch sampler for skip-gram pairs.

        Args:
            data_source: Dataset object (assumes line-based text data).
            window_size: Size of the context window for generating skip-gram pairs.
            batch_size: Number of skip-gram pairs per batch.
        """
        self.data_source = data_source
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_lines = len(data_source)
#         self.len = sum(
#             len(self.generate_skipgram_pairs(self.data_source[i], self.window_size))
#             for i in range(self.num_lines)
#         )
        self.len = len(self.generate_skipgram_pairs(self.data_source[0], self.window_size))*self.num_lines
        
    def generate_skipgram_pairs(self,line,window_size=10):
        """Generate skip-gram pairs from a given line of text."""
        pairs = []
        for center_idx, center_word in enumerate(line):
            start = max(0, center_idx - window_size)
            end = min(len(line), center_idx + window_size + 1)
            for context_idx in range(start, end):
                if center_idx != context_idx:
                    pairs.append((center_word, line[context_idx]))
        return pairs

    def __iter__(self):
        remaining_lines = list(range(self.num_lines))
        batch = []
        
        while remaining_lines:
            # Randomly pick a line index without replacement
            line_idx = random.choice(remaining_lines)
            remaining_lines.remove(line_idx)
            
            # Fetch line from data source
            line = self.data_source[line_idx]
            
            # Generate skip-gram pairs for the line
            pairs = self.generate_skipgram_pairs(line, self.window_size)
            
            # Add to the current batch
            batch.extend(pairs)
            
            # Yield batch if it reaches the desired size
            if len(batch) >= self.batch_size:
                yield torch.tensor(batch[:self.batch_size],dtype=torch.long)
                batch = batch[self.batch_size:]

        # Yield the remaining pairs if any
        if batch:
            yield torch.tensor(batch,dtype=torch.long)

    def __len__(self):
        """Estimate the number of batches."""
        return (self.len + self.batch_size - 1) // self.batch_size

def build_corpus(data):
    word_to_idx={'[UNK]':0}
    word_counts={}
    corpus = []
    stemmer = PorterStemmer()
    tokenizer = RegexpTokenizer(r'[\w-]+')
    for x in tqdm(data):
        x = x['text']
        line=[]
        for word in tokenizer.tokenize(x.lower()):
            word = stemmer.stem(word)
            word_to_idx[word] = word_to_idx.get(word,len(word_to_idx))
            word_counts[word] = word_counts.get(word,0)+1
            line.append(word_to_idx[word])
        corpus.append(torch.tensor(line,dtype=torch.long))
    idx_to_word = {v:k for k,v in word_to_idx.items()}
    return word_to_idx,idx_to_word,word_counts,corpus

def corpus_to_dataset(corpus,window_size=2,cbow=True):
    X = []
    y = []
    for line in tqdm(corpus):
        for i in range(window_size,len(line)-window_size):
            if cbow:
                label=line[i]
                x = line[i-window_size:i]+line[i+1:i+1+window_size]
                X.append(x)
                y.append(label)
            else:
                label = line[i-window_size:i]+line[i+1:i+1+window_size]
                x = line[i]
                X.extend([x]*2*window_size)
                y.extend(label)
    return X,y

def get_negative_samples(target, num_negative_samples, vocab_size):
    neg_samples = []
    while len(neg_samples) < num_negative_samples:
        neg_sample = np.random.randint(0, vocab_size)
        if neg_sample != target:
            neg_samples.append(neg_sample)
    return neg_samples


w2idx,idx2w,wc,corpus=build_corpus(ds)

with open('/scratch/marcelolocatelli/w2idx.pkl', 'wb') as f:
    pickle.dump(w2idx, f)

with open('/scratch/marcelolocatelli/idx2w.pkl', 'wb') as f:
    pickle.dump(idx2w, f)

with open('/scratch/marcelolocatelli/wc.pkl', 'wb') as f:
    pickle.dump(wc, f)


#corpus = TensorDataset(corpus)
dataset = W2VDataset(data_source=corpus,batch_size=10000,window_size=10)

class neg_skipgram(nn.Module):
    def __init__(self,vocab_size,embedding_dimension=100):
        super(neg_skipgram,self).__init__()
        self.embeddings=nn.Embedding(vocab_size, embedding_dimension)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dimension)
        init_range = 0.5 / embedding_dimension
        nn.init.uniform_(self.embeddings.weight.data, -init_range, init_range)
        nn.init.constant_(self.context_embeddings.weight.data, 0)

    def forward(self, target, context, negative_samples):
        target_embedding = self.embeddings(target)
        context_embedding = self.context_embeddings(context)
        negative_embeddings = self.context_embeddings(negative_samples)
        
        positive_score = F.logsigmoid(torch.sum(target_embedding * context_embedding, dim=1))
        negative_score = F.logsigmoid(-torch.bmm(negative_embeddings, target_embedding.unsqueeze(2)).squeeze(2)).sum(1)
        
        loss = - (positive_score + negative_score).mean()
        return loss


# In[14]:


model = neg_skipgram(len(w2idx),300).to("cuda")


num_negative_samples=5
losses = []
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.025)

for epoch in range(10):
    total_loss = []
    for inputs in tqdm(dataset):
        target = inputs[:,1]
        context = inputs[:,0]
        negative_samples = torch.tensor([get_negative_samples(t,num_negative_samples,len(w2idx)) for t in target],dtype=torch.long)
        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        context_idxs = context

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        loss = model(target.to("cuda"),context_idxs.to("cuda"),negative_samples.to("cuda"))

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
        #loss = loss_function(log_probs, target)

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss.append(loss.item())
    losses.append(total_loss)
    torch.save(model.state_dict(), f'/scratch/marcelolocatelli/{epoch}_w2v.pt')
    torch.save(model.state_dict(), f'/home_cerberus/speed/locatellimarcelo/w2v/w2v.pt')
with open('/scratch/marcelolocatelli/losses.pkl', 'wb') as f:
    pickle.dump(losses, f)



# In[58]:



def most_similar(target_word, model, word_to_idx, idx_to_word, top_k=10):
    """
    Find the most similar words to the target word.

    Args:
        target_word (str): The word for which to find similar words.
        model (torch.nn.Module): The trained skip-gram model.
        word_to_idx (dict): Mapping from words to indices.
        idx_to_word (dict): Mapping from indices to words.
        top_k (int): Number of most similar words to return.

    Returns:
        List of tuples: Most similar words and their similarity scores.
    """
    # Get the embedding matrix
    embedding_matrix = model.embeddings.weight  # (vocab_size, embedding_dim)

    # Get the embedding of the target word
    target_idx = word_to_idx[target_word]
    target_embedding = embedding_matrix[target_idx]  # (embedding_dim,)

    # Compute cosine similarity between the target word and all other words
    similarities = F.cosine_similarity(
        target_embedding.unsqueeze(0),  # Add batch dimension
        embedding_matrix,               # Compare to all embeddings
        dim=1                           # Across the embedding dimension
    )

    # Get the top-k most similar words (excluding the target word itself)
    similar_indices = torch.topk(similarities, top_k + 1).indices.tolist()
    similar_words = [
        (idx_to_word[idx], similarities[idx].item())
        for idx in similar_indices if idx != target_idx
    ]

    return similar_words[:top_k]


# In[ ]:




