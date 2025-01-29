import torch
from torch import nn
import torch.nn.functional as F
import pickle

class neg_skipgram(nn.Module):
    def __init__(self,
                 vocab_size,
                 w2idx_path,
                 embedding_dimension=300,
                 regularization=0.01,
                 ):
        
        super(neg_skipgram,self).__init__()
        self.embeddings=nn.Embedding(vocab_size, embedding_dimension)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dimension)
        self.reg = regularization
        init_range = 0.5 / embedding_dimension
        nn.init.uniform_(self.embeddings.weight.data, -init_range, init_range)
        nn.init.constant_(self.context_embeddings.weight.data, 0)
        
        self.w2idx = pickle.load(open(w2idx_path, "rb"))
        

    def forward(self, target, context, negative_samples):
        target_embedding = self.embeddings(target)
        context_embedding = self.context_embeddings(context)
        negative_embeddings = self.context_embeddings(negative_samples)
        
        positive_score = F.logsigmoid(torch.sum(target_embedding * context_embedding, dim=1))
        negative_score = F.logsigmoid(-torch.bmm(negative_embeddings, target_embedding.unsqueeze(2)).squeeze(2)).sum(1)
        
        #regularization_term = torch.square(torch.norm(target_embedding,dim=1) - idx2kl[target.to('cpu')].to('cuda')).mean()
        #regularization_term2 = torch.square(torch.norm(negative_embeddings,dim=2) - idx2kl[negative_samples.to('cpu')].to('cuda')).mean() + torch.square(torch.norm(context_embedding,dim=1) - idx2kl[context.to('cpu')].to('cuda')).mean()
        loss = - (positive_score + negative_score).mean() #+ self.reg*(regularization_term+regularization_term2)
        return loss
    
    def embed(self, word: str):
        idx = self.w2idx.get(word)
        if idx is None:
            raise KeyError('Word not in vocabulary')
        
        return self.embeddings.weight.data[idx]
    
    def get_similar(self, vector, n):
        vector = vector / vector.norm(p=2)

        embeddings = self.embeddings.weight.data
        normalized_embeddings = embeddings / embeddings.norm(p=2, dim=1, keepdim=True)

        similarities = torch.matmul(normalized_embeddings, vector)

        top_n_indices = torch.topk(similarities, n).indices

        idx2word = {idx: word for word, idx in self.w2idx.items()}
        top_n_words = [(idx2word[idx.item()], similarities[idx].item()) for idx in top_n_indices]

        return top_n_words[1:]

    def __contains__(self, word):
        return word in self.w2idx