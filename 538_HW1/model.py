"""
author-gh: @adithya8
editor-gh: ykl7
"""

import math 

import numpy as np
import torch
import torch.nn as nn

sigmoid = lambda x: 1/(1 + torch.exp(-x))

class WordVec(nn.Module):
    def __init__(self, V, embedding_dim, loss_func, counts):
        super(WordVec, self).__init__()
        self.center_embeddings = nn.Embedding(num_embeddings=V, embedding_dim=embedding_dim)
        self.center_embeddings.weight.data.normal_(mean=0, std=1/math.sqrt(embedding_dim))
        self.center_embeddings.weight.data[self.center_embeddings.weight.data<-1] = -1
        self.center_embeddings.weight.data[self.center_embeddings.weight.data>1] = 1

        self.context_embeddings = nn.Embedding(num_embeddings=V, embedding_dim=embedding_dim)
        self.context_embeddings.weight.data.normal_(mean=0, std=1/math.sqrt(embedding_dim))
        self.context_embeddings.weight.data[self.context_embeddings.weight.data<-1] = -1 + 1e-10
        self.context_embeddings.weight.data[self.context_embeddings.weight.data>1] = 1 - 1e-10
        
        self.loss_func = loss_func
        self.counts = counts

    def forward(self, center_word, context_word):

        if self.loss_func == "nll":
            return self.negative_log_likelihood_loss(center_word, context_word)
        elif self.loss_func == "neg":
            return self.negative_sampling(center_word, context_word)
        else:
            raise Exception("No implementation found for %s"%(self.loss_func))
    
    def negative_log_likelihood_loss(self, center_word, context_word):
        ### TODO(students): start
        # TODO: Do we need to exclude 'UNK'
        mul = center_word.mul(context_word)                     # u_o^T v_c
        log_sum_exp = torch.log(torch.exp(mul).sum())           # log \sum{ exp(u_o^T v_c) }
        loss = torch.sum(log_sum_exp.subtract(mul))             # \sum {log_sum_exp - u_o^T v_c}
        ### TODO(students): end

        return loss
    
    def negative_sampling(self, center_word, context_word):
        ### TODO(students): start
        corpus_indices = torch.nonzero(context_word > 0)    # indices of corpus word
        unk_indices = torch.nonzero(context_word == 0)      # indices of negative word

        corpus_u, corpus_v = center_word[corpus_indices], context_word[corpus_indices]
        unk_x, unk_y = center_word[unk_indices], context_word[unk_indices]

        unk_exp = torch.exp(unk_x.mul(unk_y))
        unk_prob = unk_exp.divide(unk_exp.sum())            # Pr(unk words) = exp(unk)/\sum{exp(unk)}
        unk_neg = unk_prob.subtract(torch.tensor(1))        # -(1-Pr(unk words))

        unk_loss = torch.log(unk_neg).sum()                 # \sum{ log(Pr(unk)-1) }
        corpus_loss = self.negative_log_likelihood_loss(corpus_u, corpus_v)
        loss = corpus_loss + unk_loss

        ### TODO(students): end

        return loss

    def print_closest(self, validation_words, reverse_dictionary, top_k=8):
        print('Printing closest words')
        embeddings = torch.zeros(self.center_embeddings.weight.shape).copy_(self.center_embeddings.weight)
        embeddings = embeddings.data.cpu().numpy()

        validation_ids = validation_words
        norm = np.sqrt(np.sum(np.square(embeddings),axis=1,keepdims=True))
        normalized_embeddings = embeddings/norm
        validation_embeddings = normalized_embeddings[validation_ids]
        similarity = np.matmul(validation_embeddings, normalized_embeddings.T)
        for i in range(len(validation_ids)):
            word = reverse_dictionary[validation_words[i]]
            nearest = (-similarity[i, :]).argsort()[1:top_k+1]
            print(word, [reverse_dictionary[nearest[k]] for k in range(top_k)])            