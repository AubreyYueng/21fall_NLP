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
        mul = context_word.mul(center_word)                     # u_o^T v_c
        log_sum_exp = torch.log(torch.exp(mul).sum())           # log \sum{ exp(u_o^T v_c) }
        loss = torch.sum(log_sum_exp.subtract(mul))             # \sum {log_sum_exp - u_o^T v_c}
        ### TODO(students): end

        return loss
    
    def negative_sampling(self, center_word, context_word):
        ### TODO(students): start
        positive_sample = []        # construct positive samples for negative sample checking
        center_arr = center_word.numpy()
        context_arr = context_word.numpy()
        for i, x in enumerate(center_arr):
            positive_sample.append((x, context_arr[i]))

        word_freq = {}
        for x in context_word:  # counting words
            if x in word_freq:
                word_freq[x] += 1
            else:
                word_freq[x] = 0
        freq_sum = 0
        for (k, v) in word_freq.items():
            word_freq[k] = v ** (3 / 4)  # adjust count according to the paper
            freq_sum += word_freq[k]
        for (k, v) in word_freq.items():  # calculate adjusted frequencies
            word_freq[k] = v / freq_sum

        sample_size = len(center_word)
        center_idx = 0
        neg_context = []
        while center_idx < sample_size:
            center = center_arr[center_idx]     # use the same center word
            random_context = np.random.choice(list(word_freq.keys()), p=list(word_freq.values()))
            if (center, random_context) in positive_sample:
                continue
            neg_context.append(random_context)
            center_idx += 1

        neg_u = torch.LongTensor(np.array(neg_context, dtype=np.int32))
        neg_v = torch.LongTensor(np.array(center_arr, dtype=np.int32))

        exp_pos = torch.exp(torch.neg(context_word.mul(center_word)))   # exp(-u_o^T v_c) of positive data
        exp_neg = torch.exp(neg_u.mul(neg_v))                           # exp(u_o^T v_c) of neg data

        denominator_pos = torch.tensor(1).add(exp_pos)      # 1 + exp(-u_o^T v_c) of positive data
        denominator_neg = torch.tensor(1).add(exp_neg)      # 1 + exp(u_o^T v_c) of neg data

        # sum of log sigmoid
        ll_pos = torch.log(torch.tensor(1).divide(denominator_pos)).sum()
        ll_neg = torch.log(torch.tensor(1).divide(denominator_neg)).sum()

        loss = -(ll_pos + ll_neg)
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