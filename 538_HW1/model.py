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
        center_embeds = self.center_embeddings(center_word)
        context_embeds = self.context_embeddings(context_word)
        mul_1 = torch.sum(context_embeds * center_embeds, dim=1)            # u_{c-m+j}^T v_c
        mul_2 = center_embeds.mm(torch.transpose(context_embeds, 0, 1))     # u_k^T v_c
        exp = torch.exp(mul_1)                                  # exp(u_{c-m+j}^T v_c)
        sum_exp = torch.exp(mul_2).sum()                        # \sum exp(u_k^T v_c)
        loss = -(torch.log(exp.divide(sum_exp))).mean()         # -sum(log exp/sum_exp)
        ### TODO(students): end

        return loss

    def negative_sampling(self, center_word, context_word):
        ### TODO(students): start
        sample_size = len(center_word)
        weights = torch.tensor(self.counts, dtype=torch.float) ** 0.75
        neg_center= torch.multinomial(weights, sample_size, replacement = True)
        neg_context = torch.multinomial(weights, sample_size, replacement=True)

        positive_sample = []                # construct positive samples for negative sample checking
        center_arr = center_word.numpy()
        context_arr = context_word.numpy()
        for i, x in enumerate(center_arr):
            positive_sample.append((x, context_arr[i]))

        cur_size = 0
        neg_center_arr = []
        neg_context_arr = []
        while cur_size < sample_size:
            neg_center = neg_center[cur_size]                   # negative center word
            neg_context = neg_context[cur_size]                 # negative context word
            if (neg_center, neg_context) in positive_sample:    # chosen sample exists, random choose another sample
                neg_center[cur_size] = torch.multinomial(weights, 1, replacement=True)
                neg_context[cur_size] = torch.multinomial(weights, 1, replacement=True)
                continue
            neg_center_arr.append(neg_center)
            neg_context_arr.append(neg_context)
            cur_size += 1

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        neg_center = torch.LongTensor(np.array(center_arr, dtype=np.int32)).to(device)
        neg_context = torch.LongTensor(np.array(neg_context_arr, dtype=np.int32)).to(device)

        # calculate loss for negative
        neg_center_embeds = self.center_embeddings(neg_center)
        neg_context_embeds = self.context_embeddings(neg_context)
        neg_mul = torch.sum(neg_context_embeds * neg_center_embeds, dim=-1)
        neg_loss = torch.log(sigmoid(-neg_mul))

        # calculate loss for positive
        center_embeds = self.center_embeddings(center_word)
        context_embeds = self.context_embeddings(context_word)
        mul = torch.sum(context_embeds * center_embeds, dim = -1)
        pos_loss = torch.log(sigmoid(mul))

        loss = -(pos_loss + neg_loss).mean()
        ### TODO(students): end

        return loss

    def print_closest(self, validation_words, reverse_dictionary, top_k=8):
        print('Printing closest words')
        embeddings = torch.zeros(self.center_embeddings.weight.shape).copy_(self.center_embeddings.weight)
        embeddings = embeddings.data.cpu().numpy()

        validation_ids = validation_words
        norm = np.sqrt(np.sum(np.square(embeddings), axis=1, keepdims=True))
        normalized_embeddings = embeddings / norm
        validation_embeddings = normalized_embeddings[validation_ids]
        similarity = np.matmul(validation_embeddings, normalized_embeddings.T)
        for i in range(len(validation_ids)):
            word = reverse_dictionary[validation_words[i]]
            nearest = (-similarity[i, :]).argsort()[1:top_k + 1]
            print(word, [reverse_dictionary[nearest[k]] for k in range(top_k)])
