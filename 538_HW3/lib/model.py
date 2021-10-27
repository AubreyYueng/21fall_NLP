# inbuilt lib imports:
from typing import Dict
import math

# external libs
import torch
import torch.nn as nn
from scipy.stats import truncnorm

# project imports

def truncated_normal(size, stddev):
    threshold = 2 * stddev
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values

class CubicActivation(nn.Module):
    """
    Cubic activation as described in the paper.
    """
    def __init__(self) -> None:
        super(CubicActivation, self).__init__()

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        vector : ``torch.Tensor``
            hidden vector of dimension (batch_size, hidden_dim)

        Returns tensor after applying the activation
        """
        # TODO(Students) Start
        return torch.pow(vector, 3)
        # TODO(Students) End


class DependencyParser(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 vocab_size: int,
                 num_tokens: int,
                 hidden_dim: int,
                 num_transitions: int,
                 regularization_lambda: float,
                 trainable_embeddings: bool,
                 device: str,
                 activation_name: str = "cubic") -> None:
        """
        This model defines a transition-based dependency parser which makes
        use of a classifier powered by a neural network. The neural network
        accepts distributed representation inputs: dense, continuous
        representations of words, their part of speech tags, and the labels
        which connect words in a partial dependency parse.

        This is an implementation of the method described in

        Danqi Chen and Christopher Manning.
        A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

        Parameters
        ----------
        embedding_dim : ``str``
            Dimension of word embeddings
        vocab_size : ``int``
            Number of words in the vocabulary.
        num_tokens : ``int``
            Number of tokens (words/pos) to be used for features
            for this configuration.
        hidden_dim : ``int``
        num_transitions : ``int``
            Number of transitions to choose from.
            Hidden dimension of feedforward network
        regularization_lambda : ``float``
            Regularization loss fraction lambda as given in paper.
        trainable_embeddings : `bool`
            Is the embedding matrix trainable or not.
        """
        super(DependencyParser, self).__init__()
        self._regularization_lambda = regularization_lambda

        if activation_name == "cubic":
            self._activation = CubicActivation()
        elif activation_name == "sigmoid":
            self._activation = torch.sigmoid
        elif activation_name == "tanh":
            self._activation = torch.tanh
        else:
            raise Exception(f"activation_name: {activation_name} is from the known list.")

        # Trainable Variables
        # TODO(Students) Start
        self._device = device
        self._trainable = trainable_embeddings

        def init_embedding(num_embeddings):
            emb = nn.Embedding(num_embeddings, embedding_dim)
            emb.weight.data.normal_(mean=0, std=1 / math.sqrt(embedding_dim))
            emb.weight.requires_grad = trainable_embeddings
            return emb

        # embedding + linear transformation
        self._word_embedding = nn.Sequential(init_embedding(vocab_size), nn.Linear(18, hidden_dim)).to(device)
        self._pos_embedding = nn.Sequential(init_embedding(num_tokens), nn.Linear(18, hidden_dim)).to(device)
        self._label_embedding = nn.Sequential(init_embedding(num_transitions), nn.Linear(12, hidden_dim)).to(device)
        # output
        self._output = nn.Sequential(
            nn.Linear(hidden_dim, num_transitions, bias=False),
            nn.Sigmoid()).to(device)
        # TODO(Students) End

    def forward(self,
             inputs: torch.Tensor,
             labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Dependency Parser.

        Parameters
        ----------
        inputs : ``torch.Tensor``
            Tensorized version of the batched input text. It is of shape:
            (batch_size, num_tokens) and entries are indices of tokens
            in to the vocabulary. These tokens can be word or pos tag.
            Each row corresponds to input features a configuration.
        labels : ``torch.Tensor``
            Tensor of shape (batch_size, num_transitions)
            Each row corresponds to the correct transition that
            should be made in the given configuration.

        Returns
        -------
        An output dictionary consisting of:
        logits : ``torch.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.
        loss : ``torch.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start
        logits_list = []
        for row in inputs:
            word_emb = self._word_embedding(row[:, :18])
            pos_emb = self._pos_embedding(row[:, 18:36])
            label_emb = self._label_embedding(row[:, 36:])
            hidden = self._activation(word_emb + pos_emb + label_emb)
            logits_list.append(self._output(hidden))
        logits = torch.stack(logits_list)
        # TODO(Students) End
        output_dict = {"logits": logits}

        if labels is not None:
            output_dict["loss"] = self.compute_loss(logits, labels)
        return output_dict

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.float32:
        """
        Parameters
        ----------
        logits : ``torch.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.

        Returns
        -------
        loss : ``torch.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start
        l2_reg = torch.tensor(0.)
        if labels is not None:
            for cur_label in labels:
                l2_reg += torch.norm(cur_label, p=2) / 2
            l2_reg *= self._regularization_lambda

        entropy_losses = []
        for cur_logit in logits:
            entropy_losses.append(-torch.log(cur_logit).sum())
        cross_entropy_loss = sum(entropy_losses) / len(entropy_losses)

        loss = cross_entropy_loss + l2_reg
        # TODO(Students) End
        return loss
