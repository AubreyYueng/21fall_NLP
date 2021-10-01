'''
author: Sounak Mondal
'''

# std lib imports
from typing import Dict

# external libs
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

class SequenceToVector(nn.Module):
    """
    It is an abstract class defining SequenceToVector enocoder
    abstraction. To build you own SequenceToVector encoder, subclass
    this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    """
    def __init__(self,
                 input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim

    def forward(self,
             vector_sequence: torch.Tensor,
             sequence_mask: torch.Tensor,
             training=False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``torch.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``torch.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : torch.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : torch.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """
        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError


class DanSequenceToVector(SequenceToVector):
    """
    It is a class defining Deep Averaging Network based Sequence to Vector
    encoder. You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this DAN encoder.
    dropout : `float`
        Token dropout probability as described in the paper.
    """
    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2, device = 'cpu'):
        super(DanSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        self._dropout = dropout     # define local variables
        self._device = device
        self._num_layers = num_layers

        seq_models = nn.ModuleList()    # create a sequential model for (n-1) hidden layers
        for i in range(num_layers):
            seq_models.append(nn.Linear(input_dim, input_dim))  # linear transformation
            seq_models.append(nn.ReLU())        # ReLu activation function
        self._hidden_layers = seq_models
        # TODO(students): end

    def forward(self,
             vector_sequence: torch.Tensor,
             sequence_mask: torch.Tensor,
             training=False) -> torch.Tensor:
        # TODO(students): start
        batch_size = len(vector_sequence)
        final_layers = []
        all_layers = []
        for i in range(batch_size):
            cur_batch = vector_sequence[i]
            with torch.no_grad():
                mask = torch.ones(len(cur_batch)).to(self._device)
                if training:        # dropout in training helps prevent overfitting
                    mask = torch.bernoulli(mask * (1 - self._dropout))
                    # print(f'number of samples chosen by bernoulli: {mask.sum()}')
                mask *= sequence_mask[i]        # filter out padding tokens
                input_seq = cur_batch[mask > 0]
                # print(f'#number of vectors after masking: {len(input_seq)}')

            batch_layers = []
            layer = torch.mean(input_seq, dim=0)      # avg(vector_sequence)
            # print(f'cur.shape: {layer.shape}')
            for j in range(self._num_layers):       # hidden layers
                # print(f'cur.shape: {layer.shape}')
                layer = self._hidden_layers[j*2](layer)
                layer = self._hidden_layers[j*2+1](layer)
                batch_layers.append(layer)
            final_layers.append(layer)
            all_layers.append(torch.stack(batch_layers))

        combined_vector = torch.stack(final_layers)
        # print(f'combined_vector.shape: {combined_vector.shape}')
        layer_representations = torch.stack(all_layers)
        # print(f'layer_representations.shape: {layer_representations.shape}')

        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}


class GruSequenceToVector(SequenceToVector):
    """
    It is a class defining GRU based Sequence To Vector encoder.
    You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """
    def __init__(self, input_dim: int, num_layers: int, device = 'cpu'):
        super(GruSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        self._device = device       # define local variables
        self._num_layers = num_layers
        self._input_dim = input_dim
        self._gru = nn.GRU(input_size=input_dim, hidden_size=input_dim, num_layers=num_layers, batch_first=True)
        # self._bn = nn.BatchNorm1d(input_dim)      # experiment with normalized output
        # TODO(students): end

    def forward(self,
             vector_sequence: torch.Tensor,
             sequence_mask: torch.Tensor,
             training=False) -> torch.Tensor:
        # TODO(students): start
        seq_lengths = sequence_mask.sum(dim=1).to("cpu")    # lengths has to be a 1D CPU int64 tensor
        packed_seq_batch = nn.utils.rnn.pack_padded_sequence(vector_sequence, lengths=seq_lengths, batch_first=True,
                                                             enforce_sorted=False)
        # experiment with xavier_normal weights
        # batch_size = len(sequence_mask)
        # hidden = nn.Parameter(
        #     nn.init.xavier_normal(torch.zeros(self._num_layers, batch_size, self._input_dim)),
        #     requires_grad=True)
        # out, hn = self._gru(packed_seq_batch, hidden)

        out, hn = self._gru(packed_seq_batch)
        # combined_vector = self._bn(hn[-1])    # experiment with normalized output
        combined_vector = hn[-1]
        # print(f'combined_vector.shape: {combined_vector.shape}')
        layer_representations = hn.transpose(0, 1)
        # print(f'layer_representations.shape: {layer_representations.shape}')
        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}
