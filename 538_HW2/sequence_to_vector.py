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
        for i in range(num_layers-1):
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
                mask = torch.ones(len(cur_batch))
                if training:        # dropout in training helps prevent overfitting
                    mask = torch.bernoulli(mask * (1 - self._dropout))
                    # print(f'number of samples chosen by bernoulli: {mask.sum()}')
                mask *= sequence_mask[i]
                input_seq = cur_batch[mask > 0]     # filter out padding tokens
                # print(f'#number of vectors after masking: {len(input_seq)}')

            batch_layers = []
            layer = torch.mean(input_seq, dim=0)      # avg(vector_sequence)
            # print(f'cur.shape: {layer.shape}')
            for j in range(self._num_layers-1):       # hidden layers
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
        self._wh = nn.Linear(input_dim, 3 * input_dim)      # 3 hidden states related weights
        self._wx = nn.Linear(input_dim, 3 * input_dim)      # 3 input states related weights
        # TODO(students): end

    def forward(self,
             vector_sequence: torch.Tensor,
             sequence_mask: torch.Tensor,
             training=False) -> torch.Tensor:
        # TODO(students): start
        def gru_unit(x, h):     # a GRU unit
            ha, hb, hc = self._wh(h).chunk(3, dim=1)
            xa, xb, xc = self._wx(x).chunk(3, dim=1)
            update_gate = torch.sigmoid(ha + xa)
            reset_gate = torch.sigmoid(hb + xb)
            candidate = torch.tanh(xc + reset_gate * hc)
            return update_gate * h + (1 - update_gate) * candidate

        layers_output = []
        prev_input = vector_sequence[sequence_mask > 0]     # filter out padding tokens
        for i in range(self._num_layers):
            h_list = []
            h = torch.zeros(self._input_dim, self._input_dim).to(self._device)  # init hidden state with zero
            for x in prev_input:
                h = gru_unit(x, h)
                h_list.append(h)
            layers_output.append(h_list[-1])
            prev_input = h_list     # the newly computed h_list is used as the input of next layer

        combined_vector = layers_output[-1]
        layer_representations = torch.stack(layers_output, dim=-1)
        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}
