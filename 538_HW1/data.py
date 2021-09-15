"""
author-gh: @adithya8
editor-gh: ykl7
"""

import collections

import numpy as np
import torch

np.random.seed(1234)
torch.manual_seed(1234)

# Read the data into a list of strings.
def read_data(filename):
    with open(filename) as file:
        text = file.read()
        data = [token.lower() for token in text.strip().split(" ")]
    return data

def build_dataset(words, vocab_size):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocab_size - 1))
    # token_to_id dictionary, id_to_taken reverse_dictionary
    vocab_token_to_id = dict()
    for word, _ in count:
        vocab_token_to_id[word] = len(vocab_token_to_id)
    data = list()
    unk_count = 0
    for word in words:
        if word in vocab_token_to_id:
            index = vocab_token_to_id[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    vocab_id_to_token = dict(zip(vocab_token_to_id.values(), vocab_token_to_id.keys()))
    return data, count, vocab_token_to_id, vocab_id_to_token

class Dataset:
    def __init__(self, data, batch_size=128, num_skips=8, skip_window=4):
        """
        @data_index: the index of a word. You can access a word using data[data_index]
        @batch_size: the number of instances in one batch
        @num_skips: the number of samples you want to draw in a window 
                (In the below example, it was 2)
        @skip_windows: decides how many words to consider left and right from a context word. 
                    (So, skip_windows*2+1 = window_size)
        """

        self.data_index=0
        self.data = data
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window

        self.batch_size = batch_size
        self.num_skips = num_skips
        self.skip_window = skip_window
    
    def reset_index(self, idx=0):
        self.data_index=idx

    def generate_batch(self):
        """
        Write the code generate a training batch

        batch will contain word ids for context words. Dimension is [batch_size].
        labels will contain word ids for predicting(target) words. Dimension is [batch_size, 1].
        """
        
        center_word = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        context_word = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        
        # stride: for the rolling window
        stride = 1 

        ### TODO(students): start
        center_array = []
        context_array = []
        sample_id = 0
        while self.data_index + self.skip_window * 2 < len(data):
            # draw samples inside a window
            center_in_win = []
            context_in_win = []
            w_c = self.data[self.data_index + self.skip_window]

            i = 0
            while i < self.num_skips:
                if i != self.skip_window:
                    center_in_win.append(w_c)
                    context_in_win.append(self.data_index + i)
                i += 1

            self.data_index += 1

            n = min(self.batch_size - sample_id , self.num_skips)
            if n == 0:
                break
            for i in range(n):
                center_array.append(center_in_win[i])
                context_array.append([context_in_win[i]])
                sample_id += 1

        if len(center_array) > 0:
            center_word = np.array(center_array, dtype=np.int32)
            context_word = np.array(context_array, dtype=np.int32)
            print(f'data_index: {self.data_index}, ceter shape: {center_word.shape}, context shape: {context_word.shape}')

        ### TODO(students): end

        return torch.LongTensor(center_word), torch.LongTensor(context_word)