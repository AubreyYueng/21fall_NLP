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
        cur_batch_size = 0
        id = self.skip_window      # make it the index of current center word
        while id+self.skip_window < len(self.data):
            id += 1                    # update data_index

            w_c = self.data[id]    # center word

            # Draw samples of a window
            center_in_win = []
            context_in_win = []
            cur_id = max(0, id - self.skip_window)     # start of window
            samples_in_win = 0   # number of samples drawn in current window
            while samples_in_win < self.num_skips:
                if cur_id >= min(id + self.skip_window, len(self.data)):
                    break                       # reach the end of window or data
                if cur_id != id:   # not a center word
                    center_in_win.append(w_c)
                    context_in_win.append(self.data[cur_id])
                    samples_in_win += 1         # increase number of samples in current window
                cur_id += 1

            # Add drawn samples to batch
            n = min(self.batch_size - cur_batch_size , len(context_in_win))     # how many more samples can be add
            if n == 0:      # have reached batch_size
                break
            for i in range(n):
                center_array.append([center_in_win[i]])
                context_array.append([context_in_win[i]])
                cur_batch_size += 1

        if len(center_array) > 0:
            center_word = np.array(center_array, dtype=np.int32)
            context_word = np.array(context_array, dtype=np.int32)
            print(f'center index: {id}, center shape: {center_word.shape}, context shape: {context_word.shape}')

        ### TODO(students): end

        return torch.LongTensor(center_word), torch.LongTensor(context_word)