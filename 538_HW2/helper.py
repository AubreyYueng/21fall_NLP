# -*- coding:utf-8 -*-

"""
@author: Yiyun Yang
@time: 2021/9/30 17:26
"""

import os
import sys
from datetime import datetime
import inspect


p_output = "output"                     # output file path
p_predict = "predictions"               # predicted results file path
os.system(f"mkdir -p {p_predict}")      # create if not exists
os.system(f"mkdir -p {p_output}")

dan = 'dan'
gru = 'gru'
bigram = 'bigram'
imdb = 'imdb'
size_5 = '5k'
size_10 = '10k'
size_15 = '15k'
imdb_sentiment_test = 'imdb_sentiment_test'

f_ts = lambda: int(round(datetime.now().timestamp()))
f_model_suffix = lambda dan_or_gru, data_size: f'_{dan_or_gru}_{data_size}_with_emb'    # eg. _dan_5k_with_emb
f_model = lambda dan_or_gru, data_size: f'main{f_model_suffix(dan_or_gru, data_size)}'  # eg. main_dan_5k_with_emb


# --------------------------------------------- 0. Train Models ---------------------------------------------
def train():
    func_name = inspect.currentframe().f_code.co_name
    ts = f_ts()
    exc_output = f'{p_output}/{func_name}_{ts}'

    for dan_or_gru in [gru, dan]:
        for data_size in [size_5, size_10, size_15]:
            os.system(f'echo "======== Training: {dan_or_gru}_{data_size} ========" >> {exc_output}')
            cmd = f'python train.py main data/imdb_sentiment_train_{data_size}.jsonl \
                        data/imdb_sentiment_dev.jsonl \
                        --seq2vec-choice {dan_or_gru} \
                        --embedding-dim 50 \
                        --num-layers 4 \
                        --num-epochs 8 \
                        --suffix-name {f_model_suffix(dan_or_gru, data_size)} \
                        --pretrained-embedding-file data/glove.6B.50d.txt \
                        2>&1 | tee {exc_output}'
            os.system(cmd)


# --------------------------------------------- 1. Learning Curves ----------------------------------------------------
# ---------------------------- (a) Performance with respect to training dataset size ----------------------------
def plot_performance_against_data_size():
    func_name = inspect.currentframe().f_code.co_name
    exc_output = f'{p_output}/{func_name}_{f_ts()}'
    os.system(f"python plot_performance_against_data_size.py 2>&1 | tee {exc_output}")


# ---------------------------- (b) Performance with respect to training time ----------------------------
def train_dan_for_long():
    func_name = inspect.currentframe().f_code.co_name
    exc_output = f'{p_output}/{func_name}_{f_ts()}'
    os.system(f"./train_dan_for_long.sh 2>&1 | tee {exc_output}")


# check train/validation losses on the tensorboard (http://localhost:6006/) after running:
# Note: If you run training multiple times with same name, make sure to clean-up tensorboard directory.
#       Or else, it will have multiple plots in same chart.
def run_tensorboard():
    exc_output = f'{p_output}/run_tensorboard_{f_ts()}'
    os.system(f"tensorboard --logdir serialization_dirs/main_dan_5k_with_emb_for_50k 2>&1 | tee {exc_output}")


# --------------------------------------------- 2. Error Analysis ----------------------------------------------------
# run trained models on example test cases
def error_analysis():
    func_name = inspect.currentframe().f_code.co_name
    ts = f_ts()
    exc_output = f'{p_output}/{func_name}_{ts}'
    for dan_or_gru in [gru, dan]:
        for data_size in [size_5, size_10, size_15]:
            model = f_model(dan_or_gru, data_size)
            test_data = imdb_sentiment_test
            predict_file = f'{p_predict}/{model}_{test_data}'

            os.system(f'echo "======== Making prediction using {model} on {test_data} ========" >> {exc_output}')
            os.system(f"python3 predict.py serialization_dirs/{model} \
                                    data/{test_data}.jsonl \
                                    --predictions-file {predict_file}.txt \
                                    2>&1 | tee -a {exc_output}")

            os.system(f'echo "======== Evaluating prediction results: {predict_file} ========" >> {exc_output}')
            os.system(f"python3 evaluate.py data/{test_data}.jsonl \
                                    {predict_file}.txt \
                                    2>&1 | tee -a {exc_output}")

# ---------------------------------- 3. Probing Performances on Sentiment Task ----------------------------------


# ---------------------------------- 4. Probing Performances on Bigram Order Task ----------------------------------


# ------------------------------------------ 5. Perturbation Analysis ------------------------------------------


if __name__ == '__main__':
    sys.path.extend(['/Users/evenyoung/Desktop/2021_NLP/hw/538_HW2'])
    os.system("pwd")

    # train_models_small_epoch()

    # plot_performance_against_data_size()

    # train_dan_for_long()
    # run_tensorboard()

    # error_analysis()

