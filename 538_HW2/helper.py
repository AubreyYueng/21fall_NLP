# -*- coding:utf-8 -*-

"""
@author: Yiyun Yang
@time: 2021/9/30 17:26
"""

import os
import sys
from datetime import datetime
import inspect
import json


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
p_test = {bigram: "bigram_order_test", imdb: "imdb_sentiment_test"}

f_ts = lambda: int(round(datetime.now().timestamp()))
f_model_suffix = lambda dan_or_gru, data_size: f'_{dan_or_gru}_{data_size}_with_emb'    # eg. _dan_5k_with_emb
f_model = lambda dan_or_gru, data_size: f'main{f_model_suffix(dan_or_gru, data_size)}'  # eg. main_dan_5k_with_emb


# --------------------------------------------- 0. Data Preparation ---------------------------------------------
def train():
    func_name = inspect.currentframe().f_code.co_name
    ts = f_ts()
    exc_output = f'{p_output}/{func_name}_{ts}'

    cmd_list = ['python train.py main data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --seq2vec-choice dan --embedding-dim 50 --num-layers 4 --num-epochs 8 --suffix-name _dan_5k_with_emb --pretrained-embedding-file data/glove.6B.50d.txt ',
                'python train.py main data/imdb_sentiment_train_10k.jsonl data/imdb_sentiment_dev.jsonl --seq2vec-choice dan --embedding-dim 50 --num-layers 4 --num-epochs 8 --suffix-name _dan_10k_with_emb --pretrained-embedding-file data/glove.6B.50d.txt ',
                'python train.py main data/imdb_sentiment_train_15k.jsonl data/imdb_sentiment_dev.jsonl --seq2vec-choice dan --embedding-dim 50 --num-layers 4 --num-epochs 8 --suffix-name _dan_15k_with_emb --pretrained-embedding-file data/glove.6B.50d.txt ',
                'python train.py main data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --seq2vec-choice gru --embedding-dim 50 --num-layers 4 --num-epochs 4 --suffix-name _gru_5k_with_emb --pretrained-embedding-file data/glove.6B.50d.txt ',
                'python train.py main data/imdb_sentiment_train_10k.jsonl data/imdb_sentiment_dev.jsonl --seq2vec-choice gru --embedding-dim 50 --num-layers 4 --num-epochs 4 --suffix-name _gru_10k_with_emb --pretrained-embedding-file data/glove.6B.50d.txt ',
                'python train.py main data/imdb_sentiment_train_15k.jsonl data/imdb_sentiment_dev.jsonl --seq2vec-choice gru --embedding-dim 50 --num-layers 4 --num-epochs 4 --suffix-name _gru_15k_with_emb --pretrained-embedding-file data/glove.6B.50d.txt ']
    for cmd in cmd_list:
        os.system(f'echo "======== executing: {cmd} ========" >> {exc_output}')
        os.system(f'{cmd} 2>&1 | tee -a {exc_output}')


def probing():
    func_name = inspect.currentframe().f_code.co_name
    ts = f_ts()
    exc_output = f'{p_output}/{func_name}_{ts}'

    cmd_list = ['python train.py probing data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --base-model-dir serialization_dirs/main_dan_5k_with_emb --layer-num 1 --num-epochs 8 --suffix-name _sentiment_dan_with_emb_on_5k_at_layer_1',
                'python train.py probing data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --base-model-dir serialization_dirs/main_dan_5k_with_emb --layer-num 2 --num-epochs 8 --suffix-name _sentiment_dan_with_emb_on_5k_at_layer_2',
                'python train.py probing data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --base-model-dir serialization_dirs/main_dan_5k_with_emb --layer-num 3 --num-epochs 8 --suffix-name _sentiment_dan_with_emb_on_5k_at_layer_3',
                'python train.py probing data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --base-model-dir serialization_dirs/main_dan_5k_with_emb --layer-num 4 --num-epochs 8 --suffix-name _sentiment_dan_with_emb_on_5k_at_layer_4',
                'python train.py probing data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --base-model-dir serialization_dirs/main_gru_5k_with_emb --layer-num 1 --num-epochs 4 --suffix-name _sentiment_gru_with_emb_on_5k_at_layer_1',
                'python train.py probing data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --base-model-dir serialization_dirs/main_gru_5k_with_emb --layer-num 2 --num-epochs 4 --suffix-name _sentiment_gru_with_emb_on_5k_at_layer_2',
                'python train.py probing data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --base-model-dir serialization_dirs/main_gru_5k_with_emb --layer-num 3 --num-epochs 4 --suffix-name _sentiment_gru_with_emb_on_5k_at_layer_3',
                'python train.py probing data/imdb_sentiment_train_5k.jsonl data/imdb_sentiment_dev.jsonl --base-model-dir serialization_dirs/main_gru_5k_with_emb --layer-num 4 --num-epochs 4 --suffix-name _sentiment_gru_with_emb_on_5k_at_layer_4']
    for cmd in cmd_list:
        os.system(f'echo "======== executing: {cmd} ========" >> {exc_output}')
        os.system(f'{cmd} 2>&1 | tee -a {exc_output}')

# --------------------------------------------- 1. Learning Curves ----------------------------------------------------
# ---------------------------- (a) Performance with respect to training dataset size ----------------------------
# run plot_performance_against_data_size.py


# ---------------------------- (b) Performance with respect to training time ----------------------------
def train_dan_for_long():
    func_name = inspect.currentframe().f_code.co_name
    exc_output = f'{p_output}/{func_name}_{f_ts()}'
    os.system(f"./train_dan_for_long.sh 2>&1 | tee {exc_output}")


# check train/validation losses on the tensorboard (http://localhost:6006/) after running:
# Note: If you run training multiple times with same name, make sure to clean-up tensorboard directory.
#       Or else, it will have multiple plots in same chart.
# tensorboard --logdir serialization_dirs/main_dan_5k_with_emb_for_50k


# --------------------------------------------- 2. Error Analysis ----------------------------------------------------
# run trained models on example test cases
def error_analysis():
    func_name = inspect.currentframe().f_code.co_name
    ts = f_ts()
    exc_output = f'{p_output}/{func_name}_{ts}'
    for dan_or_gru in [gru, dan]:
        for data_size in [size_5, size_10, size_15]:
            for task in [imdb, bigram]:
                model = f_model(dan_or_gru, data_size)
                test_data = p_test[task]
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


def record_failure_cases():
    for dan_or_gru in [gru, dan]:
        for data_size in [size_5, size_10, size_15]:
            for task in [imdb, bigram]:
                model = f_model(dan_or_gru, data_size)
                test_data = p_test[task]
                prediction_data_path = f'{p_predict}/{model}_{test_data}.txt'
                gold_data_path = f'data/{test_data}.jsonl'

                error_file = f'{prediction_data_path}_error'
                os.system(f'rm -f {error_file}')

                with open(gold_data_path) as file:
                    gold_labels = [int(json.loads(line.strip())["label"])
                                   for line in file.readlines() if line.strip()]

                with open(gold_data_path) as file:
                    all_texts = [str(json.loads(line.strip())["text"]) for line in file.readlines() if line.strip()]

                with open(prediction_data_path) as file:
                    predicted_labels = [int(line.strip())
                                        for line in file.readlines() if line.strip()]

                f = open(error_file, "w")
                for i, result in enumerate(gold_labels):
                    gold = gold_labels[i]
                    predict = predicted_labels[i]
                    if gold != predict:
                        f.write(f'{all_texts[i]}\n')
                f.close()


# ---------------------------------- 3. Probing Performances on Sentiment Task ----------------------------------
# run plot_probing_performances_on_sentiment_task.py


# ---------------------------------- 4. Probing Performances on Bigram Order Task ----------------------------------
# run plot_probing_performances_on_bigram_order_task.py


# ------------------------------------------ 5. Perturbation Analysis ------------------------------------------
def plot_perturbation():
    func_name = inspect.currentframe().f_code.co_name
    ts = f_ts()
    exc_output = f'{p_output}/{func_name}_{ts}'
    os.system(f'python3 plot_perturbation_analysis.py 2>&1 | tee -a {exc_output}')


if __name__ == '__main__':
    sys.path.extend(['/Users/evenyoung/Desktop/2021_NLP/hw/538_HW2'])
    os.system("pwd")

    train()
    # probing()

    # run plot_performance_against_data_size.py

    # train_dan_for_long()
    # tensorboard --logdir serialization_dirs/main_dan_5k_with_emb_for_50k

    # error_analysis()
    # record_failure_cases()

    # run plot_probing_performances_on_sentiment_task.py
    # run plot_probing_performances_on_bigram_order_task.py

    # plot_perturbation()
