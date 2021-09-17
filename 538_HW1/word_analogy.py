"""
author-gh: @adithya8
editor-gh: ykl7
"""

import os
import pickle
import numpy as np
import argparse

np.random.seed(1234)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./baseline_models', help='Base directory of folder where models are saved')
    parser.add_argument('--input_filepath', type=str, default='./data/word_analogy_dev.txt', help='Word analogy file to evaluate on')
    parser.add_argument('--output_filepath', type=str, required=True, help='Predictions filepath')
    parser.add_argument("--loss_model", help="The loss function for training the word vector", default="nll", choices=["nll", "neg"])
    args, _ = parser.parse_known_args()
    return args

def read_data(file_path):
    with open(file_path,'r') as f:
        data = f.readlines()
    
    candidate, test = [], []
    for line in data:
        a, b = line.strip().split("||")
        a = [i[1:-1].split(":") for i in a.split(",")]
        b = [i[1:-1].split(":") for i in b.split(",")]
        candidate.append(a)
        test.append(b)
    
    return candidate, test

def get_embeddings(examples, embeddings):

    """
    For the word pairs in the 'examples' array, fetch embeddings and return.
    You can access your trained model via dictionary and embeddings.
    dictionary[word] will give you word_id
    and embeddings[word_id] will return the embedding for that word.

    word_id = dictionary[word]
    v1 = embeddings[word_id]

    or simply

    v1 = embeddings[dictionary[word_id]]
    """

    norm = np.sqrt(np.sum(np.square(embeddings),axis=1,keepdims=True))
    normalized_embeddings = embeddings/norm

    embs = []
    for line in examples:
        temp = []
        for pairs in line:
            temp.append([ normalized_embeddings[dictionary[pairs[0]]], normalized_embeddings[dictionary[pairs[1]]] ])
        embs.append(temp)

    result = np.array(embs)
    
    return result

def evaluate_pairs(candidate_embs, test_embs):

    """
    Write code to evaluate a relation between pairs of words.
    Find the best and worst pairs and return that.
    """

    best_pairs = []
    worst_pairs = []

    ### TODO(students): start
    for i, line in enumerate(candidate):
        # normalized diff vectors
        test_diffs = [vec / np.linalg.norm(vec) for vec in [(x-y) for [x, y] in test_embs[i]]]
        candidate_diffs = [vec / np.linalg.norm(vec) for vec in [(x-y) for [x, y] in candidate_embs[i]]]

        word_similarity = []    # candidate word pair -> sum of cos
        for j, cur in enumerate(line):
            can = candidate_diffs[j]
            word_similarity.append([cur, sum([(can * t).sum() for t in test_diffs])])
        word_similarity.sort(key=lambda x: x[1])    # sort by cos
        best_pairs.append(word_similarity[-1][0])   # best: largest cos
        worst_pairs.append(word_similarity[0][0])
    ### TODO(students): end
    
    return best_pairs, worst_pairs

def write_solution(best_pairs, worst_pairs, test, path):

    """
    Write best and worst pairs to a file, that can be evaluated by evaluate_word_analogy.pl
    """
    
    ans = []
    for i, line in enumerate(test):
        temp = [f'"{pairs[0]}:{pairs[1]}"' for pairs in line]
        temp.append(f'"{line[worst_pairs[i]][0]}:{line[worst_pairs[i]][1]}"')
        temp.append(f'"{line[best_pairs[i]][0]}:{line[best_pairs[i]][1]}"')
        ans.append(" ".join(temp))

    with open(path, 'w') as f:
        f.write("\n".join(ans))


if __name__ == '__main__':

    args = parse_args()

    loss_model = args.loss_model
    model_path = args.model_path
    input_filepath = args.input_filepath

    print(f'Model file: {model_path}/word2vec_{loss_model}.model')
    model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

    dictionary, embeddings = pickle.load(open(model_filepath, 'rb'))

    candidate, test = read_data(input_filepath)

    candidate_embs = get_embeddings(candidate, embeddings)
    test_embs = get_embeddings(test, embeddings)

    best_pairs, worst_pairs = evaluate_pairs(candidate_embs, test_embs)

    out_filepath = args.output_filepath
    print(f'Output file: {out_filepath}')
    write_solution(best_pairs, worst_pairs, test, out_filepath)