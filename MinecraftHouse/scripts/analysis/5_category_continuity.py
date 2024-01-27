import pickle
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def remove_consecutive_duplicates(seq):
    result = []
    prev = None
    for item in seq:
        if item != prev:
            result.append(item)
            prev = item
    return result

if __name__ == '__main__':
    file_path = '../../datasets/preprocessed/sequence_datasets.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    coords_sequences = data['coords_sequences']
    id_sequences = data['id_sequences']
    category_sequences = data['category_sequences']


    dict = {}
    for idx in range(len(category_sequences)):
        category_sequence = remove_consecutive_duplicates(category_sequences[idx])
        for jdx in range(len(category_sequence) - 1):
            if category_sequence[jdx] not in dict:
                dict[category_sequence[jdx]] = []

            dict[category_sequence[jdx]].append(category_sequence[jdx + 1])

    for target_category in dict:
        unique_categories, category_counts = np.unique(dict[target_category], return_counts=True)

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(unique_categories)), category_counts, tick_label=unique_categories)
        plt.xlabel(f'Categories ({target_category})')
        plt.ylabel('Frequency')
        plt.title(f'Categories ({target_category})')
        plt.xticks(rotation=45)  # 라벨이 긴 경우 대각선으로 표시
        plt.savefig(f'{target_category}.png')