import pickle
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def category_continuity(category_sequences):
    # 카테고리별 연속성 계산
    category_continuity = defaultdict(int)
    category_count = defaultdict(int)

    for idx in range(len(category_sequences)):
        for i in range(len(category_sequences[idx]) - 1):
            category = category_sequences[idx][i]
            category_count[category] += 1
            if category_sequences[idx][i] == category_sequences[idx][i + 1]:
                category_continuity[category] += 1

    # 카테고리별 연속성 비율 계산 및 출력
    for category in category_continuity:
        continuity_rate = category_continuity[category] / category_count[category]
        print(f"Category: {category}, Continuity Rate: {continuity_rate:.2f}")

def plot(category_sequences):
    # 카테고리별 연속성 계산
    category_continuity = defaultdict(int)
    category_count = defaultdict(int)

    for idx in range(len(category_sequences)):
        for i in range(len(category_sequences[idx]) - 1):
            category = category_sequences[idx][i]
            category_count[category] += 1
            if category_sequences[idx][i] == category_sequences[idx][i + 1]:
                category_continuity[category] += 1

    categories = list(category_continuity.keys())
    continuity_rates = [category_continuity[cat] / category_count[cat] for cat in categories]

    plt.figure(figsize=(15, 6))  # 가로 15인치, 세로 6인치
    # 막대 그래프 시각화
    plt.bar(categories, continuity_rates)
    plt.xlabel('Category')
    plt.ylabel('Continuity Rate')
    plt.title('Continuity Rate by Block Category')

    # 눈금 라벨을 대각선으로 설정
    plt.xticks(rotation=45)  # 라벨을 45도 각도로 회전

    plt.show()

if __name__ == '__main__':
    file_path = '../../datasets/preprocessed/sequence_datasets.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    coords_sequences = data['coords_sequences']
    id_sequences = data['id_sequences']
    category_sequences = data['category_sequences']


    relative_sequence = []
    for idx in range(len(category_sequences)):
        right = np.array(category_sequences[idx][1:])
        left = np.array(category_sequences[idx][:-1])
        relative_sequence += np.equal(right, left).tolist()

    sum_sequence = np.array(relative_sequence, dtype=int)
    sum_sequence = np.sum(sum_sequence)
    print(sum_sequence / len(relative_sequence), sum_sequence, len(relative_sequence))

    category_continuity(category_sequences)
    plot(category_sequences)