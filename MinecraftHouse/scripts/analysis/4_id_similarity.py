import pickle
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def id_continuity(id_sequences):
    # 카테고리별 연속성 계산
    id_continuity = defaultdict(int)
    id_count = defaultdict(int)

    for idx in range(len(id_sequences)):
        for i in range(len(id_sequences[idx]) - 1):
            id = id_sequences[idx][i]
            id_count[id] += 1
            if id_sequences[idx][i] == id_sequences[idx][i + 1]:
                id_continuity[id] += 1

    # 카테고리별 연속성 비율 계산 및 출력
    for id in id_continuity:
        continuity_rate = id_continuity[id] / id_count[id]
        print(f"Category: {id}, Continuity Rate: {continuity_rate:.2f}")

def plot(id_sequences):
    # 카테고리별 연속성 계산
    id_continuity = defaultdict(int)
    id_count = defaultdict(int)

    for idx in range(len(id_sequences)):
        for i in range(len(id_sequences[idx]) - 1):
            id = id_sequences[idx][i]
            id_count[id] += 1
            if id_sequences[idx][i] == id_sequences[idx][i + 1]:
                id_continuity[id] += 1

    categories = list(id_continuity.keys())
    continuity_rates = [id_continuity[cat] / id_count[cat] for cat in categories]

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


    relative_sequence = []
    for idx in range(len(id_sequences)):
        right = np.array(id_sequences[idx][1:])
        left = np.array(id_sequences[idx][:-1])
        relative_sequence += np.equal(right, left).tolist()

    sum_sequence = np.array(relative_sequence, dtype=int)
    sum_sequence = np.sum(sum_sequence)
    print(sum_sequence / len(relative_sequence), sum_sequence, len(relative_sequence))

    id_continuity(id_sequences)
    plot(id_sequences)