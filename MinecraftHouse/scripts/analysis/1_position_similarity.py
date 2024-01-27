import numpy as np
import pickle
from scipy import stats
import matplotlib.pyplot as plt

def shapiro(sum_relative_sequence):
    # 정규성 검사 (Shapiro-Wilk)
    shapiro_statistic, shapiro_p = stats.shapiro(sum_relative_sequence)

    # 정규성 검사 결과 출력
    print("Shapiro-Wilk Test statistic:", shapiro_statistic)
    print("Shapiro-Wilk Test p-value:", shapiro_p)

def wilcoxon(sum_relative_sequence):
    w_statistic, p_value = stats.wilcoxon(sum_relative_sequence - np.median(sum_relative_sequence))

    # 결과 출력
    print("Wilcoxon statistic:", w_statistic)
    print("p-value:", p_value)

def plot_distriubtion(sum_relative_sequence):
    bins = np.arange(min_relative_sequence, max_relative_sequence + 2)  # max_distance+1까지 포함하기 위해 +2

    plt.figure(figsize=(15, 6))  # 가로 15인치, 세로 6인치

    # 히스토그램 생성
    plt.hist(sum_relative_sequence, bins=bins, align='mid', edgecolor='black')  # bins는 구간의 개수를 조절합니다.

    # 시각화 설정
    plt.title("Histogram of Distances Between Blocks")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")

    # x축 눈금을 1씩 증가하도록 설정
    plt.xticks(np.arange(min_relative_sequence, max_relative_sequence + 1, 1))

    # 히스토그램 표시
    plt.show()

if __name__ == '__main__':
    file_path = '../../datasets/preprocessed/sequence_datasets.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    coords_sequences = data['coords_sequences']
    id_sequences = data['id_sequences']
    category_sequences = data['category_sequences']


    relative_sequence = []
    for idx in range(len(coords_sequences)):
        right = np.array(coords_sequences[idx][1:])
        left = np.array(coords_sequences[idx][:-1])
        relative_sequence += (right - left).tolist()

    relative_sequence = np.array(relative_sequence)
    relative_sequence = np.abs(relative_sequence)

    sum_relative_sequence = np.sum(relative_sequence, axis=1)
    mean_relative_sequence = np.mean(sum_relative_sequence)
    std_relative_sequence = np.std(sum_relative_sequence)
    min_relative_sequence = np.min(sum_relative_sequence)
    max_relative_sequence = np.max(sum_relative_sequence)
    median_relative_sequence = np.median(sum_relative_sequence)

    print(mean_relative_sequence, std_relative_sequence, min_relative_sequence, max_relative_sequence, median_relative_sequence)

    wilcoxon(sum_relative_sequence)