import pickle
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file_path = '../../datasets/preprocessed/sequence_datasets.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    coords_sequences = data['coords_sequences']
    id_sequences = data['id_sequences']
    category_sequences = data['category_sequences']

    relative_sequence = []
    max_idx = len(coords_sequences)
    max_idx = 10
    for idx in range(max_idx):
        coords = np.array(coords_sequences[idx])
        x_range = (coords[1:, 0] - coords[:-1, 0])
        y_range = (coords[1:, 1] - coords[:-1, 1])
        z_range = (coords[1:, 2] - coords[:-1, 2])

        # 선 그래프 그리기
        plt.plot(range(len(x_range)), x_range, label='x_pos')
        plt.plot(range(len(y_range)), y_range, label='y_pos')
        plt.plot(range(len(z_range)), z_range, label='z_pos')

        # 제목 및 레이블 추가
        plt.title("position / time")
        plt.xlabel("time")
        plt.ylabel("position")

        # 범례 추가
        plt.legend(loc='upper left')

        # 그래프 표시
        plt.show()