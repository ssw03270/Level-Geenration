import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

# 데이터 로드
with open('../../../datasets/preprocessed/reorder_sequence_datasets.pkl', 'rb') as f:
    data = pickle.load(f)

# 첫 번째 시퀀스만 사용
seq = [[0, 0, 0], [0, 0, 10], [0, 0, 11], [0, 1, 0], [0, 1, 11], [0, 2, 0], [0, 2, 11], [0, 3, 0], [0, 3, 11], [0, 4, 0], [0, 4, 11], [1, 0, 10], [1, 1, 10], [1, 2, 10], [1, 3, 10], [1, 4, 10], [2, 0, 10], [2, 1, 10], [2, 2, 10], [2, 3, 10], [2, 4, 10], [3, 0, 10], [3, 1, 10], [3, 2, 10], [3, 3, 10], [3, 4, 10], [4, 0, 10], [4, 1, 10], [4, 2, 10], [4, 3, 10], [4, 4, 10], [5, 0, 0], [5, 0, 1], [5, 1, 0], [5, 1, 1], [6, 0, 0], [6, 0, 1], [6, 1, 0], [6, 1, 1], [7, 0, 0], [7, 0, 1], [7, 1, 0], [7, 1, 1], [8, 0, 0], [8, 0, 1], [8, 1, 0], [8, 1, 1], [9, 0, 0], [9, 0, 1], [9, 1, 0], [9, 1, 1], [10, 0, 0], [10, 0, 1], [10, 1, 0], [10, 1, 1], [11, 0, 3], [11, 0, 4], [11, 1, 3], [11, 1, 4], [12, 0, 3], [12, 0, 4], [12, 1, 3], [12, 1, 4], [13, 0, 3], [13, 0, 4], [13, 1, 3], [13, 1, 4], [14, 0, 3], [14, 0, 4], [14, 1, 3], [14, 1, 4], [15, 0, 3], [15, 0, 4], [15, 1, 3], [15, 1, 4], [16, 0, 3], [16, 0, 4], [16, 1, 3], [16, 1, 4], [17, 0, 3], [17, 0, 4], [17, 1, 3], [17, 1, 4], [18, 0, 3], [18, 0, 4], [18, 1, 3], [18, 1, 4], [19, 0, 3], [19, 0, 4], [19, 1, 3], [19, 1, 4], [20, 0, 3], [20, 0, 4], [20, 1, 3], [20, 1, 4], [21, 0, 3], [21, 0, 4]]

# 플롯 초기화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([0, 30])
ax.set_ylim([0, 30])
ax.set_zlim([0, 30])

# 각 좌표에 대해 반복
for idx, coord in enumerate(tqdm(seq)):
    # 점 추가
    ax.scatter(coord[0], coord[2], coord[1], c='b')

    # 현재 상태의 그림 저장
plt.show()

plt.close()