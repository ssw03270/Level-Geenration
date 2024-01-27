import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

# 데이터 로드
with open('../../../datasets/preprocessed/reorder_sequence_datasets.pkl', 'rb') as f:
    data = pickle.load(f)

# 첫 번째 시퀀스만 사용
start_idx = 0
end_idx = start_idx + 10
coords_seq = data['reorder_coords_sequences'][start_idx:end_idx]
categories_seq = data['reorder_category_sequences'][start_idx:end_idx]

for seq_index, (coords, categories) in enumerate(zip(coords_seq, categories_seq)):
    # 플롯 초기화
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([0, 30])
    ax.set_ylim([0, 30])
    ax.set_zlim([0, 30])

    # 고유한 카테고리들 추출 및 랜덤 색상 할당
    unique_categories = set(categories)
    colors = {category: "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
              for category in unique_categories}

    # 각 좌표에 대해 반복
    for idx, (coord, category) in enumerate(zip(tqdm(coords), categories)):
        # 점 추가
        ax.scatter(coord[0], coord[2], coord[1], c=colors[category])

        # 현재 상태의 그림 저장
        if idx % 10 == 0:
            plt.savefig(f"images/{seq_index + 1}-{idx + 1}.png")

    plt.close()