import numpy as np
import pickle
import matplotlib.pyplot as plt

def plot(sorted_vectors, sorted_counts):
    # 벡터의 빈도를 바 차트로 시각화
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(sorted_vectors)), sorted_counts, tick_label=sorted_vectors)
    plt.xlabel('Direction Vector')
    plt.ylabel('Frequency')
    plt.title('Frequency of Direction Vectors (Descending Order)')
    plt.xticks(rotation=45)  # 라벨이 긴 경우 대각선으로 표시
    plt.show()

def plot_count(relative_sequence):
    # 벡터를 문자열로 변환하여 중복성 확인 및 빈도 계산
    vector_strs = [str(vector) for vector in relative_sequence]
    unique_vectors, counts = np.unique(vector_strs, return_counts=True)

    # 빈도가 50 이상인 벡터만 필터링
    filtered_indices = np.where(counts >= 1000)
    filtered_vectors = unique_vectors[filtered_indices]
    filtered_counts = counts[filtered_indices]

    # 빈도에 따라 내림차순으로 정렬
    sorted_indices = np.argsort(-filtered_counts)
    sorted_vectors = filtered_vectors[sorted_indices]
    sorted_counts = filtered_counts[sorted_indices]

    # 정렬된 결과 출력
    for vector, count in zip(sorted_vectors, sorted_counts):
        print(f"Vector: {vector}, Count: {count}")

    plot(sorted_vectors, sorted_counts)

if __name__ == '__main__':
    file_path = '../../datasets/preprocessed/sequence_datasets.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    coords_sequences = data['coords_sequences']
    id_sequences = data['id_sequences']
    category_sequences = data['category_sequences']

    same_direction = []
    for idx in range(len(coords_sequences)):
        right_coords = np.array(coords_sequences[idx][1:])
        left_coords = np.array(coords_sequences[idx][:-1])
        right_categories = np.array(category_sequences[idx][1:])
        left_categories = np.array(category_sequences[idx][:-1])

        # 카테고리가 같은 경우에만 방향 벡터 계산
        current_direction_sequence = np.array([right_coords[j] - left_coords[j]
                                               for j in range(len(right_coords))
                                               if right_categories[j] == left_categories[j]])

        for jdx in range(1, len(current_direction_sequence)):
            right = np.array(current_direction_sequence[jdx])
            left = np.array(current_direction_sequence[jdx - 1])

            same_direction += [np.all(right == left)]

    sum_sequence = np.array(same_direction, dtype=int)
    sum_sequence = np.sum(sum_sequence)
    print(sum_sequence / len(same_direction), sum_sequence, len(same_direction))