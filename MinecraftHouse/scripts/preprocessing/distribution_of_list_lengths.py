import pickle
import matplotlib.pyplot as plt

with open('../../datasets/training_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(data['input_sequences'][0])

# 각 리스트의 길이를 구합니다.
lengths = [len(lst) for lst in data['input_sequences']]

# 히스토그램을 그립니다.
# bins 파라미터는 히스토그램의 바구니 수를 의미합니다.
plt.hist(lengths, bins='auto')

# 그래프의 제목과 x축, y축의 레이블을 설정합니다.
plt.title('Distribution of List Lengths')
plt.xlabel('Length')
plt.ylabel('Frequency')

# 그래프를 출력합니다.
plt.show()

categories = set(item[-1] for sublist in data['input_sequences'] for item in sublist)
print("Number of unique categories:", len(categories))
print("Unique categories:", categories)

block_id = set(item[-2] for sublist in data['input_sequences'] for item in sublist)
print("Number of unique block id:", len(block_id))
print("Unique block_id:", block_id)