import pickle
import matplotlib.pyplot as plt
from collections import Counter

with open('../../datasets/training_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(data['input_sequences'][0])

ids = list(item[-2] for sublist in data['input_sequences'] for item in sublist)
ids = sorted(ids)

id_counts = Counter(ids)
for word, count in id_counts.items():
    print(f"{word}: {count}번")

print(len(ids))
ids = [i for i in ids if i != 2 and i != 5]
plt.hist(ids, bins='auto')

plt.title('Distribution of List Lengths')
plt.xlabel('Length')
plt.ylabel('Frequency')

plt.show()

