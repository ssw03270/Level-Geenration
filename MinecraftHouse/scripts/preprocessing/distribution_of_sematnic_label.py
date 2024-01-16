import pickle
import matplotlib.pyplot as plt
from collections import Counter

with open('../../datasets/training_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(data['input_sequences'][0])

categories = list(item[-1] for sublist in data['input_sequences'] for item in sublist)
categories = sorted(categories)

word_counts = Counter(categories)
for word, count in word_counts.items():
    print(f"{word}: {count}번")

plt.hist(categories, bins='auto')

plt.title('Distribution of List Lengths')
plt.xlabel('Length')
plt.ylabel('Frequency')

plt.show()

