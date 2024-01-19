import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

dir_dictionary = {
    0: [-1, -1, -1],
    1: [-1, -1, 0],
    2: [-1, -1, 1],
    3: [-1, 0, -1],
    4: [-1, 0, 0],
    5: [-1, 0, 1],
    6: [-1, 1, -1],
    7: [-1, 1, 0],
    8: [-1, 1, 1],
    9: [0, -1, -1],
    10: [0, -1, 0],
    11: [0, -1, 1],
    12: [0, 0, -1],
    13: [0, 0, 1],
    14: [0, 1, -1],
    15: [0, 1, 0],
    16: [0, 1, 1],
    17: [1, -1, -1],
    18: [1, -1, 0],
    19: [1, -1, 1],
    20: [1, 0, -1],
    21: [1, 0, 0],
    22: [1, 0, 1],
    23: [1, 1, -1],
    24: [1, 1, 0],
    25: [1, 1, 1]
}

# dir_dictionary = {
#     0: [0, 0, -1],
#     1: [0, 0, 1],
#     2: [0, -1, 0],
#     3: [0, 1, 0],
#     4: [-1, 0, 0],
#     5: [1, 0, 0],
# }

def check(height_list, schematic, annotated_schematic, annotation_list):
    input_sequence = [[[0, 0, 0], 0, 'sos']]
    output_sequence = [[0, 0]]

    block_sequence = []
    for height in height_list:
        block_sequence.append([height.tolist(), 2, 'terrain'])

    least_num = 0
    while len(block_sequence) > 0:
        new_input_sequence = []
        for idx, new_seq in enumerate(input_sequence):
            for block_seq in block_sequence:
                for dir_idx in range(len(dir_dictionary)):
                    dx = dir_dictionary[dir_idx][0]
                    dy = dir_dictionary[dir_idx][1]
                    dz = dir_dictionary[dir_idx][2]

                    if block_seq[0] == [new_seq[0][0] + dx, new_seq[0][1] + dy, new_seq[0][2] + dz]:
                        new_input_sequence.append(block_seq)
                        block_sequence.remove(block_seq)

                        output_sequence.append([idx, dir_idx])

        input_sequence += new_input_sequence

        if len(block_sequence) == 0:
            break

        if least_num == len(block_sequence):
            break

        least_num = len(block_sequence)

    for x in range(schematic.shape[0]):
        for y in range(schematic.shape[1]):
            for z in range(schematic.shape[2]):
                if schematic[x, y, z] > 0:
                    block_sequence.append(
                        [[x, y, z], schematic[x, y, z], annotation_list[int(annotated_schematic[x, y, z])]])

    least_num = 0
    while len(block_sequence) > 0:
        new_input_sequence = []
        for idx, new_seq in enumerate(input_sequence):
            for block_seq in block_sequence:
                for dir_idx in range(len(dir_dictionary)):
                    dx = dir_dictionary[dir_idx][0]
                    dy = dir_dictionary[dir_idx][1]
                    dz = dir_dictionary[dir_idx][2]

                    if block_seq[0] == [new_seq[0][0] + dx, new_seq[0][1] + dy, new_seq[0][2] + dz]:
                        new_input_sequence.append(block_seq)
                        block_sequence.remove(block_seq)

                        output_sequence.append([idx, dir_idx])

        input_sequence += new_input_sequence

        if len(block_sequence) == 0:
            return 0, input_sequence, output_sequence

        if least_num == len(block_sequence):
            return 1, input_sequence, output_sequence

        least_num = len(block_sequence)

    return 0, input_sequence, output_sequence

if __name__ == '__main__':
    file_path = '../../../datasets/instance_segmentation_data/preprocessed_training_data_with_terrain.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    schematics = data['schematics']
    annotated_schematics = data['annotated_schematics']
    annotation_lists = data['annotation_list']
    house_names = data['house_names']
    height_lists = data['height_list']

    fail_count = 0
    input_sequences = []
    output_sequences = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(check, height_lists[idx], schematics[idx], annotated_schematics[idx], annotation_lists[idx]) for idx in range(len(schematics))]
        for future in tqdm(as_completed(futures), total=len(schematics)):
            count, input_sequence, output_sequence = future.result()
            fail_count += count
            if count == 0:
                input_sequences.append(input_sequence[1:])
                output_sequences.append(output_sequence[1:])

    print(f'fail: {fail_count}, total: {len(schematics)}')
    print(f'longest_sequence: {len(max(input_sequences, key=len))}')

    with open('../../../datasets/training_data2.pkl', 'wb') as f:
        pickle.dump({'input_sequences': input_sequences,
                     'output_sequences': output_sequences}, f)