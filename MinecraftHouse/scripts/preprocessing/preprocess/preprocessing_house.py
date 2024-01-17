import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def check(height_list, schematic, annotated_schematic, annotation_list):
    block_sequence = []
    for x in range(schematic.shape[0]):
        for y in range(schematic.shape[1]):
            for z in range(schematic.shape[2]):
                if schematic[x, y, z] > 0:
                    block_sequence.append(
                        [[x, y, z], schematic[x, y, z], annotation_list[int(annotated_schematic[x, y, z])]])

    terrain_sequence = []
    for height in height_list:
        terrain_sequence.append([height.tolist(), 2, 'terrain'])

    input_sequence = terrain_sequence
    output_sequence = [[0, 0]] * len(terrain_sequence)

    least_num = 0
    while len(block_sequence) > 0:
        new_input_sequence = []
        for idx, new_seq in enumerate(input_sequence):
            for block_seq in block_sequence:
                dir_idx = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            # 자기 자신을 제외한 모든 방향
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                            if [block_seq[0][0] + dx, block_seq[0][1] + dy, block_seq[0][2] + dz] == new_seq[0]:
                                new_input_sequence.append(block_seq)
                                block_sequence.remove(block_seq)

                                output_sequence.append([idx + 1, dir_idx])

                            dir_idx += 1

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
                input_sequences.append(input_sequence)
                output_sequences.append(output_sequence)

    print(f'fail: {fail_count}, total: {len(schematics)}')
    print(f'longest_sequence: {len(max(input_sequences, key=len))}')

    with open('../../../datasets/training_data.pkl', 'wb') as f:
        pickle.dump({'input_sequences': input_sequences,
                     'output_sequences': output_sequences}, f)