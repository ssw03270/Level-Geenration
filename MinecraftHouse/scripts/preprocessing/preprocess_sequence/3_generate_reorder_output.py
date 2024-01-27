import pickle
from tqdm import tqdm

def find_single_adjacent_coord(coords):
    def get_direction(coord1, coord2):
        dx, dy, dz = coord2[0] - coord1[0], coord2[1] - coord1[1], coord2[2] - coord1[2]
        return (dx, dy, dz)

    def is_adjacent(coord1, coord2):
        dx, dy, dz = abs(coord1[0] - coord2[0]), abs(coord1[1] - coord2[1]), abs(coord1[2] - coord2[2])
        return dx <= 1 and dy <= 1 and dz <= 1

    parent_indices = [0] * len(coords)  # Initialize with -1 to indicate no parent
    parent_directions = [(0, 0, 0)] * len(coords)

    for i in range(1, len(coords)):  # Start from the second coordinate
        for j in range(i - 1, -1, -1):
            if is_adjacent(coords[i], coords[j]):
                parent_indices[i] = j + 1
                parent_directions[i] = get_direction(coords[j], coords[i])
                break  # Only find the first adjacent coordinate

    return parent_indices, parent_directions

if __name__ == '__main__':
    file_path = '../../../datasets/preprocessed/reorder_sequence_datasets.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    reorder_coords_sequences = data['reorder_coords_sequences']
    reorder_id_sequences = data['reorder_id_sequences']
    reorder_category_sequences = data['reorder_category_sequences']

    parent_sequences = []
    direction_sequences = []
    for idx in tqdm(range(len(reorder_coords_sequences))):
        reorder_coords_sequence = reorder_coords_sequences[idx]
        indices, direction = find_single_adjacent_coord(reorder_coords_sequence)

        parent_sequences.append(indices)
        direction_sequences.append(direction)

    print(parent_sequences[0], direction_sequences[0])
    with open('../../../datasets/preprocessed/output_sequence_datasets.pkl', 'wb') as f:
        pickle.dump({'parent_sequences': parent_sequences,
                     'direction_sequences': direction_sequences}, f)
