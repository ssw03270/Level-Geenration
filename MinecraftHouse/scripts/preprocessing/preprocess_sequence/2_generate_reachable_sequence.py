import pickle
from tqdm import tqdm

def is_every_coordinate_reachable_improved_26_directions(coords):
    def is_directly_reachable(coord1, coord2):
        dx, dy, dz = abs(coord1[0] - coord2[0]), abs(coord1[1] - coord2[1]), abs(coord1[2] - coord2[2])
        return dx <= 1 and dy <= 1 and dz <= 1

    # Check if each coordinate is reachable from any of the previous coordinates
    for i in range(1, len(coords)):
        if not any(is_directly_reachable(coords[j], coords[i]) for j in range(i)):
            return False  # If any coordinate is not reachable, return False

    return True  # All coordinates are reachable

def reorder_to_ensure_reachability_26_directions(coords, ids, categorys):
    def is_directly_reachable(coord1, coord2):
        dx, dy, dz = abs(coord1[0] - coord2[0]), abs(coord1[1] - coord2[1]), abs(coord1[2] - coord2[2])
        return dx <= 1 and dy <= 1 and dz <= 1

    n = len(coords)
    swap_count = 0
    for i in range(1, n):
        # If the current coordinate is not reachable from any previous ones, find a reachable coordinate to swap
        if not any(is_directly_reachable(coords[j], coords[i]) for j in range(i)):
            for j in range(i + 1, n):
                if any(is_directly_reachable(coords[k], coords[j]) for k in range(i)):
                    coords[i], coords[j] = coords[j], coords[i]
                    ids[i], ids[j] = ids[j], ids[i]
                    categorys[i], categorys[j] = categorys[j], categorys[i]
                    swap_count += 1
                    break

    return coords, ids, categorys, swap_count

if __name__ == '__main__':
    file_path = '../../../datasets/preprocessed/sequence_datasets.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    coords_sequences = data['coords_sequences']
    id_sequences = data['id_sequences']
    category_sequences = data['category_sequences']

    reorder_coords_sequences = []
    reorder_id_sequences = []
    reorder_category_sequences = []

    for idx in tqdm(range(len(coords_sequences))):
        coords_sequence = coords_sequences[idx]
        id_sequence = id_sequences[idx]
        category_sequence = category_sequences[idx]

        reorder_coords_sequence, reorder_id_sequence, reorder_category_sequence, swap_count = reorder_to_ensure_reachability_26_directions(coords_sequence, id_sequence, category_sequence)
        # print(swap_count, is_every_coordinate_reachable_improved_26_directions(reorder_coords_sequence))

        reorder_coords_sequences.append(reorder_coords_sequence)
        reorder_id_sequences.append(reorder_id_sequence)
        reorder_category_sequences.append(reorder_category_sequence)

    with open('../../../datasets/preprocessed/reorder_sequence_datasets.pkl', 'wb') as f:
        pickle.dump({'reorder_coords_sequences': reorder_coords_sequences,
                     'reorder_id_sequences': reorder_id_sequences,
                     'reorder_category_sequences': reorder_category_sequences}, f)
