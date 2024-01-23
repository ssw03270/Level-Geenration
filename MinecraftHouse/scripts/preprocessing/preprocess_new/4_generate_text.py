import pickle
import json
from tqdm import tqdm

def create_id_name_dict(json_data):
    id_name_dict = {}
    for item in json_data:
        id_name_dict[item['id']] = item['name']
    return id_name_dict

if __name__ == '__main__':
    training_data_path = '../../../datasets/preprocessed/reorder_sequence_datasets.pkl'
    with open(training_data_path, 'rb') as f:
        data = pickle.load(f)
        reorder_id_sequences = data['reorder_id_sequences']
        reorder_category_sequences = data['reorder_category_sequences']

    block_id_path = '../../../datasets/blocks_273.json'
    with open(block_id_path, 'r') as file:
        block_id_json = json.load(file)
    id_dictionary = create_id_name_dict(block_id_json)

    output_texts = []
    for reorder_id_sequence, reorder_category_sequence in zip(tqdm(reorder_id_sequences), reorder_category_sequences):
        input_texts = []
        input_dict = {}
        for id, category in zip(reorder_id_sequence, reorder_category_sequence):
            category_name = category
            try:
                block_name = id_dictionary[id].replace('minecraft:', '').replace('_block', '') + '_block'
            except:
                block_name = 'undetermined'

            if category_name not in input_dict:
                input_dict[category_name] = []
            if block_name not in input_dict[category_name]:
                input_dict[category_name].append(block_name)

        n_block = len(reorder_id_sequence)
        output_text = f'This house is composed of almost <{n_block}> blocks. This house consists of '

        input_dict = sorted(input_dict.items(), key=lambda x: x[0])
        for idx, data in enumerate(input_dict):
            key, value = data
            output_text += f'<{key}> made of <'
            for jdx, v in enumerate(value):
                if jdx == len(value) - 1:
                    output_text += f'{v}>, '
                else:
                    output_text += f'{v}, '
        output_text = output_text[:-2]
        output_text += '.'
        print(output_text)
        output_texts.append(output_text)

    with open('../../../datasets/preprocessed/text_sequence_datasets.pkl', 'wb') as f:
        pickle.dump({'texts': output_texts}, f)
    print(max(output_texts, key=len), len(max(output_texts, key=len)))
    print(min(output_texts, key=len), len(min(output_texts, key=len)))
