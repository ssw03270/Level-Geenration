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
        for id, category in zip(reorder_id_sequence, reorder_category_sequence):
            category_name = category
            try:
                block_name = id_dictionary[id].replace('minecraft:', '').replace('_block', '').replace('_', ' ') + ' blocks'
            except:
                block_name = 'undetermined'
            input_text = f'{category_name}s made of {block_name}'
            if input_text not in input_texts and 'terrain' not in input_text:
                input_texts.append(input_text)

        output_text = 'This house consists of'
        for idx, input_text in enumerate(input_texts):
            if idx == 0:
                output_text += f' {input_text}'
            elif idx == len(input_texts) - 1:
                output_text += f' and {input_text}.'
            else:
                output_text += f', {input_text}'

        output_texts.append(output_text)

    with open('../../../datasets/preprocessed/text_sequence_datasets.pkl', 'wb') as f:
        pickle.dump({'texts': output_texts}, f)
    print(max(output_texts, key=len), len(max(output_texts, key=len)))
    print(min(output_texts, key=len), len(min(output_texts, key=len)))
