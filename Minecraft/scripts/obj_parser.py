import numpy as np
from tqdm import tqdm

def parsing(file_path):
    vn_list = []
    vt_list = []
    v_list = []
    f_list = []
    g_list = {}
    text_list = []
    cur_g = ''

    with open(file_path) as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.replace('\n', '')
            split = line.split()
            if len(split) == 0:
                continue

            if split[0] == 'vn':
                vn_list.append(split[1:])
            elif split[0] == 'vt':
                vt_list.append(split[1:])
            elif split[0] == 'v':
                v_list.append(split[1:])
            elif split[0] == 'g':
                cur_g = split[1]
                g_list[cur_g] = [len(f_list), len(f_list)]
            elif split[0] == 'f':
                f_split = []
                for i in range(1, len(split)):
                    f_split.append(split[i].replace('-', '').split('/'))
                f_list.append(f_split)
                g_list[cur_g][1] += 1
                text_list.append(cur_g)

        vn_list = np.array(vn_list).astype(float)
        print(f'vn_list shape: {vn_list.shape}')    # vn_list shape: (210, 3)
        vt_list = np.array(vt_list).astype(float)
        print(f'vt_list shape: {vt_list.shape}')    # vt_list shape: (1888, 2)
        v_list = np.array(v_list).astype(float)
        print(f'v_list shape: {v_list.shape}')      # v_list shape: (3754643, 3)
        f_list = np.array(f_list).astype(int) - 1
        print(f'f_list shape: {f_list.shape}')      # f_list shape: (4809774, 4, 3)
        text_list = np.array(text_list)
        print(f'text_list shape: {text_list.shape}')

        # 각 범위의 길이 계산
        range_lengths = {item[0]: item[1][1] - item[1][0] for item in g_list.items()}
        sorted_range_lengths = sorted(range_lengths.items(), key=lambda x: x[1], reverse=True)
        print(f'g_list: {g_list}')
        print("Value 기준으로 정렬된 범위의 길이:", sorted_range_lengths)

        return vn_list, vt_list, v_list, f_list, g_list, text_list