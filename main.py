import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
import matplotlib.pyplot as plt

def obj_reader(file_path):
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

def draw_model(vts, vns, vs, fs, gs, fs_range, texts):
    fs[:, :, 0] = vs.shape[0] - fs[:, :, 0] - 1
    fs = fs[fs_range, :, 0]
    texts = texts[fs_range]
    texts = np.concatenate((texts, texts), axis=0)

    # 사각형을 삼각형으로 변환
    triangles1 = fs[:, [0, 1, 2]]  # 첫 번째 삼각형
    triangles2 = fs[:, [0, 2, 3]]  # 두 번째 삼각형
    triangles = np.concatenate((triangles1, triangles2), axis=0)

    x, y, z = vs[:, 0], vs[:, 1], vs[:, 2]

    y_limit = 0.0
    triangles = np.array([triangle for triangle in triangles if y[triangle[0]] >= y_limit])

    i, j, k = triangles[:, 0], triangles[:, 1], triangles[:, 2]

    new_texts = [''] * vs.shape[0]
    for ii, jj, kk, text in zip(i, j, k, texts):
        if text not in new_texts[ii]:
            new_texts[ii] += text + ', '
        if text not in new_texts[jj]:
            new_texts[jj] += text + ', '
        if text not in new_texts[kk]:
            new_texts[kk] += text + ', '

    colors = plt.cm.viridis(y)  # Using a matplotlib colormap

    # Convert RGBA colors to HEX format
    hex_colors = ['#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in colors]

    print(x.shape, y.shape, z.shape, i.shape, j.shape, k.shape, len(new_texts), len(texts))

    # Create a mesh object
    mesh = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,  # These are the vertex indices for each face
        opacity=1,
        vertexcolor=hex_colors,
        text=new_texts
    )

    fig = go.Figure(data=[mesh])
    fig.show()

file_path = 'resources/map_data/kitpvp-map-abandoned-town/source/Mineways2Skfb_obj/Mineways2Skfb.obj'
vts, vns, vs, fs, gs, texts = obj_reader(file_path)

exception_block = ['Oak_Leaves', 'Stone']
true_block = ['Stone', 'Wool', 'Grass_Block', 'Terracotta']
fs_range = []
for block_name in gs:
    # if block_name in true_block:
    fs_range.extend(range(gs[block_name][0], gs[block_name][1]))
draw_model(vts, vns, vs, fs, gs, fs_range, texts)