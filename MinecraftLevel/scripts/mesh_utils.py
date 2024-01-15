import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def range_generating(gs, draw_dict):
    fs_range = []
    for block_name in gs:
        if len(draw_dict['unused_blocks']) > 0:
            if block_name in draw_dict['unused_blocks']:
                continue
            fs_range.extend(range(gs[block_name][0], gs[block_name][1]))
        else:
            fs_range.extend(range(gs[block_name][0], gs[block_name][1]))

    return fs_range

def mesh_generating(vs, gs, fs, texts, draw_dict):
    fs_range = range_generating(gs, draw_dict)

    fs[:, :, 0] = vs.shape[0] - fs[:, :, 0] - 1
    fs = fs[fs_range, :, 0]
    texts = texts[fs_range]
    texts = np.concatenate((texts, texts), axis=0)

    # 사각형을 삼각형으로 변환
    triangles1 = fs[:, [0, 1, 2]]  # 첫 번째 삼각형
    triangles2 = fs[:, [0, 2, 3]]  # 두 번째 삼각형
    triangles = np.concatenate((triangles1, triangles2), axis=0)

    x, y, z = vs[:, 0], vs[:, 1], vs[:, 2]

    if len(draw_dict['x_range']) == 2:
        x_min = min(draw_dict['x_range'])
        x_max = max(draw_dict['x_range'])

        filtered_triangles = []
        filtered_texts = []
        for triangle, text in zip(triangles, texts):
            if x_min <= x[triangle[0]] <= x_max:
                filtered_triangles.append(triangle)
                filtered_texts.append(text)
        triangles = np.array(filtered_triangles)
        texts = np.array(filtered_texts)

    if len(draw_dict['y_range']) == 2:
        y_min = min(draw_dict['y_range'])
        y_max = max(draw_dict['y_range'])

        filtered_triangles = []
        filtered_texts = []
        for triangle, text in zip(triangles, texts):
            if y_min <= y[triangle[0]] <= y_max:
                filtered_triangles.append(triangle)
                filtered_texts.append(text)
        triangles = np.array(filtered_triangles)
        texts = np.array(filtered_texts)

    if len(draw_dict['z_range']) == 2:
        z_min = min(draw_dict['z_range'])
        z_max = max(draw_dict['z_range'])

        filtered_triangles = []
        filtered_texts = []
        for triangle, text in zip(triangles, texts):
            if z_min <= z[triangle[0]] <= z_max:
                filtered_triangles.append(triangle)
                filtered_texts.append(text)
        triangles = np.array(filtered_triangles)
        texts = np.array(filtered_texts)

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

    return mesh

def draw_meshs(meshs):
    fig = go.Figure(data=meshs)
    fig.show()

def draw_center_point(vs, gs, fs, draw_dict):
    fs_range = range_generating(gs, draw_dict)

    fs[:, :, 0] = vs.shape[0] - fs[:, :, 0] - 1
    fs = fs[fs_range, :, 0]

    quadrangles = fs[:, [0, 1, 2, 3]]

    x, y, z = vs[:, 0], vs[:, 1], vs[:, 2]

    if len(draw_dict['x_range']) == 2:
        x_min = min(draw_dict['x_range'])
        x_max = max(draw_dict['x_range'])
        quadrangles = np.array([quadrangle for quadrangle in quadrangles if x_min <= x[quadrangle[0]] <= x_max])

    if len(draw_dict['y_range']) == 2:
        y_min = min(draw_dict['y_range'])
        y_max = max(draw_dict['y_range'])
        quadrangles = np.array([quadrangle for quadrangle in quadrangles if y_min <= y[quadrangle[0]] <= y_max])
    if len(draw_dict['z_range']) == 2:
        z_min = min(draw_dict['z_range'])
        z_max = max(draw_dict['z_range'])
        quadrangles = np.array([quadrangle for quadrangle in quadrangles if z_min <= z[quadrangle[0]] <= z_max])

    mean_vs = np.mean(vs[quadrangles], axis=1)
    x, y, z = mean_vs[:, 0], mean_vs[:, 1], mean_vs[:, 2]
    print(np.array([x, y, z]).shape)

    colors = plt.cm.viridis(y)  # Using a matplotlib colormap

    # Convert RGBA colors to HEX format
    hex_colors = ['#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in colors]

    scatter = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=hex_colors
        )
    )

    return scatter, np.array([x, y, z])