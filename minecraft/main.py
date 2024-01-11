import pickle
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scripts.obj_parser import parsing
from scripts.mesh_utils import mesh_generating, draw_meshs, draw_center_point

file_path = 'resources/map_data/kitpvp-map-abandoned-town/source/Mineways2Skfb_obj/Mineways2Skfb.obj'
vts, vns, vs, fs, gs, texts = parsing(file_path)

meshs = []

# original_dict = {'x_range': [], 'y_range': [], 'z_range': [], 'used_blocks': []}
# original_mesh = mesh_generating(vs.copy(), gs.copy(), fs.copy(), texts.copy(), original_dict)
# meshs.append(original_mesh)

# draw_dict1 = {'x_range': [-0.32, -0.16], 'y_range': [0.87, 100], 'z_range': [-0.01, 0.06], 'used_blocks': []}
# scatter1, pos = draw_center_point(vs.copy(), gs.copy(), fs.copy(), draw_dict1)
# meshs.append(scatter1)
# mesh1 = mesh_generating(vs.copy(), gs.copy(), fs.copy(), texts.copy(), draw_dict1)
# meshs.append(mesh1)

draw_dict1 = {'x_range': [0.206667, 0.0133333], 'y_range': [0.87, 100], 'z_range': [0.32, 0.426667], 'unused_blocks': ['Grass']}
scatter1, pos1 = draw_center_point(vs.copy(), gs.copy(), fs.copy(), draw_dict1)
meshs.append(scatter1)
# mesh1 = mesh_generating(vs.copy(), gs.copy(), fs.copy(), texts.copy(), draw_dict1)
# meshs.append(mesh1)

draw_dict2 = {'x_range': [0.3, 0.2], 'y_range': [0.87333, 100], 'z_range': [0.2, 0.266667], 'unused_blocks': ['Oak_Leaves', 'Oak_Planks', 'Oak_Log', 'Grass']}
scatter2, pos2 = draw_center_point(vs.copy(), gs.copy(), fs.copy(), draw_dict2)
meshs.append(scatter2)
# mesh2 = mesh_generating(vs.copy(), gs.copy(), fs.copy(), texts.copy(), draw_dict2)
# meshs.append(mesh2)

draw_dict3 = {'x_range': [0, 0.1], 'y_range': [0.86, 100], 'z_range': [0.18, 0.253333], 'unused_blocks': ['Oak_Leaves', 'Oak_Planks', 'Oak_Log', 'Grass']}
scatter3, pos3 = draw_center_point(vs.copy(), gs.copy(), fs.copy(), draw_dict3)
meshs.append(scatter3)
# mesh3 = mesh_generating(vs.copy(), gs.copy(), fs.copy(), texts.copy(), draw_dict3)
# meshs.append(mesh3)

draw_dict4 = {'x_range': [-0.166667, -0.306667], 'y_range': [0.88, 100], 'z_range': [0.193333, 0.3], 'unused_blocks': ['Oak_Leaves', 'Oak_Planks', 'Oak_Log', 'Grass']}
scatter4, pos4 = draw_center_point(vs.copy(), gs.copy(), fs.copy(), draw_dict4)
meshs.append(scatter4)
# mesh4 = mesh_generating(vs.copy(), gs.copy(), fs.copy(), texts.copy(), draw_dict4)
# meshs.append(mesh4)

draw_dict5 = {'x_range': [0.493333, 0.386667], 'y_range': [0.87, 100], 'z_range': [-0.146667, 0.0466667], 'unused_blocks': ['Oak_Leaves', 'Oak_Planks', 'Oak_Log', 'Grass']}
scatter5, pos5 = draw_center_point(vs.copy(), gs.copy(), fs.copy(), draw_dict5)
meshs.append(scatter5)
# mesh5 = mesh_generating(vs.copy(), gs.copy(), fs.copy(), texts.copy(), draw_dict5)
# meshs.append(mesh5)

draw_dict6 = {'x_range': [0.16, 0.286667], 'y_range': [0.87, 100], 'z_range': [0.12, 0.00666667], 'unused_blocks': ['Oak_Leaves', 'Oak_Planks', 'Oak_Log', 'Grass']}
scatter6, pos6 = draw_center_point(vs.copy(), gs.copy(), fs.copy(), draw_dict6)
meshs.append(scatter6)
# mesh6 = mesh_generating(vs.copy(), gs.copy(), fs.copy(), texts.copy(), draw_dict6)
# meshs.append(mesh6)

draw_dict7 = {'x_range': [0.1, 0.00666667], 'y_range': [0.86, 100], 'z_range': [-0.1, 0.0266667], 'unused_blocks': ['Oak_Leaves', 'Oak_Planks', 'Oak_Log', 'Grass', 'Dandelion']}
scatter7, pos7 = draw_center_point(vs.copy(), gs.copy(), fs.copy(), draw_dict7)
meshs.append(scatter7)
# mesh7 = mesh_generating(vs.copy(), gs.copy(), fs.copy(), texts.copy(), draw_dict7)
# meshs.append(mesh7)

draw_dict8 = {'x_range': [-0.18, -0.273333], 'y_range': [0.89, 100], 'z_range': [-0.0133333, 0.06], 'unused_blocks': ['Oak_Leaves', 'Oak_Planks', 'Oak_Log', 'Grass']}
scatter8, pos8 = draw_center_point(vs.copy(), gs.copy(), fs.copy(), draw_dict8)
meshs.append(scatter8)
# mesh8 = mesh_generating(vs.copy(), gs.copy(), fs.copy(), texts.copy(), draw_dict8)
# meshs.append(mesh8)

draw_dict9 = {'x_range': [-0.22, -0.106667], 'y_range': [0.88, 100], 'z_range': [-0.146667, -0.0666667], 'unused_blocks': ['Oak_Leaves', 'Oak_Planks', 'Oak_Log', 'Grass']}
scatter9, pos9 = draw_center_point(vs.copy(), gs.copy(), fs.copy(), draw_dict9)
meshs.append(scatter9)
# mesh9 = mesh_generating(vs.copy(), gs.copy(), fs.copy(), texts.copy(), draw_dict9)
# meshs.append(mesh9)

draw_dict10 = {'x_range': [0.26, 0.146667], 'y_range': [0.86, 100], 'z_range': [-0.2, -0.28], 'unused_blocks': ['Oak_Leaves', 'Oak_Planks', 'Oak_Log', 'Grass']}
scatter10, pos10 = draw_center_point(vs.copy(), gs.copy(), fs.copy(), draw_dict10)
meshs.append(scatter10)
# mesh10 = mesh_generating(vs.copy(), gs.copy(), fs.copy(), texts.copy(), draw_dict10)
# meshs.append(mesh10)

draw_dict11 = {'x_range': [0.126667, 0.0133333], 'y_range': [0.86, 100], 'z_range': [-0.246667, -0.386667], 'unused_blocks': ['Oak_Leaves', 'Oak_Planks', 'Oak_Log', 'Grass']}
scatter11, pos11 = draw_center_point(vs.copy(), gs.copy(), fs.copy(), draw_dict11)
meshs.append(scatter11)
# mesh11 = mesh_generating(vs.copy(), gs.copy(), fs.copy(), texts.copy(), draw_dict11)
# meshs.append(mesh11)

draw_dict12 = {'x_range': [-0.206667, -0.126667], 'y_range': [0.87, 100], 'z_range': [-0.36, -0.246667], 'unused_blocks': ['Oak_Leaves', 'Oak_Planks', 'Oak_Log', 'Grass']}
scatter12, pos12 = draw_center_point(vs.copy(), gs.copy(), fs.copy(), draw_dict12)
meshs.append(scatter12)
# mesh12 = mesh_generating(vs.copy(), gs.copy(), fs.copy(), texts.copy(), draw_dict12)
# meshs.append(mesh12)

pos_list = [pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9, pos10, pos11, pos12]
with open('./resources/preprocessed_data/pos_list.pkl', 'wb') as file:
    pickle.dump(pos_list, file)
draw_meshs(meshs)
