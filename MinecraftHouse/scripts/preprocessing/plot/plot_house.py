import matplotlib.pyplot as plt
import numpy as np
import os

def plot(data, folder_name, file_name):
    # 폴더 생성
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # 그래프 준비
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.jet(np.linspace(0, 1, np.max(data) + 1))

    # 0이 아닌 값에 대해 점 찍기
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            for z in range(data.shape[2]):
                if data[x, y, z] != 0:
                    ax.scatter(x, z, y, c=colors[data[x, y, z]])

    max_range = max(data.shape)
    ax.set_xlim([0, max_range])
    ax.set_ylim([0, max_range])
    ax.set_zlim([0, max_range])

    # 레이블 및 제목 설정
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Data Visualization')

    # 그래프를 이미지 파일로 저장
    plt.savefig(f'{folder_name}/{file_name}')