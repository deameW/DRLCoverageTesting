"""
    计算解的平均距离
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


def read_data(file_path, x):
    """
    读取文件前x行数据，每行数据形成一个列表。
    """
    data = []
    with open(file_path, 'r') as file:
        for i in range(x):
            line = file.readline().strip()
            if not line:
                break
            point = list(map(int, line.split()))
            data.append(point)
    return data


def calculate_average_distance(points):
    """
    计算点之间的平均距离。
    """
    num_points = len(points)
    total_distance = 0
    count = 0

    for i in range(num_points):
        for j in range(i + 1, num_points):
            dist = np.linalg.norm(np.array(points[i][:2]) - np.array(points[j][:2]))
            total_distance += dist
            count += 1

    if count == 0:
        return 0

    return total_distance / count


def plot_points_3d(points_3d):
    """
    在三维空间中绘制降维后的点。
    """
    sns.set(style="darkgrid")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('3D Projection of 4D Points using PCA')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
    plt.show()


def deal_wq_data(path):
    # 去除首尾的中括号并根据逗号分割成列表
    with open(path, 'r') as file:
        data = file.readline()
        # 将字符串中的列表转换为Python列表
        data_list = eval(data)

        # 将列表中的每个列表元素转换为字符串，并连接成一行
        formatted_data = '\n'.join([' '.join(map(str, sublist)) for sublist in data_list])

        # 写入新的txt文件
        output_file_path = "../result/wq_result_format.txt"  # 新文件路径
        with open(output_file_path, "w") as file:
            file.write(formatted_data)
        print("数据已成功写入到文件：", output_file_path)
        file.close()

if __name__ == '__main__':
    # data = read_data("../buaa_result.txt", 1000)
    # distance_avg = calculate_average_distance(data)
    # print("buaa 平均距离：", distance_avg)
    #
    # # 保持原始数据用于计算距离
    # original_data = data
    #
    # # 使用PCA进行降维
    # pca = PCA(n_components=3)
    # reduced_data = pca.fit_transform(original_data)
    # # 绘制降维后的点
    # plot_points_3d(reduced_data)

    deal_wq_data("../result/wq_result.txt")
    data = read_data("../result/wq_result_format.txt", 20)
    distance_avg = calculate_average_distance(data)
    print("wq 平均距离：", distance_avg)

    # 保持原始数据用于计算距离
    original_data = data

    # 使用PCA进行降维
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(original_data)
    # 绘制降维后的点
    plot_points_3d(reduced_data)
