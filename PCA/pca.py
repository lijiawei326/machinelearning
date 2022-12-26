import numpy as np
from sklearn.datasets import load_iris
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def pca(X, k, method):
    """

    :param method: 'eig','svd','sklearn'
    :param X: 输入数据
    :param k: 需要保留的纬度
    :return: 降维后的数据
    """

    if method not in ['eig', 'svd', 'sklearn']:
        raise ValueError('{} don\'t exist! Only eig, svd, sklearn !'.format(str(method)))

    if method == 'eig':
        # 计算均值进行中心化
        mean = np.mean(X, axis=0)
        norm_X = X - mean

        # 计算协方差矩阵 nxn
        cov = np.cov(norm_X, rowvar=False)

        # 获取特征值和特征向量
        eig_val, eig_vec = np.linalg.eig(cov)
        # 打包后，对特征值大小进行排序
        eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(len(eig_val))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        # 取前k个特征值对应特征向量 kxn
        feature = np.array([eig_pairs[i][1] for i in range(k)])
        output = np.dot(norm_X, feature.T)
        return output

    if method == 'svd':
        mean = np.mean(X, axis=0)
        norm_X = X - mean
        V = np.linalg.svd(norm_X)[2][:k, :]
        output = np.dot(norm_X, V.T)
        return output

    if method == 'sklearn':
        model = PCA(n_components=k).fit(X)
        output = model.fit_transform(X)
        return output


def draw(data, dimension, label, label_name, method):
    color = ['red', 'blue', 'pink']
    if dimension == 2:
        for cls in set(label):
            index = np.argwhere(label == cls).squeeze()
            plt.scatter(data[index, 0], data[index, 1],
                        marker='o', c=color[cls], label=label_name[cls])
        plt.legend()

    if dimension == 3:
        fig = plt.figure()
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        for cls in set(label):
            index = np.argwhere(label == cls).squeeze()
            ax.scatter(data[index, 0], data[index, 1], data[index, 2],
                       marker='o', c=color[cls], label=label_name[cls])
        plt.legend()
    plt.savefig(f'./{dimension}d_with_{method}.png', dpi=300)


def main():
    # 载入数据
    dataset = load_iris()
    data = dataset.data
    label = dataset.target
    decomposited_dimension = 2
    method = 'sklearn'
    decomposited_data = pca(data, decomposited_dimension, method)

    label_name = dataset.target_names

    # 绘制二维散点图
    draw(decomposited_data, decomposited_dimension, label, label_name, method)


if __name__ == '__main__':
    main()
