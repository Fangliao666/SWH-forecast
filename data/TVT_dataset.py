import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms


# from parser_my import args


def getData(corpusFile, sequence_length, batchSize):
    """
    预处理数据并创建数据加载器

    参数:
        corpusFile: CSV文件路径
        sequence_length: 序列长度
        batchSize: 批次大小

    返回:
        Test_WVHT_max: 测试集波高最大值
        Test_WVHT_min: 测试集波高最小值
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    # 数据预处理，去除时间戳等对训练无用的无效数据
    SWH_data = read_csv(corpusFile)  # 从CSV文件中读取波高数据
    SWH_data.drop('Time', axis=1, inplace=True)  # 删除第一列Time

    total_len = len(SWH_data)
    print(f"数据总长度: {total_len}")

    # 数据集划分：80%训练，10%验证，10%测试
    Train = SWH_data[:int(0.8 * total_len)]
    Val = SWH_data[int(0.8 * total_len):int(0.9 * total_len)]
    Test = SWH_data[int(0.9 * total_len):]

    # 记录测试集有效波高的最大值和最小值
    Test_WVHT_max = Test['SWH'].max()
    Test_WVHT_min = Test['SWH'].min()

    # 对训练集、验证集、测试集分别进行归一化
    scaler = MinMaxScaler()  # 初始化MinMaxScaler对象

    normalized_data_train = scaler.fit_transform(Train)
    Train = pd.DataFrame(normalized_data_train, columns=Train.columns)

    normalized_data_val = scaler.fit_transform(Val)
    Val = pd.DataFrame(normalized_data_val, columns=Val.columns)

    normalized_data_test = scaler.fit_transform(Test)
    Test = pd.DataFrame(normalized_data_test, columns=Test.columns)

    # 构造特征和标签
    # 根据前n半小时的数据，预测未来半小时的波高(SWH)
    sequence = sequence_length  # 半小时数
    multi_ahead = 0  # 多步提前预测

    Train_X = []  # 训练特征
    Train_Y = []  # 训练标签
    Val_X = []  # 验证特征
    Val_Y = []  # 验证标签
    Test_X = []  # 测试特征
    Test_Y = []  # 测试标签

    # 构建训练集特征和标签
    for i in range(len(Train) - sequence - multi_ahead):
        Train_X.append(np.array(Train.iloc[i:(i + sequence), ].values, dtype=np.float32))
        Train_Y.append(np.array(Train.iloc[(i + sequence + multi_ahead), 0], dtype=np.float32))

    # 构建验证集特征和标签
    for i in range(len(Val) - sequence - multi_ahead):
        Val_X.append(np.array(Val.iloc[i:(i + sequence), ].values, dtype=np.float32))
        Val_Y.append(np.array(Val.iloc[(i + sequence + multi_ahead), 0], dtype=np.float32))

    # 构建测试集特征和标签
    for i in range(len(Test) - sequence - multi_ahead):
        Test_X.append(np.array(Test.iloc[i:(i + sequence), ].values, dtype=np.float32))
        Test_Y.append(np.array(Test.iloc[(i + sequence + multi_ahead), 0], dtype=np.float32))

    # 构建DataLoader，用于批次训练
    train_loader = DataLoader(
        dataset=Mydataset(Train_X, Train_Y, transform=transforms.ToTensor()),
        batch_size=batchSize,
        shuffle=True
    )

    val_loader = DataLoader(
        dataset=Mydataset(Val_X, Val_Y, transform=transforms.ToTensor()),
        batch_size=batchSize,
        shuffle=False
    )

    test_loader = DataLoader(
        dataset=Mydataset(Test_X, Test_Y),
        batch_size=batchSize,
        shuffle=False
    )

    return Test_WVHT_max, Test_WVHT_min, train_loader, val_loader, test_loader


class Mydataset(Dataset):
    """
    自定义数据集类，继承自torch.utils.data.Dataset
    """

    def __init__(self, xx, yy, transform=None):
        """
        初始化数据集

        参数:
            xx: 特征数据列表
            yy: 标签数据列表
            transform: 数据变换函数
        """
        self.x = xx  # 特征数据
        self.y = yy  # 标签数据
        self.tranform = transform  # 数据变换（如果有的话）

    def __getitem__(self, index):
        """
        获取指定索引的数据项

        参数:
            index: 数据索引

        返回:
            特征和标签对
        """
        x1 = self.x[index]  # 获取指定索引的特征数据
        y1 = self.y[index]  # 获取指定索引的标签数据
        if self.tranform is not None:
            return self.tranform(x1), y1  # 如果有数据变换，则应用变换
        return x1, y1  # 否则，直接返回数据

    def __len__(self):
        """返回数据集的长度"""
        return len(self.x)


if __name__ == "__main__":
    # 测试数据加载
    Test_WVHT_max, Test_WVHT_min, train_loader, val_loader, test_loader = getData(
        './data/interpolated_file51000.csv', 28, 64
    )

    # 以下代码用于调试，平时注释掉
    # for idx, (data, label) in enumerate(train_loader):
    #     print(idx)
    #     print(data.shape)
    #     print(label.shape)