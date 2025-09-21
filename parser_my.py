import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--corpusFile', default='./data/interpolated_file41004.csv')
# 常改动参数
parser.add_argument('--gpu', default=0, type=int)  # gpu 卡号
parser.add_argument('--epochs', default=150, type=int)  # 训练轮数
parser.add_argument('--lr', default=0.0001, type=float)  # learning rate 学习率
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--useGPU', default=True, type=bool)  # 是否使用GPU
parser.add_argument('--dropout', default=0.2, type=float)
# 输入输出维度
parser.add_argument('--sequence_length', default=28, type=int)  # sequence的长度，默认是用前五天的数据来预测下半个小时的有效波高
parser.add_argument('--input_size', default=10, type=int)  # 输入特征的维度
parser.add_argument('--output_size', default=1, type=int)  # 输出层的维度

# GRU
parser.add_argument('--hidden_size', default=32, type=int)  # 隐藏层的维度
parser.add_argument('--batch_first', default=True, type=bool)  # 是否将batch_size放在第一维
parser.add_argument('--bidirectional', default=True, type=bool)  # 双向选择

# CNN
parser.add_argument('--hidden_channel', default=32, type=int)


# 文件路径
parser.add_argument('--save_best_file', default='Results/Proposed/best.pth')  # 最优模型保存位置
parser.add_argument('--save_final_file', default='Results/Proposed/final.pth')  # 最终模型保存位置
parser.add_argument('--loss_file', default='Results/Proposed/loss.csv')  # 模型训练损失保存位置
parser.add_argument('--forecast_file', default='Results/Proposed/Forecast.csv')  # 模型预测结果保存位置


args = parser.parse_args()

# 设置设备
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.useGPU else "cpu")
args.device = device