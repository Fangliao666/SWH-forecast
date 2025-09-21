import numpy as np
import random
from torch.autograd import Variable
import torch.nn as nn
import torch
from model.Propoesd import MyNet, model_A, model_B, model_C
from parser_my import args
from data.TVT_dataset import getData
import csv
import time
import sys
import os
from tqdm import tqdm, trange


# 设置随机种子
def set_random_seed(seed):
    """设置所有随机数生成器的种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_random_seed(1234)  # 设置固定种子以确保实验可重复

# 确保目录存在
os.makedirs(os.path.dirname(args.loss_file), exist_ok=True)

# 在训练开始前先写入表头
with open(args.loss_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Epoch', 'Train_Loss', 'Val_loss'])


# 训练循环中每轮训练后执行以下代码
def record_loss(epoch, total_loss, val_loss):
    """记录每个epoch的训练和验证损失到CSV文件"""
    with open(args.loss_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch, total_loss, val_loss])


def train():
    """训练模型的主函数"""
    # 初始化模型
    model = MyNet(input_channel=args.input_size,
                  hidden_channel=args.hidden_channel,
                  hidden_size=args.hidden_size,
                  output_size=args.output_size,
                  batch_first=args.batch_first,
                  bidirectional=args.bidirectional)

    model.to(args.device)
    criterion = nn.MSELoss()  # 定义损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)  # Adam优化器

    # 获取预处理后的数据
    Test_WVHT_max, Test_WVHT_min, train_loader, val_loader, test_loader = getData(
        args.corpusFile, args.sequence_length, args.batch_size)

    best_val_loss = float('inf')
    epoch_losses = []  # 初始化列表用于记录每个epoch的总损失
    start_time = time.time()  # 记录训练开始时间

    # 训练循环
    for i in range(args.epochs):
        # 训练阶段
        train_loss = 0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for idx, (data, label) in enumerate(train_bar):
            if args.useGPU:
                data1 = data.squeeze(1).cuda()
                pred = model(Variable(data1).cuda())
                label = label.unsqueeze(1).cuda()
            else:
                data1 = data.squeeze(1)
                pred = model(Variable(data1))
                label = label.unsqueeze(1)

            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_bar.desc = f"训练 epoch[{i + 1}/{args.epochs}] train_loss:{train_loss:.3f}"

        # 验证阶段
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for idx, (x, target) in enumerate(val_bar):
                if args.useGPU:
                    x = x.squeeze(1).cuda()  # batch_size,seq_len,input_size
                    target = target.unsqueeze(1).cuda()
                else:
                    x = x.squeeze(1)
                    target = target.unsqueeze(1)
                output = model(x)
                loss = criterion(output, target)
                val_loss += loss.item()
                val_bar.desc = f"验证 epoch[{i + 1}/{args.epochs}] val_loss:{val_loss:.3f}"

        # 确保保存模型的目录存在
        os.makedirs(os.path.dirname(args.save_best_file), exist_ok=True)
        os.makedirs(os.path.dirname(args.save_final_file), exist_ok=True)

        # 检查验证损失是否改善
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'state_dict': model.state_dict()}, args.save_best_file)
            print(f'第{i}个epoch，验证损失改善至{val_loss:.4f}，已保存最佳模型')

        # 记录损失到CSV文件
        record_loss(i + 1, train_loss, val_loss)

    # 保存最终模型
    torch.save({'state_dict': model.state_dict()}, args.save_final_file)
    training_time = time.time() - start_time
    print(f"训练完成，总耗时: {training_time:.2f}秒")


if __name__ == "__main__":
    train()
