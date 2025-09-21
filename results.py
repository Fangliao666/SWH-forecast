import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import torch
import csv
import os
from model.Propoesd import MyNet
from data.TVT_dataset import getData
from parser_my import args


def eval():
    """评估模型性能的主函数"""
    # 加载预训练模型
    model = MyNet(input_channel=args.input_size,
                  hidden_channel=args.hidden_channel,
                  hidden_size=args.hidden_size,
                  output_size=args.output_size,
                  batch_first=args.batch_first,
                  bidirectional=args.bidirectional)

    model.to(args.device)  # 将模型移到指定的设备（CPU或GPU）
    checkpoint = torch.load(args.save_best_file)  # 加载模型的状态字典
    model.load_state_dict(checkpoint['state_dict'])  # 将状态字典加载到模型中
    model.eval()  # 设置为评估模式

    preds = []  # 初始化预测值列表
    labels = []  # 初始化真实标签值列表

    # 获取训练和测试数据
    Test_WVHT_max, Test_WVHT_min, train_loader, val_dataloader, test_loader = getData(
        args.corpusFile, args.sequence_length, args.batch_size)

    # 遍历测试数据进行预测
    with torch.no_grad():
        for idx, (x, label) in enumerate(test_loader):
            if args.useGPU:
                x = x.squeeze(1).cuda()  # batch_size,seq_len,input_size
            else:
                x = x.squeeze(1)  # 调整数据维度

            pred = model(x)  # 使用模型进行预测
            pred_list = pred.data.squeeze(1).tolist()
            preds.extend(pred_list)  # 记录预测值
            labels.extend(label.tolist())  # 记录真实标签值

    # 反归一化处理
    labels = np.array(labels)
    labels = labels * (Test_WVHT_max - Test_WVHT_min) + Test_WVHT_min

    preds = np.array(preds).reshape(-1)
    preds = preds * (Test_WVHT_max - Test_WVHT_min) + Test_WVHT_min

    # 计算误差
    error = np.abs(preds - labels)
    std = np.std(error)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.forecast_file), exist_ok=True)

    # 保存预测结果到CSV文件
    np.savetxt(args.forecast_file,
               np.column_stack((labels, preds, error)),
               delimiter=',',
               header='labels,preds,error',
               comments='')

    print('预测误差的标准差:', std)
    return labels, preds


def analyze_results(actual, predict):
    """分析预测结果并计算各种评价指标"""
    print(f"数据形状: 实际值 {actual.shape}, 预测值 {predict.shape}")

    # 根据实际值划分海况
    calm_mask = actual <= 2.5
    regular_mask = (actual > 2.5) & (actual <= 5.5)
    extreme_mask = actual > 5.5

    # 计算各海况占比
    total_samples = len(actual)
    calm_percent = np.sum(calm_mask) / total_samples * 100
    regular_percent = np.sum(regular_mask) / total_samples * 100
    extreme_percent = np.sum(extreme_mask) / total_samples * 100

    # 计算常规海况均值
    regular_mean = np.mean(actual[regular_mask]) if np.any(regular_mask) else 0

    # 计算常规海况偏度
    regular_skewness = 0
    if np.any(regular_mask):
        regular_data = actual[regular_mask]
        regular_skewness = np.sum(((regular_data - np.mean(regular_data)) / np.std(regular_data)) ** 3) / len(
            regular_data)

    # 计算极端海况极端值
    extreme_max = np.max(actual[extreme_mask]) if np.any(extreme_mask) else 0

    # 计算各海况下的评价指标
    calm_metrics = calculate_metrics(actual[calm_mask], predict[calm_mask])
    regular_metrics = calculate_metrics(actual[regular_mask], predict[regular_mask])
    extreme_metrics = calculate_metrics(actual[extreme_mask], predict[extreme_mask])
    overall_metrics = calculate_metrics(actual, predict)

    # 打印结果
    print("海况分布情况：")
    print(f"平静海况 (≤2.5m): {calm_percent:.2f}%, 样本数: {np.sum(calm_mask)}")
    print(f"常规海况 (2.5~5.5m): {regular_percent:.2f}%, 均值: {regular_mean:.2f}m, 样本数: {np.sum(regular_mask)}")
    print(f"极端海况 (>5.5m): {extreme_percent:.2f}%, 极端值: {extreme_max:.2f}m, 样本数: {np.sum(extreme_mask)}")

    print("\n预测精度评价：")
    print_metrics("平静海况 (≤2.5m)", calm_metrics)
    print_metrics("常规海况 (2.5~5.5m)", regular_metrics)
    print_metrics("极端海况 (>5.5m)", extreme_metrics)
    print_metrics("总体评价", overall_metrics)


def calculate_metrics(y_true, y_pred):
    """计算各种评价指标"""
    if len(y_true) == 0:
        return {"MAPE": 0, "R2": 0, "RMSE": 0, "TIC": 0, "MAE": 0, "DELTA_P": 0}

    # MAPE (平均绝对百分比误差)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # R2 (决定系数)
    r2 = r2_score(y_true, y_pred)

    # RMSE (均方根误差)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # 转换为torch张量进行计算
    y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
    y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)

    # TIC (Theil不等系数)
    tic = torch.sqrt(torch.mean((y_true_tensor - y_pred_tensor) ** 2)) / (
            torch.sqrt(torch.mean(y_true_tensor ** 2)) + torch.sqrt(torch.mean(y_pred_tensor ** 2)))

    # MAE (平均绝对误差)
    mae = torch.mean(torch.abs(y_pred_tensor - y_true_tensor))

    # DELTA_P (能量误差)
    delta_p = np.mean(np.abs(y_true ** 2 - y_pred ** 2) / (y_true ** 2)) * 100

    return {"MAPE": mape, "R2": r2, "RMSE": rmse, "TIC": tic.item(), "MAE": mae.item(), "DELTA_P": delta_p}


def print_metrics(category, metrics):
    """打印评价指标"""
    print(f"{category}:")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"  R2: {metrics['R2']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}m")
    print(f"  TIC: {metrics['TIC']:.4f}")
    print(f"  MAE: {metrics['MAE']:.4f}m")
    print(f"  DELTA_P: {metrics['DELTA_P']:.2f}%")


if __name__ == "__main__":
    # 执行模型评估
    actual, predict = eval()

    # 如果已有预测结果文件，可以直接加载
    # df = pd.read_csv(args.forecast_file)
    # actual = df.iloc[:, 0].values
    # predict = df.iloc[:, 1].values

    # 分析预测结果
    analyze_results(actual, predict)
