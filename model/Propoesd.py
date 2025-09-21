import torch
import torch.nn as nn
from parser_my import args


class MyNet(nn.Module):
    def __init__(self, input_channel=10, hidden_channel=32, hidden_size=32, output_size=1, batch_first=True, bidirectional=False):
        super(MyNet, self).__init__()
        self.input_channel = input_channel
        self.hidden_channel = hidden_channel
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        self.stem = nn.Conv1d(in_channels=self.input_channel, out_channels=32, kernel_size=5, padding=2)

        self.conv1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1,
                               padding=0)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5,
                               padding=2)

        self.conv = nn.Conv1d(in_channels=4*32, out_channels=self.hidden_channel, kernel_size=5,
                              padding=2)
        self.relu = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=self.hidden_channel, out_channels=self.hidden_channel, kernel_size=5,
                               padding=2)

        self.GRU = nn.GRU(input_size=self.hidden_channel, hidden_size=self.hidden_size,
                          batch_first=self.batch_first, bidirectional=False, dropout=0)

        self.dense = nn.Linear(64, 64)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(64 * args.sequence_length, output_size)
        self.Leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.relu(self.stem(out))
        out_1 = self.relu(self.conv1(out))
        out_2 = self.relu(self.conv2(out_1))
        out_3 = self.relu(self.conv3(out_2))
        out = self.relu(self.conv(torch.cat((out, out_1, out_2, out_3), dim=1)))

        out_4 = (out+self.relu(self.conv4(out))).permute(0, 2, 1)  # [64, 14, 32]
        out_5, _ = self.GRU(out.permute(0, 2, 1))  # [64, 14, 32]
        out_5 = self.tanh(out_5) + out.permute(0, 2, 1)
        # out_5 = out_5 + out.permute(0, 2, 1)

        out = torch.cat((out_4, out_5), dim=-1)  # [64, 14, 64]
        weights = self.sigmoid(self.dense(out.mean(dim=1)).unsqueeze(1))
        out = out + out * weights

        out = self.fc(out.contiguous().view(out.size(0), -1))

        return out


class model_A(nn.Module):
    def __init__(self, input_channel=10, hidden_channel=32, hidden_size=32, output_size=1, batch_first=True, bidirectional=False):
        super(model_A, self).__init__()
        self.input_channel = input_channel
        self.hidden_channel = hidden_channel
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        self.stem = nn.Conv1d(in_channels=self.input_channel, out_channels=32, kernel_size=5, padding=2)

        self.conv1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1,
                               padding=0)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5,
                               padding=2)

        self.conv = nn.Conv1d(in_channels=32, out_channels=self.hidden_channel, kernel_size=5,
                              padding=2)
        self.relu = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=self.hidden_channel, out_channels=self.hidden_channel, kernel_size=5,
                               padding=2)

        self.GRU = nn.GRU(input_size=self.hidden_channel, hidden_size=self.hidden_size,
                          batch_first=self.batch_first, bidirectional=False, dropout=0)

        self.dense = nn.Linear(64, 64)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(64 * args.sequence_length, output_size)
        self.Leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.relu(self.stem(out))
        out = self.relu(self.conv(out))

        out_4 = self.relu(self.conv4(out)).permute(0, 2, 1)  # [64, 14, 32]
        out_5, _ = self.GRU(out.permute(0, 2, 1))  # [64, 14, 32]
        out_5 = self.tanh(out_5)

        out = torch.cat((out_4, out_5), dim=-1)  # [64, 14, 64]

        out = self.fc(out.contiguous().view(out.size(0), -1))

        return out


class model_B(nn.Module):
    def __init__(self, input_channel=10, hidden_channel=32, hidden_size=32, output_size=1, batch_first=True, bidirectional=False):
        super(model_B, self).__init__()
        self.input_channel = input_channel
        self.hidden_channel = hidden_channel
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        self.stem = nn.Conv1d(in_channels=self.input_channel, out_channels=32, kernel_size=5, padding=2)

        self.conv1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1,
                               padding=0)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5,
                               padding=2)

        self.conv = nn.Conv1d(in_channels=4*32, out_channels=self.hidden_channel, kernel_size=5,
                              padding=2)
        self.relu = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=self.hidden_channel, out_channels=self.hidden_channel, kernel_size=5,
                               padding=2)

        self.GRU = nn.GRU(input_size=self.hidden_channel, hidden_size=self.hidden_size,
                          batch_first=self.batch_first, bidirectional=False, dropout=0)

        self.dense = nn.Linear(64, 64)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(64 * args.sequence_length, output_size)
        self.Leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.relu(self.stem(out))
        out_1 = self.relu(self.conv1(out))
        out_2 = self.relu(self.conv2(out_1))
        out_3 = self.relu(self.conv3(out_2))
        out = self.relu(self.conv(torch.cat((out, out_1, out_2, out_3), dim=1)))

        out_4 = self.relu(self.conv4(out)).permute(0, 2, 1)  # [64, 14, 32]
        out_5, _ = self.GRU(out.permute(0, 2, 1))  # [64, 14, 32]
        out_5 = self.tanh(out_5)

        out = torch.cat((out_4, out_5), dim=-1)  # [64, 14, 64]

        out = self.fc(out.contiguous().view(out.size(0), -1))

        return out


class model_C(nn.Module):
    def __init__(self, input_channel=10, hidden_channel=32, hidden_size=32, output_size=1, batch_first=True, bidirectional=False):
        super(model_C, self).__init__()
        self.input_channel = input_channel
        self.hidden_channel = hidden_channel
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        self.stem = nn.Conv1d(in_channels=self.input_channel, out_channels=32, kernel_size=5, padding=2)

        self.conv1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1,
                               padding=0)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5,
                               padding=2)

        self.conv = nn.Conv1d(in_channels=4*32, out_channels=self.hidden_channel, kernel_size=5,
                              padding=2)
        self.relu = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=self.hidden_channel, out_channels=self.hidden_channel, kernel_size=5,
                               padding=2)

        self.GRU = nn.GRU(input_size=self.hidden_channel, hidden_size=self.hidden_size,
                          batch_first=self.batch_first, bidirectional=False, dropout=0)

        self.dense = nn.Linear(64, 64)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(64 * args.sequence_length, output_size)
        self.Leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.relu(self.stem(out))
        out_1 = self.relu(self.conv1(out))
        out_2 = self.relu(self.conv2(out_1))
        out_3 = self.relu(self.conv3(out_2))
        out = self.relu(self.conv(torch.cat((out, out_1, out_2, out_3), dim=1)))

        out_4 = (out+self.relu(self.conv4(out))).permute(0, 2, 1)  # [64, 14, 32]
        out_5, _ = self.GRU(out.permute(0, 2, 1))  # [64, 14, 32]
        out_5 = self.tanh(out_5) + out.permute(0, 2, 1)

        out = torch.cat((out_4, out_5), dim=-1)  # [64, 14, 64]

        out = self.fc(out.contiguous().view(out.size(0), -1))

        return out

if __name__ == "__main__":
    batch_size = 64
    sequence_length = 28
    feature_dim = 10

    # 创建输入数据
    input = torch.randn(batch_size, sequence_length, feature_dim)


    # 初始化模型
    model = model_B(input_channel=args.input_size, hidden_channel=args.hidden_channel, hidden_size=args.hidden_size,
                  output_size=args.output_size, batch_first=args.batch_first, bidirectional=args.bidirectional)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params}')

    # 前向传播
    output = model(input)

    print("输入形状:", input.shape)
    print("输出形状:", output.shape)
