import torch
import torch.nn as nn

class GatedCombiner(nn.Module):
    def __init__(self, input_dim):
        super(GatedCombiner, self).__init__()
        self.gate = nn.Parameter(torch.randn(input_dim))  # 初始化 gate 参数

    def forward(self, x, mambax):
        # 使用 sigmoid 激活函数确保 gate 在 [0, 1] 范围内
        gate_weights = torch.sigmoid(self.gate)  # (input_dim,)，通过 Sigmoid 激活

        # 扩展 gate_weights 使其能与 x 和 mambax 的形状对齐
        gate_weights = gate_weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, input_dim, 1, 1)

        # 确保 gate_weights 的形状与 x 和 mambax 兼容
        gate_weights = gate_weights.expand(x.shape[0], -1, x.shape[2], x.shape[3])  # (batch_size, input_dim, H, W)

        # 选择 x 和 mambax，分别根据 gate_weights 中的值加权
        x_weighted = gate_weights * x  # 选择 x
        mambax_weighted = (1 - gate_weights) * mambax  # 选择 mambax

        # 加权融合 x 和 mambax
        combined = x_weighted + mambax_weighted

        return combined

def main():
    # 创建两个张量 x 和 mambax，假设 batch_size=8，height=32，width=32，channels=768
    batch_size = 8
    height = 32
    width = 32
    channels = 768

    # 创建随机张量
    x = torch.randn(batch_size, channels, height, width)  # (batch_size, channels, height, width)
    mambax = torch.randn(batch_size, channels, height, width)  # (batch_size, channels, height, width)

    # 初始化 GatedCombiner，假设 input_dim 为 768（通道数）
    combiner = GatedCombiner(input_dim=channels)

    # 通过 combiner 进行融合
    combined = combiner(x, mambax)

    # 打印结果
    print("x shape:", x.shape)
    print("mambax shape:", mambax.shape)
    print("Combined shape:", combined.shape)

if __name__ == "__main__":
    main()
