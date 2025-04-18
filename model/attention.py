from torch import nn


class AttentionDefense(nn.Module):
    def __init__(self):
        super(AttentionDefense, self).__init__()
        self.feature_attention = nn.Sequential(  # 特征级注意力
            nn.Linear(3, 8),  # 三类距离特征
            nn.ReLU(),
            nn.Linear(8, 3),
            nn.Softmax(dim=1)
        )
        self.spatial_attention = nn.Conv1d(1, 1, kernel_size=3, padding=1)  # 维度级注意力

    def forward(self, x):
        # x: [batch_size, 3] (三度量特征)
        feat_weights = self.feature_attention(x)  # 学习特征权重
        x = x * feat_weights
        x = self.spatial_attention(x.unsqueeze(1)).squeeze(1)  # 捕捉维度异常
        return x
