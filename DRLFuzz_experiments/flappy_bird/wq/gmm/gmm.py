import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成随机数据
num_samples = 1000
num_features = 4
data = np.random.rand(num_samples, num_features)

# 转换为PyTorch张量
data_tensor = torch.tensor(data, dtype=torch.float32)

# 初始化高斯混合模型参数
num_clusters = 10
mu = torch.randn(num_clusters, num_features, requires_grad=True)  # 均值向量
cov = torch.randn(num_clusters, num_features, num_features, requires_grad=True)  # 协方差矩阵
pi = torch.randn(num_clusters, requires_grad=True)  # 混合系数

# EM算法迭代
max_iters = 100
optimizer = torch.optim.Adam([mu, cov, pi], lr=0.01)  # 优化器

for step in range(max_iters):
    # E 步骤：计算每个数据点属于每个高斯分布的后验概率
    logits = []
    for k in range(num_clusters):
        mvn = torch.distributions.MultivariateNormal(mu[k], covariance_matrix=cov[k])
        log_prob = mvn.log_prob(data_tensor)
        logits.append(log_prob + torch.log(pi[k]))
    logits = torch.stack(logits, dim=1)  # shape: (num_samples, num_clusters)
    log_sum_exp = torch.logsumexp(logits, dim=1, keepdim=True)  # 对数和指数求和
    gamma = torch.exp(logits - log_sum_exp)  # 后验概率

    # M 步骤：更新参数
    N_k = gamma.sum(dim=0)  # 聚类中各样本的权重和
    mu_new = torch.matmul(gamma.t(), data_tensor) / N_k.unsqueeze(-1)

    cov_new = torch.zeros(num_clusters, num_features, num_features, dtype=torch.float32)
    for k in range(num_clusters):
        diff = data_tensor - mu[k]
        weighted_diff = diff * gamma[:, k].unsqueeze(-1)
        cov_new[k] = torch.matmul(weighted_diff.transpose(1, 2), diff) / N_k[k]

        # Add a small positive value to the diagonal elements for numerical stability
        cov_new[k].fill_diagonal_(cov_new[k].diagonal() + 1e-6)

    pi_new = N_k / len(data_tensor)

    # 更新参数
    optimizer.zero_grad()
    loss = -torch.sum(log_sum_exp)  # 最大化对数似然
    loss.backward()
    optimizer.step()

    # 打印迭代信息
    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item()}")

# 可视化聚类结果
labels = torch.argmax(gamma, dim=1)

# 绘制数据点
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for k in range(num_clusters):
    data_k = data_tensor[labels == k].numpy()
    ax.scatter(data_k[:, 0], data_k[:, 1], data_k[:, 2], label=f'Cluster {k}')

# 绘制聚类中心
mu_np = mu.detach().numpy()
ax.scatter(mu_np[:, 0], mu_np[:, 1], mu_np[:, 2], marker='x', color='red', s=100, label='Cluster Centers')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()
plt.show()
