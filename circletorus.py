import numpy as np
import matplotlib.pyplot as plt
from losses import DistTopLoss
import torch

# Circle experiment
# Generate circle and noisy circle
res = 100
angles = np.linspace(0, 2*np.pi, res)
x = np.zeros((res, 2))
x[:, 0], x[:, 1] = np.cos(angles), np.sin(angles)
y = torch.tensor(x + np.random.normal(0, 0.5, x.shape), requires_grad=True)
x = torch.tensor(x)
# Optimize the embedding
loss = DistTopLoss(torch.cdist(x, x), (0, 1), (1, 1), 25, 2)
opt = torch.optim.Adam([y], lr=1e-3)
# keep track of y throughout training
num_rounds = 3
num_iterations = 2048
ys = np.zeros((num_rounds + 1,) + y.shape)
ys[0, ...] = y.detach().numpy()
# optimize y
for i in range(num_rounds):
	for _ in range(num_iterations):
		opt.zero_grad()
		loss(y).backward()
		opt.step()
	ys[i + 1, ...] = y.detach().numpy()
# plot embeddings
plt.figure(figsize=((num_rounds+1)*5, 5), constrained_layout=True)
plt.tick_params(bottom=False, labelbottom=False)
for i in range(ys.shape[0]):
    ax = plt.subplot(1, ys.shape[0], i + 1)
    ax.scatter(ys[i, :, 0], ys[i, :, 1], c=np.arange(res), marker='o')
    ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
plt.savefig('circletorus/circle.eps')

# Torus experiment
# Generate torus and noisy torus
res = 16
x = np.zeros((res**2, 3))
u = np.linspace(0, 2*np.pi, num=res)
v = np.linspace(0, 2*np.pi, num=res)
U, V = np.meshgrid(u, v)
x[:, 0] = ((np.sin(V) + 2) * np.cos(U)).ravel()
x[:, 1] = ((np.sin(V) + 2) * np.sin(U)).ravel()
x[:, 2] = np.cos(V).ravel()
y = torch.tensor(x + np.random.normal(0, 0.5, x.shape), requires_grad=True)
x = torch.tensor(x)
# Optimize the embedding
loss = DistTopLoss(torch.cdist(x, x), (0, 1), (1, 1), 25, 2)
opt = torch.optim.Adam([y], lr=1e-3)
# keep track of y throughout training
num_rounds = 3
num_iterations = 32768
ys = np.zeros((num_rounds + 1,) + y.shape)
ys[0, ...] = y.detach().numpy()
# optimize y
for i in range(num_rounds):
	for _ in range(num_iterations):
		opt.zero_grad()
		loss(y).backward()
		opt.step()
	ys[i + 1, ...] = y.detach().numpy()
# plot embeddings
plt.figure(figsize=((num_rounds+1)*5, 5), constrained_layout=True)
plt.tick_params(bottom=False, labelbottom=False)
for i in range(ys.shape[0]):
    ax = plt.subplot(1, ys.shape[0], i + 1, projection='3d')
    ax.scatter(ys[i, :, 0], ys[i, :, 1], ys[i, :, 2],
               s=8, c=np.arange(x.shape[0]))
    ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
plt.savefig('circletorus/torus.eps')
