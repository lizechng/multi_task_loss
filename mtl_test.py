import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Create random data
task_nums = 2
batch_size = 64
input_size = 10
label_list = [torch.randn(batch_size, 1) for _ in range(task_nums)]

class MTLLoss(nn.Module):
    def __init__(self, task_nums):
        super(MTLLoss, self).__init__()
        x = torch.zeros(task_nums, dtype=torch.float32)
        self.log_var2s = nn.Parameter(x)

    def forward(self, logit_list, label_list):
        loss = 0
        for i in range(len(self.log_var2s)):
            mse = (logit_list[i] - label_list[i]) ** 2
            pre = torch.exp(-self.log_var2s[i])
            loss += torch.sum(pre * mse + self.log_var2s[i], dim=-1)
        return torch.mean(loss)


# Define MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, task_nums)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create an instance of the MTLLoss and MLP models
mtl_loss = MTLLoss(task_nums=task_nums)
mlp_model = MLP()

# Training loop to update log_var2s
num_epochs = 100
learning_rate = 0.01
log_var2s_history = []

optimizer = torch.optim.Adam(mtl_loss.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    optimizer.zero_grad()
    logits = mlp_model(torch.randn(batch_size, input_size))
    loss = mtl_loss(logits, label_list)
    loss.backward()
    optimizer.step()
    log_var2s_history.append([param.item() for param in mtl_loss.log_var2s])

# Plot the change in log_var2s over epochs
log_var2s_history = torch.tensor(log_var2s_history)
for i in range(task_nums):
    plt.plot(log_var2s_history[:, i], label=f"log_var2s_{i + 1}")

plt.xlabel('Epoch')
plt.ylabel('log_var2s Value')
plt.legend()
plt.title('Change in log_var2s Over Epochs')
plt.show()
