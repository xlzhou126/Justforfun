import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim

batch_size = 64
lr = 0.01
num_epochs = 20

train_data = datasets.MNIST("./dataset", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST("./dataset", train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 图像28*28*1
        self.con1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        self.con2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),# 经历两次pooling后图片的尺寸为7*7
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.con1(x)
        x = self.con2(x)
        x = x.view(x.size(0), -1)# 将第二层的结果展开为一维
        x = self.fc(x)
        return x

model = Net()
model = model.cuda()

cost = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=lr)# 在学习线性回归的时候我自己写过一个简单的梯度下降算法，这里就直接调用了
# 训练模型
epoch = 0
for data in train_loader:
    img, label = data

    img = img.cuda()
    label = label.cuda()

    out = model(img)
    loss = cost(out, label)
    print_loss = loss.data.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch += 1
    if epoch % 50 == 0:
        print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))
'''for i in range(5):
    for data in train_loader:
        img, label = data

        img = img.cuda()
        label = label.cuda()

        out = model(img)
        loss = cost(out, label)
        print_loss = loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch += 1
        if epoch % 50 == 0:
            print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))'''

model.eval()
eval_loss = 0
eval_acc = 0
for data in test_loader:
    img, label = data

    img = img.cuda()
    label = label.cuda()

    out = model(img)
    loss = cost(out, label)
    eval_loss += loss.data.item() * label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_data)),
    eval_acc / (len(test_data))
))