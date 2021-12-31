import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.utils.data import DataLoader
from torchvision.models import alexnet

from dataset import CustomImageDataset
from model1 import AlexNet1
from model2 import AlexNet2
from model3 import AlexNet3

batch_size = 32
epochs = 30

train_dataloader = DataLoader(CustomImageDataset('Train'), batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(CustomImageDataset('Test'), batch_size=batch_size, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


def train_loop(dataloader, model, loss_fn, optimizer):
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        X.to(device)
        y.to(device)
        pred = model(X.to(device))
        loss = loss_fn(pred, y.to(device))

        loss.backward()
        optimizer.step()
        total_loss += loss.item()


def test_loop(dataloader, model, mode, loss_fn):
    size = len(dataloader.dataset)
    num_batches = size / batch_size
    correct1 = 0
    correct5 = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X.to(device)
            y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y.to(device))

            _, predicted = pred.topk(1, 1, True, True)
            correct1 += predicted.eq(y.view(-1, 1).expand_as(predicted)).sum().item()
            _, predicted = pred.topk(5, 1, True, True)
            correct5 += predicted.eq(y.view(-1, 1).expand_as(predicted)).sum().item()
            total_loss += loss.item()
            total += y.size(0)

    acc1 = 100 * correct1 / total
    acc5 = 100 * correct5 / total
    avg_loss = total_loss / num_batches

    print('Accuracy of the network on the {} images: {}'.format(mode, acc1))
    print('Accuracy 5 top of the network on the {} images: {}'.format(mode, acc5))
    print(f"{mode} loss: {avg_loss:>7f}")
    return avg_loss, acc1, acc5


def training(model, num, lr=0.01):
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler1 = ExponentialLR(optimizer, gamma=0.9)
    scheduler2 = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
    print('lr = ', lr)
    train_loss = []
    test_loss = []
    test_top1 = []
    test_top5 = []
    train_top1 = []
    train_top5 = []
    eps = []

    for t in range(epochs):
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_res = test_loop(test_dataloader, model, 'test', loss_fn)
        train_res = test_loop(train_dataloader, model, 'train', loss_fn)
        test_loss.append(test_res[0])
        test_top1.append(test_res[1])
        test_top5.append(test_res[2])
        train_loss.append(train_res[0])
        train_top1.append(train_res[1])
        train_top5.append(train_res[2])
        eps.append(t + 1)
        scheduler1.step()
        scheduler2.step()

    plt.plot(eps, train_loss, label='train loss_' + num)
    plt.plot(eps, test_loss, label='test loss_' + num)
    plt.savefig('30_epochs_loss_' + num + '.jpg')
    plt.legend()
    plt.close()

    plt.plot(eps, train_top1, label='train_top1_' + num)
    plt.plot(eps, test_top1, label='test_top1_' + num)
    plt.savefig('30_epochs_top1_' + num + '.jpg')
    plt.legend()
    plt.close()

    plt.plot(eps, train_top5, label='train_top5_' + num)
    plt.plot(eps, test_top5, label='test_top5_' + num)
    plt.savefig('30_epochs_top5_' + num + '.jpg')
    plt.legend()
    plt.close()


print('############################## part 1 ##############################')
Model1 = AlexNet1()
LR = 0.01
# training(Model1, '1', LR)
print("Done!")

print('############################## part 2 ##############################')
Model2 = AlexNet2()
LR = 0.01
# training(Model2, '2', LR)
print("Done!")

print('############################## part 3 ##############################')
Model3 = AlexNet3()
LR = 0.01
training(Model3, '3', LR)
print("Done!")

print('############################## part 4 ##############################')
Model4 = alexnet(True)
for param in Model4.parameters():
    param.requires_grad = False
Model4.classifier[-1] = torch.nn.Linear(4096, 15)
LR = 0.01
training(Model4, '4', LR)
print("Done!")

print('############################## part 5 ##############################')
Model5 = alexnet(True)
Model5.classifier[-1] = torch.nn.Linear(4096, 15)
LR = 0.001
training(Model5, '5', LR)
print("Done!")
