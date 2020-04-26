import os
import random

import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import Parameter
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from matplotlib import pyplot as plt

DATA_PATH = 'dnn2020-1'

TRAIN_PATH = DATA_PATH + '/train'
VALID_PATH = DATA_PATH + '/valid'
TEST_PATH = DATA_PATH + '/test'

VALID_SIZE = 100
MB_SIZE = 40
NUM_EPOCHS = 15
LR = 0.00005
NUM_CLASSES=28
STAT_PERIOD=30

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# DEVICE = torch.device("cpu")
if not torch.cuda.is_available():
    print('WARNING! CUDA is not available - running on CPU.')


def prepare_data_dir() -> None:
    ''' If validation dir does not exist, use stratified subset from training data. '''
    if not os.path.isdir(VALID_PATH):
        os.mkdir(VALID_PATH)

        for d in os.listdir(TRAIN_PATH):
            from_path = TRAIN_PATH + '/' + d
            to_path = VALID_PATH + '/' + d

            os.mkdir(to_path)

            files = os.listdir(TRAIN_PATH + '/' + d)
            random.shuffle(files)

            for f in files[:VALID_SIZE]:
                os.rename(from_path + '/' + f, to_path + '/' + f)


def preprocess(x, y):
    return x.to(DEVICE), y.to(DEVICE)


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield self.func(*b)


def load_dir(dir: str, shuffle=True) -> DataLoader:
    # dataset = ImageFolder(dir, transform=torchvision.transforms.ToTensor())
    #
    # mean = 0
    # for img, _ in dataset:
    #     mean += torch.mean(img, dim=(1, 2), keepdim=True)
    #
    # assert mean.shape == (3, 1, 1)
    # mean /= len(dataset)
    #
    # var = 0
    # for img, _ in dataset:
    #     var += torch.sum((img - mean) ** 2, dim=(1, 2), keepdim=True) / (250 ** 2)
    #
    # assert var.shape == (3, 1, 1)
    # var /= len(dataset)
    #
    # mean = torch.squeeze(mean)
    # var = torch.squeeze(var)
    #
    # print('Source: ' + dir)
    # print('Mean: ' + str(mean))
    # print('Var: ' + str(var))

    dataset = ImageFolder(dir, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=torch.tensor([0.47, 0.38, 0.32]), std=torch.tensor([0.3, 0.25, 0.25])),
        # torchvision.transforms.Normalize(mean=mean, std=torch.sqrt(var)),
    ]))

    return WrappedDataLoader(DataLoader(dataset, batch_size=MB_SIZE, shuffle=shuffle, pin_memory=True, drop_last=True), preprocess), dataset.classes


def get_dataloaders() -> (DataLoader, DataLoader, DataLoader):
    train, cl1 = load_dir(TRAIN_PATH)
    valid, cl2 = load_dir(VALID_PATH)
    test, cl3 = load_dir(TEST_PATH)

    assert cl1 == cl2
    assert cl2 == cl3

    return train, valid, test


# class BatchNorm(torch.nn.Module):
#     eps = 1e-8
#
#     def __init__(self, parameter_shape, sum_dimensions):
#         super().__init__()
#         self.parameter_shape = parameter_shape
#         self.sum_dimensions = sum_dimensions
#         self.alpha = Parameter(torch.ones(*parameter_shape), requires_grad=True)
#         self.beta = Parameter(torch.zeros(*parameter_shape), requires_grad=True)
#
#     def get_effective_batch_size(self, x):
#         result = 1
#         for dim in self.sum_dimensions:
#             result *= x.shape[dim]
#         return result
#
#     def forward(self, x):
#         eff_batch_size = self.get_effective_batch_size(x)
#
#         mean = torch.sum(x, dim=self.sum_dimensions, keepdim=True)
#         mean /= eff_batch_size
#
#         var = (x - mean) ** 2
#         var /= eff_batch_size
#
#         assert mean.shape == self.parameter_shape
#         assert var.shape == self.parameter_shape
#
#         normalized = (x - mean) / torch.sqrt(var + self.eps)
#
#         assert normalized.shape == self.parameter_shape
#
#         return self.alpha * normalized + self.beta
#
#
# class BatchNorm1d(BatchNorm):
#     def __init__(self, num_features):
#         super().__init__(parameter_shape=(1, num_features), sum_dimensions=[0])
#
#
# class BatchNorm2d(BatchNorm):
#     def __init__(self, num_features):
#         super().__init__(parameter_shape=(1, num_features, 1, 1), sum_dimensions=[0, 2, 3])


class CelebrityNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # size 250

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=32, track_running_stats=False)

        self.pool1 = nn.MaxPool2d(kernel_size=3)  # size 83

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(num_features=128, track_running_stats=False)

        self.pool2 = nn.MaxPool2d(kernel_size=3)  # size 27

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(num_features=256, track_running_stats=False)

        self.pool3 = nn.MaxPool2d(kernel_size=2)  # size 13

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(num_features=256, track_running_stats=False)

        self.pool4 = nn.MaxPool2d(kernel_size=2)  # size 6

        self.lin1 = nn.Linear(in_features=6 * 6 * 256, out_features=2 ** 10)

        self.bn5 = nn.BatchNorm1d(num_features=2 ** 10, track_running_stats=False)

        self.lin2 = nn.Linear(in_features=2 ** 10, out_features=NUM_CLASSES)

    def forward(self, x: torch.Tensor):
        # bn -> relu
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        x = x.view(-1, 6 * 6 * 256)

        x = F.relu(self.bn5(self.lin1(x)))

        # relu -> bn
        # x = self.pool1(self.bn1(F.relu(self.conv1(x))))
        # x = self.pool2(self.bn2(F.relu(self.conv2(x))))
        # x = self.pool3(self.bn3(F.relu(self.conv3(x))))
        # x = self.pool4(self.bn4(F.relu(self.conv4(x))))
        #
        # x = x.view(-1, 6 * 6 * 128)
        #
        # x = self.bn5(F.relu(self.lin1(x)))

        return self.lin2(x)


class CelebrityTrainer:
    def __init__(self):
        self.train_dl, self.valid_dl, self.test_dl = get_dataloaders()
        self.net = None

    def init_net(self):
        self.net = CelebrityNet()
        self.net.train()
        self.net.to(DEVICE)

    def get_accuracy_on(self, dataloader: DataLoader, criterion, full=False) -> (float, int, float):
        net = self.net

        correct = 0
        total = 0

        running_loss = 0.
        i = 0

        net.eval()
        with torch.no_grad():
            for data in dataloader:
                i += 1
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                running_loss += loss.item()

                # TODO remove
                if not full and i >= STAT_PERIOD:
                    break

        net.train()

        return correct / total, total, running_loss / i

    def train(self, reset_net=True):
        if reset_net:
            self.init_net()
        net = self.net

        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=0.01)
        optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=0.01)

        train_losses=[]
        valid_losses=[]
        epoch_losses=[]
        epoch_x = 0
        epoch_xs=[]

        try:
            for epoch in range(NUM_EPOCHS):
                running_loss = 0.0
                for i, data in enumerate(self.train_dl, 0):
                    inputs, labels = data
                    optimizer.zero_grad()

                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    if i % STAT_PERIOD == STAT_PERIOD - 1:
                        epoch_x += 1

                        train_loss = running_loss / STAT_PERIOD
                        train_losses.append(train_loss)

                        acc, total, valid_loss = self.get_accuracy_on(self.valid_dl, nn.CrossEntropyLoss())

                        valid_losses.append(valid_loss)

                        print('Epoch %d, batch %d, loss: %.4f, valid loss: %.4f' %
                              (epoch + 1, i + 1, train_loss, valid_loss))
                        running_loss = 0.0


                acc, total, valid_loss = self.get_accuracy_on(self.valid_dl, nn.CrossEntropyLoss(), full=True)
                print('Validation stats: acc: %f , loss: %.4f' %(
                    100 * acc, valid_loss))
                epoch_losses.append(valid_loss)
                epoch_xs.append(epoch_x)

                acc, total, train_loss = self.get_accuracy_on(self.train_dl, nn.CrossEntropyLoss(), full=True)
                print('Training stats: acc: %f , loss: %.4f' %(
                    100 * acc, train_loss))

        finally:
            plt.plot(
                range(len(train_losses)), train_losses, 'r',
                range(len(valid_losses)), valid_losses, 'b',
                epoch_xs, epoch_losses, 'g',
            )
            plt.show()

            acc, total, test_loss = self.get_accuracy_on(self.test_dl, nn.CrossEntropyLoss(),full=True)

            print('Accuracy of the network on the {} test images: {} %'.format(
                total, 100 * acc))


def main():
    prepare_data_dir()
    trainer = CelebrityTrainer()
    trainer.train()


if __name__ == '__main__':
    main()