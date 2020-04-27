import os
import random

from typing import Optional, Tuple

import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from matplotlib import pyplot as plt

DATA_PATH = 'dnn2020-1'

TRAIN_PATH = DATA_PATH + '/train'
VALID_PATH = DATA_PATH + '/valid'
TEST_PATH = DATA_PATH + '/test'

NUM_CLASSES = 28

VALID_SIZE = 100
MB_SIZE = 30
STAT_PERIOD = 30

NUM_EPOCHS = 15
LR = 0.0001
WEIGHT_DECAY = 0.01

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if not torch.cuda.is_available():
    print('WARNING! CUDA is not available - running on CPU.')


def prepare_data_dir() -> None:
    ''' If validation dir does not exist, use stratified subset from training data. '''
    if not os.path.isdir(VALID_PATH):
        print('Validation set not found, creating.')

        os.mkdir(VALID_PATH)

        for d in os.listdir(TRAIN_PATH):
            from_path = TRAIN_PATH + '/' + d
            to_path = VALID_PATH + '/' + d

            os.mkdir(to_path)

            files = os.listdir(TRAIN_PATH + '/' + d)
            random.shuffle(files)

            for f in files[:VALID_SIZE]:
                os.rename(from_path + '/' + f, to_path + '/' + f)


class PreprocessDataLoader(DataLoader):
    def __iter__(self):
        batches = super().__iter__()
        for b in batches:
            yield self.preprocess(*b)

    def preprocess(self, x, y):
        return x, y


class GPUDataLoader(PreprocessDataLoader):
    def preprocess(self, x, y):
        return x.to(DEVICE), y.to(DEVICE)


def load_dir(dir: str, shuffle=True, drop_last=False, mean_var: Optional[Tuple[Tensor, Tensor]] = None):
    if not mean_var:
        dataset = ImageFolder(dir, transform=torchvision.transforms.ToTensor())

        mean = 0
        for img, _ in dataset:
            mean += torch.mean(img, dim=(1, 2), keepdim=True)

        assert mean.shape == (3, 1, 1)
        mean /= len(dataset)

        var = 0
        for img, _ in dataset:
            var += torch.sum((img - mean) ** 2, dim=(1, 2), keepdim=True) / (250 ** 2)

        assert var.shape == (3, 1, 1)
        var /= len(dataset)

        mean = torch.squeeze(mean)
        var = torch.squeeze(var)

        print('Dataset stats')
        print('Source: ' + dir)
        print('Mean: ' + str(mean))
        print('Var: ' + str(var))
    else:
        mean, var = mean_var

    dataset = ImageFolder(dir, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=torch.sqrt(var)),
    ]))

    return (
        GPUDataLoader(dataset, batch_size=MB_SIZE, shuffle=shuffle, pin_memory=True, drop_last=drop_last),
        dataset.classes,
    )


def get_dataloaders() -> (DataLoader, DataLoader, DataLoader):
    # TODO remove
    mean_var = (torch.tensor([0.47, 0.38, 0.32]), torch.tensor([0.3, 0.25, 0.25]))

    train, cl1 = load_dir(TRAIN_PATH, mean_var=mean_var, drop_last=True)
    valid, cl2 = load_dir(VALID_PATH, mean_var=mean_var, drop_last=True)
    test, cl3 = load_dir(TEST_PATH, mean_var=mean_var, drop_last=False)

    assert cl1 == cl2
    assert cl2 == cl3

    return train, valid, test


class CelebrityNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # size 250
        out_channels1 = 64
        out_channels2 = 128

        out_channels3 = 256
        out_channels4 = 512
        out_lin1 = 2 ** 10

        self.layers = torch.nn.Sequential(*[
            nn.Conv2d(in_channels=3, out_channels=out_channels1, kernel_size=3),
            nn.BatchNorm2d(num_features=out_channels1, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),  # size 82

            nn.Conv2d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=3),
            nn.BatchNorm2d(num_features=out_channels2, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),  # size 26

            nn.Conv2d(in_channels=out_channels2, out_channels=out_channels3, kernel_size=3),
            nn.BatchNorm2d(num_features=out_channels3, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),  # size 8

            nn.Conv2d(in_channels=out_channels3, out_channels=out_channels4, kernel_size=3),
            nn.BatchNorm2d(num_features=out_channels4, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),  # size 2

            nn.Flatten(),

            nn.Linear(in_features=2 * 2 * out_channels4, out_features=out_lin1),
            nn.BatchNorm1d(num_features=out_lin1, track_running_stats=False),
            nn.ReLU(),

            nn.Linear(in_features=out_lin1, out_features=NUM_CLASSES)
        ])

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class CelebrityTrainer:
    def __init__(self):
        self.train_dl, self.valid_dl, self.test_dl = get_dataloaders()
        self.net = None
        self.criterion = None
        self.optimizer = None

    def init_net(self):
        self.net = CelebrityNet()
        self.net.train()
        self.net.to(DEVICE)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        # self.optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=0.05)

    def evaluate_on(self, dataloader: DataLoader, full=False) -> (float, int, float):
        with torch.no_grad():
            net = self.net
            net.eval()

            correct = 0
            total = 0

            running_loss = 0.
            i = 0

            for data in dataloader:
                i += 1
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                # TODO remove
                if not full and i >= STAT_PERIOD:
                    break

        net.train()
        return correct / total, total, running_loss / i

    def run_evaluation(self, dataloader, ds_name: str):
        acc, total, loss = self.evaluate_on(dataloader, full=True)

        print(f'{ds_name} stats: acc: {(100 * acc):.2f}%, loss: {loss:.4f}')

        return loss

    def train_batch(self, data):
        inputs, labels = data

        self.net.train()
        self.optimizer.zero_grad()

        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, reset_net=True):
        if reset_net:
            self.init_net()

        train_losses=[]
        valid_losses=[]
        epoch_losses=[]
        epoch_x = 0
        epoch_xs=[]

        try:
            for epoch in range(NUM_EPOCHS):
                running_loss = 0.0
                for i, data in enumerate(self.train_dl, 0):
                    running_loss += self.train_batch(data)

                    if i % STAT_PERIOD == STAT_PERIOD - 1:
                        epoch_x += 1

                        train_loss = running_loss / STAT_PERIOD
                        train_losses.append(train_loss)

                        acc, total, valid_loss = self.evaluate_on(self.valid_dl)

                        valid_losses.append(valid_loss)

                        print('Epoch %d, batch %d, loss: %.4f, valid loss: %.4f' %
                              (epoch + 1, i + 1, train_loss, valid_loss))
                        running_loss = 0.0

                loss = self.run_evaluation(self.valid_dl, 'VALID')
                epoch_losses.append(loss)
                epoch_xs.append(epoch_x)

                self.run_evaluation(self.train_dl, 'TRAIN')

        finally:
            plt.plot(
                range(len(train_losses)), train_losses, 'r',
                range(len(valid_losses)), valid_losses, 'b',
                epoch_xs, epoch_losses, 'g',
            )
            plt.show()

            self.run_evaluation(self.test_dl, 'TEST')


def main():
    prepare_data_dir()
    trainer = CelebrityTrainer()
    trainer.train()


if __name__ == '__main__':
    main()