import os
import random

from functools import partial
from typing import Optional, Tuple

import torch
import torchvision
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from matplotlib import pyplot as plt

from batch_norm import BatchNorm1d, BatchNorm2d

DATA_PATH = 'dnn2020-1'

TRAIN_PATH = DATA_PATH + '/train'
VALID_PATH = DATA_PATH + '/valid'
TEST_PATH = DATA_PATH + '/test'

NUM_CLASSES = 28

DEFAULT_CREATE_VALID = True

DEFAULT_VALID_SIZE = 100
DEFAULT_MB_SIZE = 16
DEFAULT_STAT_PERIOD = 100

DEFAULT_NUM_EPOCHS = 15
DEFAULT_LR = 0.0001
DEFAULT_WEIGHT_DECAY = 0.001

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if not torch.cuda.is_available():
    print('WARNING! CUDA is not available - running on CPU.')


def prepare_data_dir(create_valid=DEFAULT_CREATE_VALID, valid_size=DEFAULT_VALID_SIZE) -> None:
    """ If validation dir does not exist, use stratified subset from training data. """
    if create_valid and not os.path.isdir(VALID_PATH):
        print('Validation set not found, creating.')

        os.mkdir(VALID_PATH)

        for d in os.listdir(TRAIN_PATH):
            from_path = TRAIN_PATH + '/' + d
            to_path = VALID_PATH + '/' + d

            os.mkdir(to_path)

            files = os.listdir(TRAIN_PATH + '/' + d)
            random.shuffle(files)

            for f in files[:valid_size]:
                os.rename(from_path + '/' + f, to_path + '/' + f)

    elif not create_valid and os.path.isdir(VALID_PATH):
        print('Running without validation set, but ' + VALID_PATH + 'dir found. Recreate data dir before running.')
        exit(1)


class PreprocessDataLoader(DataLoader):
    def __iter__(self):
        batches = super().__iter__()
        for b in batches:
            yield self.preprocess(*b)

    def preprocess(self, x, y):
        return x, y


class GPUDataLoader(PreprocessDataLoader):
    def preprocess(self, x, y):
        return x.float().to(DEVICE), y.to(DEVICE)


def load_dir(
        dir: str,
        mb_size,
        drop_last,
        shuffle=True,
        mean_var: Optional[Tuple[Tensor, Tensor]] = None
):
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
        GPUDataLoader(dataset, batch_size=mb_size, shuffle=shuffle, pin_memory=True, drop_last=drop_last),
        dataset.classes,
    )




class CelebrityNet(torch.nn.Module):
    def __init__(self, batch_norm=True, custom_batch_norm=True):
        super().__init__()

        # size 250
        out_channels1 = 128
        out_channels2 = 256

        out_channels3 = 256
        out_channels4 = 512
        out_lin1 = 2048

        self.layers = torch.nn.Sequential(*[
            nn.Conv2d(in_channels=3, out_channels=out_channels1, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),  # size 82
            nn.ReLU(),
            # nn.BatchNorm2d(num_features=out_channels1, track_running_stats=False),
            BatchNorm2d(num_features=out_channels1),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),  # size 26
            nn.ReLU(),
            # nn.BatchNorm2d(num_features=out_channels2, track_running_stats=False),
            BatchNorm2d(num_features=out_channels2),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(in_channels=out_channels2, out_channels=out_channels3, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),  # size 8
            nn.ReLU(),
            # nn.BatchNorm2d(num_features=out_channels3, track_running_stats=False),
            BatchNorm2d(num_features=out_channels3),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(in_channels=out_channels3, out_channels=out_channels4, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),  # size 2
            nn.ReLU(),
            # nn.BatchNorm2d(num_features=out_channels4, track_running_stats=False),
            BatchNorm2d(num_features=out_channels4),
            nn.Dropout2d(p=0.2),

            nn.Flatten(),

            nn.Linear(in_features=2 * 2 * out_channels4, out_features=out_lin1),
            nn.ReLU(),
            # nn.BatchNorm1d(num_features=out_lin1, track_running_stats=True),
            BatchNorm1d(num_features=out_lin1),
            nn.Dropout(p=0.2),

            nn.Linear(in_features=out_lin1, out_features=NUM_CLASSES)
        ])

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class CelebrityTrainer:
    def __init__(
            self,
            optimizer_lambda = partial(optim.Adam, lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY),
            mb_size=DEFAULT_MB_SIZE,
            num_epochs=DEFAULT_NUM_EPOCHS,
            has_valid=DEFAULT_CREATE_VALID,
            stat_period=DEFAULT_STAT_PERIOD,
            **net_kwargs,
    ):
        self.net_kwargs = net_kwargs

        self.optimizer_lambda = optimizer_lambda

        self.mb_size = mb_size
        self.num_epochs=num_epochs
        self.has_valid = has_valid
        self.stat_period = stat_period

        self.net = None
        self.criterion = None
        self.optimizer = None
        self.train_dl, self.valid_dl, self.test_dl = self.get_dataloaders()

    def init_net(self):
        self.net = CelebrityNet(**self.net_kwargs)
        self.net.train()
        self.net.to(DEVICE)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.optimizer_lambda(self.net.parameters())

    def get_dataloaders(self) -> (DataLoader, DataLoader, DataLoader):
        mean_var = (torch.tensor([0.47, 0.38, 0.32]), torch.tensor([0.3, 0.25, 0.25]))

        train, cl1 = load_dir(TRAIN_PATH, mean_var=mean_var, drop_last=True, mb_size=self.mb_size)
        valid, cl2 = load_dir(
            VALID_PATH if self.has_valid else TEST_PATH, mean_var=mean_var, drop_last=True, mb_size=self.mb_size
        )
        test, cl3 = load_dir(TEST_PATH, mean_var=mean_var, drop_last=False, mb_size=self.mb_size)

        assert cl1 == cl2
        assert cl2 == cl3

        return train, valid, test

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
                if not full and i >= self.stat_period:
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

        train_losses = []
        valid_losses = []
        epoch_losses = []
        epoch_x = 0
        epoch_xs = []

        try:
            for epoch in range(self.num_epochs):
                train_loss = 0.0

                for i, data in enumerate(self.train_dl, 0):
                    train_loss += self.train_batch(data)

                    if i % self.stat_period == self.stat_period - 1:
                        epoch_x += 1

                        train_loss = train_loss / self.stat_period
                        train_losses.append(train_loss)

                        acc, total, valid_loss = self.evaluate_on(self.valid_dl)

                        valid_losses.append(valid_loss)

                        print('Epoch %d, batch %d, loss: %.4f, valid loss: %.4f' %
                              (epoch + 1, i + 1, train_loss, valid_loss))

                        train_loss = 0.0

                epoch_loss = self.run_evaluation(self.valid_dl, 'VALID')
                epoch_losses.append(epoch_loss)
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
