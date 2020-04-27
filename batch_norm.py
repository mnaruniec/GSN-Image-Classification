import torch

from torch.nn import Parameter


class BatchNorm(torch.nn.Module):
    eps = 1e-8

    def __init__(self, parameter_shape, sum_dimensions):
        super().__init__()
        self.parameter_shape = parameter_shape
        self.sum_dimensions = sum_dimensions
        self.alpha = Parameter(torch.ones(*parameter_shape), requires_grad=True)
        self.beta = Parameter(torch.zeros(*parameter_shape), requires_grad=True)

    def get_effective_batch_size(self, x):
        result = 1
        for dim in self.sum_dimensions:
            result *= x.shape[dim]
        return result

    def forward(self, x):
        eff_batch_size = self.get_effective_batch_size(x)

        mean = torch.sum(x, dim=self.sum_dimensions, keepdim=True)
        mean /= eff_batch_size

        var = (x - mean) ** 2
        var /= eff_batch_size

        assert mean.shape == self.parameter_shape
        assert var.shape == self.parameter_shape

        normalized = (x - mean) / torch.sqrt(var + self.eps)

        assert normalized.shape == self.parameter_shape

        return self.alpha * normalized + self.beta


class BatchNorm1d(BatchNorm):
    def __init__(self, num_features):
        super().__init__(parameter_shape=(1, num_features), sum_dimensions=[0])


class BatchNorm2d(BatchNorm):
    def __init__(self, num_features):
        super().__init__(parameter_shape=(1, num_features, 1, 1), sum_dimensions=[0, 2, 3])
