import torch

from torch import nn

import batch_norm as custom

DEVICE = torch.device("cuda")

EQ_PRECISSION = 1e-5


def compare(input: torch.Tensor):
    input = input.to(DEVICE)

    if len(input.shape) == 4:
        nn_cl = nn.BatchNorm2d
        custom_cl = custom.BatchNorm2d
    elif len(input.shape) == 2:
        nn_cl = nn.BatchNorm1d
        custom_cl = custom.BatchNorm1d
    else:
        raise ValueError('You\'re in bad shape m8.')

    correct = nn_cl(num_features=input.shape[1], track_running_stats=False).to(DEVICE)(input)
    out = custom_cl(num_features=input.shape[1]).to(DEVICE)(input)

    assert correct.shape == out.shape

    diff = torch.abs(correct - out)

    m = torch.max(diff)

    return m.item()


def randu_tensor(shape, max=300):
    out = torch.rand(*shape)
    out -= 0.5
    out *= 2 * max
    return out


def randn_tensor(shape):
    out = torch.randn(*shape)
    return out


def run_tests(verbose=True):
    repeats = 10

    shapes_2d = [
        (100, 3, 250, 250),
        (81, 512, 11, 11),
        (123, 128, 81, 15),
        (100, 50, 50, 500),
        (345, 34, 21, 123),
    ]

    shapes_1d = [
        (s[0], s[1]) for s in shapes_2d
    ]

    shapes_1d += [
        (10012, 3123),
        (23423, 123),
        (123, 12323),
        (512, 9),
        (9, 512),
    ]

    shapes = shapes_1d + shapes_2d

    samplers = [
        randn_tensor,
        randu_tensor
    ]

    max_diff = 0.

    for shape in shapes:
        for repeat in range(repeats):
            for sidx, sampler in enumerate(samplers):
                diff = compare(sampler(shape))
                max_diff = max(max_diff, diff)

                if verbose:
                    print(f"{shape}, {repeat}, {sidx}:\t{diff}")

    print(f"MAX DIFFERENCE: {max_diff}")


if __name__ == "__main__":
    run_tests(verbose=True)