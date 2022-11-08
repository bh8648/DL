# <학습자료 8-1>

# 프로그램 내에서 공통적으로 활용되는 모듈을 모아 놓은 스크립트

import torch

# utils.py : load_mnist 함수
def load_mnist(is_train=True, flatten=True) :
    from torchvision import datasets, transforms

    datasets = datasets.MNIST(
        '../data', train=is_train, download=True,
        transform=transforms.Compose([
            transforms.ToTensor
        ])
    )

    x = datasets.data.float() / 255.
    y = datasets.targets

    if flatten :
        x = x.view(x.size(0), -1)

    return x, y


# utils.py : split_data 함수
def split_data(x, y, train_ratio=.8) :
    train_cnt = int(x.size(0) * train_ratio)
    valid_cnt = x.size(0) - train_cnt

    # Shuffle dataset to split into train/valid set.
    indices = torch.randperm((x.size(0)))

    x = torch.index_select(
        x,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)

    y = torch.index_select(
        y,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)

    return x, y


# utils.py : get_hidden_sizes 함수
def get_hidden_sizes(input_size, output_size, n_layers) :
    step_size = int((input_size - output_size) / n_layers)

    hidden_sizes = []
    current_size = input_size
    for i in range(n_layers - 1) :
        hidden_sizes += [current_size - step_size]
        current_size = hidden_sizes[-1]

    return hidden_sizes




