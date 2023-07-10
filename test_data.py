import torchvision

train_set = torchvision.datasets.CIFAR10(
    root=r'F:\pythonProject\Datasets\CIFAR10',
    train=True,
    download=False,
)

print(train_set)
print(train_set.data.shape)
