# Step 1: Importing PyTorch and Opacus
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import numpy as np
from opacus import PrivacyEngine
from tqdm import tqdm


# Step 5: Training the private model over multiple epochs
def train(model, train_loader, optimizer, epoch, device, delta):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    for _batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    epsilon, alpha = privacy_engine.accountant.get_epsilon(delta, [5.0])
    print(
        f"Train Epoch: {epoch} \t"
        f"Loss: {np.mean(losses):.6f} "
        f"(ε = {epsilon:.2f}, δ = {delta}, alpha = {alpha})")


if __name__ == '__main__':
    # Step 2: Loading MNIST Data
    train_data = datasets.MNIST('../../data/mnist', train=True, download=True,
                                transform=transforms.Compose(
                                    [transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))]))
    subset_indices = list(range(1000))
    subset = Subset(train_data, subset_indices)
    test_data = datasets.MNIST('../../data/mnist', train=False,
                               transform=transforms.Compose(
                                   [transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))]))
    train_loader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)

    # Step 3: Creating a PyTorch Neural Network Classification Model and Optimizer
    model = torch.nn.Sequential(torch.nn.Conv2d(1, 16, 8, 2, padding=3), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 1),
                                torch.nn.Conv2d(16, 32, 4, 2), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 1),
                                torch.nn.Flatten(),
                                torch.nn.Linear(32 * 4 * 4, 32), torch.nn.ReLU(), torch.nn.Linear(32, 10))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    # Step 4: Attaching a Differential Privacy Engine to the Optimizer
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=1.2,
        max_grad_norm=1.0,
        # module=model,
        # optimizer=optimizer,
        # data_loader=train_loader,
        # target_delta=1e-5,
        # target_epsilon=5,
        # epochs=10,
        # max_grad_norm=1.0,
    )
    for epoch in range(1, 11):
        train(model, train_loader, optimizer, epoch, device="cpu", delta=1e-5)
