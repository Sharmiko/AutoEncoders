import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from VariationalAutoEncoder import VariationalAutoEncoder

from tqdm import tqdm

device = ("cuda:0" if torch.cuda.is_available() else "cpu")

# Load training data

BATCH_SIZE = 32

transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])

trainset = torchvision.datasets.MNIST(root='../data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)


# Create model
autoencoder = VariationalAutoEncoder().to(device)

# Create optimizer and loss function
optimizer = optim.Adam(autoencoder.parameters())

MSELoss = nn.MSELoss()

# Training the model

EPOCHS = 10

# training loop
for epoch in tqdm(range(EPOCHS)):

    total_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        
        # convert data to device and float data type
        inputs, labels = inputs.to(device).float(), labels.to(device)

        # zero the gradient
        optimizer.zero_grad()

        x_sample, mu, logvar = autoencoder(inputs)
        x_sample = x_sample.to(device).float()
        mu = mu.to(device).float()
        logvar = logvar.to(device).float()
        
        # reconstruction loss
        recon_loss = MSELoss(x_sample, inputs)
        
        # kl divergence loss
        kl_loss = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1.0  - logvar)
        
        # total loss
        loss = recon_loss + kl_loss
                
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch {} / {}".format(epoch + 1, EPOCHS))

print("Total Loss: {}".format(total_loss))

# Save model state
torch.save(autoencoder.state_dict(), "../model-states/VariationalAutoEncoder-[{}-Epochs]".format(EPOCHS))
