import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from VariationalAutoEncoder import VariationalAutoEncoder

from tqdm import tqdm

device = ("cuda:0" if torch.cuda.is_available() else "cpu")

# Load training data

BATCH_SIZE = 64

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
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

loss_mse = nn.MSELoss()
loss_kld = nn.KLDivLoss()
# Training the model

EPOCHS = 5

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
        recon_loss = loss_mse(x_sample, inputs)
        
        # kl divergence loss
        kld_loss = loss_kld(x_sample, inputs)
        
        # total loss
        loss = recon_loss + kld_loss
                
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch {} / {}".format(epoch + 1, EPOCHS))

print("Total Loss: {}".format(total_loss))

# Save model state
torch.save(autoencoder.state_dict(), 
           "../model-states/VAE-[{}-Epochs]".format(EPOCHS))
