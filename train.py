import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from Autoencoder import AutoEncoder

import matplotlib.pyplot as plt

from tqdm import tqdm

device = ("cuda:0" if torch.cuda.is_available() else "cpu")

# Load dataset

BATCH_SIZE = 32

transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

# Create model
autoencoder = AutoEncoder()
autoencoder.to(device) # convert model to appropriate device

# Create optimizer and loss function
optimizer = optim.Adam(autoencoder.parameters())
criterion = nn.MSELoss()


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

        encoded, decoded = autoencoder(inputs)
        encoded, decoded = encoded.float(), decoded.float()

        # calculate RMSELoss using MSELoss
        loss = torch.sqrt(criterion(decoded, inputs)).float()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch {} / {}".format(epoch + 1, EPOCHS))

print("Total Loss: {}".format(total_loss))

# Save model state
torch.save(autoencoder.state_dict(), "autoencoder-state")

# Extract images and labes from trainloader
images, labels = next(iter(trainloader))

# Generate images without grad attribute
with torch.no_grad():
    images = images.to(device)
    encoded_data, decoded = autoencoder(images)

# Helper function to show images and their corresponding generated ones
def imshow(inputs, labels, outputs):
    # create subplots
    fig, axes = plt.subplots(2, 6, figsize=(13, 5))
    # plot original images
    for i in range(6):
        axes[0, i].imshow(inputs[i].view(28,28).numpy(), cmap="gray")
        axes[0, i].set_title("Original Number {}".format(labels[i]))
    
    # plot generated images
    for i in range(6):
        axes[1, i].imshow(outputs[i].view(28,28).numpy(), cmap="gray")
        axes[1, i].set_title("Encoded Number {}".format(labels[i]))

imshow(images[:6].cpu(), labels[:6], decoded[:6].cpu())


# Loading the model
autoencoder = AutoEncoder()
autoencoder.load_state_dict(torch.load("autoencoder-state"))
autoencoder.to(device)












