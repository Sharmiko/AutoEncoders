import torch
import torchvision
import torchvision.transforms as transforms
from VariationalAutoEncoder import VariationalAutoEncoder
import matplotlib.pyplot as plt

device = ("cuda:0" if torch.cuda.is_available() else "cpu")

# Load Testing data dataset

BATCH_SIZE = 32

transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])

testset = torchvision.datasets.MNIST(root='../data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)


# Loading the model
autoencoder = VariationalAutoEncoder()
autoencoder.load_state_dict(torch.load(
        "../model-states/VAE-[5-Epochs]"))
autoencoder.to(device)


# Extract images and labes from testloader
images, labels = next(iter(testloader))

# Generate images without grad attribute
with torch.no_grad():
    images = images.to(device)
    encoded, mu, logvar = autoencoder(images)
    
    
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

imshow(images[0:6].cpu(), labels[0:6], encoded[0:6].cpu())

