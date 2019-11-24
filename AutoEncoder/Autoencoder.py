import torch.nn as nn

# Encoder Class
class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.leaky_relu = nn.LeakyReLU()

        # Output layer
        self.out = nn.Linear(7 * 7 * 64, 2)

    def forward(self, t):

        t = t

        t = self.conv1(t)
        t = self.leaky_relu(t)

        t = self.conv2(t)
        t = self.leaky_relu(t)

        t = self.conv3(t)
        t = self.leaky_relu(t)

        t = self.conv4(t)
        t = self.leaky_relu(t)

        t = t.view(-1, 7 * 7 * 64)

        t = self.out(t)

        return t

# Decoder Class
class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.input = nn.Linear(2, 64 * 7 * 7)

        # Convolutional Transpose Layers
        self.convt1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.convt2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.convt3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.convt4 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, t):
        t = t

        t = self.input(t)

        t = t.view((-1, 64, 7, 7))

        t = self.convt1(t)
        t = self.leaky_relu(t)

        t = self.convt2(t)
        t = self.leaky_relu(t)

        t = self.convt3(t)
        t = self.leaky_relu(t)

        t = self.convt4(t)

        return t

# AutoEncoder Class
class AutoEncoder(nn.Module):

    def __init__(self):

        super(AutoEncoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, t):

        encoded = self.encoder(t)
        decoded = self.decoder(encoded)

        return encoded, decoded
