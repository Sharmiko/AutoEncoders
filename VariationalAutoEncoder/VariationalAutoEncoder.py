import torch
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

        # Leaky ReLU activation function
        self.leaky_relu = nn.LeakyReLU()

        # Output layer
        self.mu = nn.Linear(7 * 7 * 64, 2)
        self.log_var = nn.Linear(7 * 7 * 64, 2)

    def forward(self, t):
        
        # (1) Input Layer
        t = t
        
        # (2) First Hidden Conv Layer
        t = self.conv1(t)
        t = self.leaky_relu(t)
        
        # (3) Second Hidden Conv Layer
        t = self.conv2(t)
        t = self.leaky_relu(t)

        # (4) Third Hidden Conv Layer
        t = self.conv3(t)
        t = self.leaky_relu(t)
        
        # (5) Forth Hidden Conv Layer
        t = self.conv4(t)
        t = self.leaky_relu(t)
        
        # (6) Flatten Layer
        t = t.view(-1, 7 * 7 * 64)
        
        # (7) Output Layer -> return mu and logvar
        return self.mu(t), self.log_var(t)

# Decoder Class
class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(2, 64 * 7 * 7)

        # Convolutional Transpose Layers
        self.convt1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.convt2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.convt3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.convt4 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, t):
        
        # (1) Input Layer
        t = t
        
        # (2) Hidden Fully Connected Layer
        t = self.fc(t)

        # (3) Reshape Layer -> reshape flat tensor into matrix
        t = t.view((-1, 64, 7, 7))

        # (4) First Hidden Conv Layer
        t = self.convt1(t)
        t = self.leaky_relu(t)
        
        # (5) Second Hidden Conv Layer
        t = self.convt2(t)
        t = self.leaky_relu(t)

        # (6) Third Hidden Conv Layer
        t = self.convt3(t)
        t = self.leaky_relu(t)
        
        # (7) Forth Hidden Conv Layer
        t = self.convt4(t)

        return t

# AutoEncoder Class
class VariationalAutoEncoder(nn.Module):

    def __init__(self):

        super(VariationalAutoEncoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def reparam(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + torch.exp(logvar / 2) * epsilon

    def forward(self, t):

       # encode
       mu, logvar = self.encoder(t)
       
       # reparametrize
       x_sample = self.reparam(mu, logvar)
       
       # decode
       out = self.decoder(x_sample)
       
       return out, mu, logvar
