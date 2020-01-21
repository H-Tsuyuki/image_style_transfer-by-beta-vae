import torch
from torch import nn 
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=64):
        return input.view(input.size(0), size, 4, 4)

class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=1024, z_dim=32):
        super(VAE, self).__init__()

        # Encoder
        self.relu = nn.ReLU()
        self.conv1 = ConvLayer(image_channels, 8, kernel_size=9, stride=4)
        self.conv2 = ConvLayer(8, 16, kernel_size=5, stride=4)
        self.conv3 = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.conv4 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.flat = Flatten()

        # Bottleneck Layers using VGG Gram_feature & Content_feature 
        self.fc4_s = nn.Linear(64**2, h_dim//8)
        self.fc5_s = nn.Linear(128**2, h_dim//8)
        self.fc6_s = nn.Linear(256**2, h_dim//8)
        self.fc7_s = nn.Linear(512**2, h_dim//8)

        self.fc1_s = nn.Linear(h_dim//2, z_dim)
        self.fc2_s = nn.Linear(h_dim//2, z_dim)
        self.fc3_s = nn.Linear(z_dim, h_dim//2)
 
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim//2)
       
        # Decoder
        self.deconv1 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.deconv2 = UpsampleConvLayer(32, 16, kernel_size=3, stride=1, upsample=2)
        self.deconv3 = UpsampleConvLayer(16, 8, kernel_size=5, stride=1, upsample=4)
        self.deconv4 = UpsampleConvLayer(8, image_channels, kernel_size=9, stride=1, upsample=4)
        self.sigmoid = nn.Sigmoid()
        self.unflat = UnFlatten()
 
    def encoder(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        
        x = self.flat(x)
        #from IPython.core.debugger import Pdb; Pdb().set_trace() 
        return x

    def decoder(self, x):
        #from IPython.core.debugger import Pdb; Pdb().set_trace() 
        x = self.unflat(x)
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.deconv4(x)
        x = self.sigmoid(x)
        return x


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size(), device=0)
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def bottleneck_style(self, h):
        h0 = self.fc4_s(self.flat(h[0]))
        h1 = self.fc5_s(self.flat(h[1]))
        h2 = self.fc6_s(self.flat(h[2]))
        h3 = self.fc7_s(self.flat(h[3]))
        h = torch.cat((h0, h1, h2, h3),1) 
        mu, logvar = self.fc1_s(h), self.fc2_s(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        

    def forward(self, x, h_style):
        h = self.encoder(x) 
        z, mu, logvar = self.bottleneck(h)
        z = self.fc3(z)

        z_s, mu_s, logvar_s = self.bottleneck_style(h_style)
        z_s = self.fc3_s(z_s)	
        return self.decoder(torch.cat((z, z_s),1)), mu, logvar, mu_s, logvar_s

    def representation(self, x, h_style):
        h = self.encoder(x) 
        z,_,_ = self.bottleneck(h)
        z_s,_,_ = self.bottleneck_style(h_style)
        return z, z_s


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = F.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
