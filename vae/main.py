from __future__ import print_function

import os 
import copy
import time
import argparse
import numpy as np
from PIL import Image

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

import utils
from vgg import Vgg16
from net import VAE


def vgg_style(data, vgg):
    style = vgg(utils.normalize_batch(data))
    gram_style = [utils.gram_matrix(i) for i in style]
    return gram_style


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, beta):
    MSE_list = [ F.mse_loss(recon_x[i], x[i], reduction='sum') for i in range(len(x))]
    MSE = sum(MSE_list)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD =  -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (MSE + beta*KLD) / len(x) 


def train(epoch, train_loader, model, vgg, optimizer, args):    
    print('Train : Epoch {0}'.format(str(epoch)))
    print('# of training data: {0}'.format(len(train_loader.dataset)))
    model.train()
    cet = 0
    for batch_idx, (x,_) in enumerate(train_loader):
        start = time.time()
        optimizer.zero_grad()
        x = utils.preprocess(x)
        x = x.to(args.device)
        
        x_gram_style = vgg_style(x, vgg)
        y, mu, logvar, mu_s, logvar_s = model(x, x_gram_style) 
        y_gram_style = vgg_style(y, vgg)

        loss_recon = loss_function([y], [x], mu, logvar, args.beta)
        loss_style = loss_function(y_gram_style, x_gram_style, mu_s, logvar_s, args.beta)
        loss = loss_recon + args.alpha*loss_style
       
        loss.backward()
        optimizer.step()
        end = time.time()
        elapsed_time = end - start
        cet += elapsed_time
        remaining_time = elapsed_time/len(x) * (len(train_loader.dataset)-(batch_idx+1)*len(x))
        if batch_idx % args.log_interval == 0:
            print('{} ({:.0f}%)\tloss: {:.0f}\t {:.0f}s\t {:.0f}s'.format(
                batch_idx * len(x), 100. * batch_idx / len(train_loader),
                loss.item(), cet, remaining_time ))
    return model, vgg


def test(epoch, test_loader, model, vgg, args):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (x,_) in enumerate(test_loader):
            x = utils.preprocess(x)
            x = x.to(args.device)
            
            x_gram_style = vgg_style(x, vgg)
            y, mu, logvar, mu_s, logvar_s = model(x, x_gram_style)    
            y_gram_style = vgg_style(y, vgg)
            
            loss_recon = loss_function([y], [x], mu, logvar, args.beta).item()
            loss_style = loss_function(y_gram_style, x_gram_style, mu_s, logvar_s, args.beta).item()
            test_loss += loss_recon + args.alpha*loss_style
            
            if i == 0:
                n = min(x.size(0), 8)
                comparison = torch.cat([x[:n], y[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(args.alpha)+'_'+str(args.beta)+'_'+str(epoch) + '.png', nrow=n)
    test_loss /= len(test_loader)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


def stylize(model, vgg, args):
    style_img = utils.load_image(args.style_file).unsqueeze(0).to(args.device)
    content_img = utils.load_image(args.content_file).unsqueeze(0).to(args.device)
    model.eval()
    with torch.no_grad():
        #Image Generation from noise
        sample = torch.randn(args.batch_size, 1024).to(args.device)
        sample = model.decoder(sample).cpu() 
        save_image(sample, 'results/sample.png')
        
        #Image Style Transfer
        gram_style = vgg_style(style_img, vgg)
        y,_,_,_,_ = model(content_img, gram_style)    
        z, z_s = model.representation(content_img, gram_style)
        y_control_content = [0]*10
        y_control_style = [0]*10
        for i in range(10):
            tmp = copy.deepcopy(z)
            tmp[0,0] = 10*i
            tmp = model.fc3(tmp)
            z_ss = model.fc3_s(z_s)
            y_control_content[i] = model.decoder(torch.cat((tmp, z_ss),1)).cpu()
            
            tmp = copy.deepcopy(z_s)
            tmp[0,0] = 10*i
            zz = model.fc3(z)
            tmp = model.fc3_s(tmp)
            y_control_style[i] = model.decoder(torch.cat((zz, tmp),1)).cpu()
    
        comparison = torch.cat([content_img, style_img, y])
        save_image(comparison.cpu(), 'results/stylize'+str(args.alpha)+'_'+str(args.beta)+'.png')
        
        y_control_content = torch.cat(y_control_content)
        save_image(y_control_content, 'results/ycc'+str(args.beta)+'.png')
        
        y_control_style = torch.cat(y_control_style)
        save_image(y_control_style, 'results/ycs'+str(args.beta)+'.png')
        




def main():
    parser = argparse.ArgumentParser(description='beta-VAE CIFAR10 Example')
    # paths
    parser.add_argument('--model', type=str, default="results/model",
                        help='load saved model')
    parser.add_argument('--style_file', type=str, default="data/cifar10-raw/test/00/aeroplane_s_000033.png",
                        help='style image file')
    parser.add_argument('--content_file', type=str, default="data/cifar10-raw/test/01/automobile_s_001827.png",
                        help='content image file')
    # training
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--beta', type=int, default=1, metavar='N',
                        help='hyperparameter beta (default: 1)')
    parser.add_argument('--alpha', type=int, default=1, metavar='N',
                        help='hyperparameter alpha (default: 1)')
    # gpu
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # log
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
        
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    
    
    """
    SEED
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    
    """
    DATA
    """
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    
    train_loader = DataLoader(
            datasets.CIFAR10('data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(
            datasets.CIFAR10('data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
   
    
    """
    MODEL
    """
    model = VAE().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    vgg = Vgg16(requires_grad=False).to(args.device)

    
    """
    TRAIN
    """
    if 0:
        loss_best = 100000000000000000000
        for epoch in range(1, args.epochs + 1):
            model, vgg = train(epoch, train_loader, model, vgg, optimizer, args)
            loss = test(epoch, test_loader, model, vgg, args)
            if loss < loss_best:
                torch.save(model.state_dict(), args.model+ str(args.alpha)+'_'+str(args.beta)+'_'+str(epoch))
                loss_best = loss
            else:
                break
    else:
        model.load_state_dict(torch.load(args.model+str(args.alpha)+'_'+str(args.beta)+'_4'))


    """
    Image Generation & Style Transfer
    """
    stylize(model, vgg , args)

if __name__ == "__main__":
   main() 
