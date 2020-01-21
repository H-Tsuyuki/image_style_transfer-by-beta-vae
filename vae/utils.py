import time
import pickle
import numpy as np
from PIL import Image

import torch
from torchvision import transforms


def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    trans= transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()
        ])
    return trans(img)

# 未使用
#def load_data():
#    with open("./data/cifar-10-batches-py/data_batch_1", "rb") as fp:
#        train1 = pickle.load(fp, encoding="latin-1")
#    with open("./data/cifar-10-batches-py/data_batch_2", "rb") as fp:
#        train2 = pickle.load(fp, encoding="latin-1")
#    with open("./data/cifar-10-batches-py/data_batch_3", "rb") as fp:
#        train3 = pickle.load(fp, encoding="latin-1")
#    with open("./data/cifar-10-batches-py/data_batch_4", "rb") as fp:
#        train4 = pickle.load(fp, encoding="latin-1")
#    with open("./data/cifar-10-batches-py/data_batch_5", "rb") as fp:
#        train5 = pickle.load(fp, encoding="latin-1")
#    with open("./data/cifar-10-batches-py/test_batch", "rb") as fp:
#        test = pickle.load(fp, encoding="latin-1")
#
#    train = np.concatenate((train1['data'],train2['data'],train3['data'],train4['data'],train4['data']),0)
#    return train, test['data']
#
#def save_image(filename, data):
#    data = np.swapaxes(data[0], 0, 2) ##なんか知らんけどこの2つのswapがないとImage.fromarrayでdtypeエラーが出る
#    img = np.swapaxes(data, 0, 1)
#    img = np.asarray(img)
#    scale = 255.0 / np.max(img)
#    img = np.uint8(img*scale)
#    img = Image.fromarray(img)
#    img.save(filename)


def preprocess(data):
    data = data.reshape(data.shape[0], 3, 32, 32)
    data = np.swapaxes(data, 1, 3)
    data = np.swapaxes(data, 1, 2)
    trans = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()
        ])
    data=np.asarray(data*255,dtype=np.uint8)
    p_data = torch.zeros(data.shape[0], 3, 256, 256)
    for i in range(data.shape[0]): 
        with Image.fromarray(data[i]) as img:
                p_data[i] = trans(img)
    return p_data
    

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
   # batch = batch.div_(255.0)
    return (batch - mean) / std
