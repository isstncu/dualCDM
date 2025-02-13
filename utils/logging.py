import torch
import os
import imageio
import numpy
import numpy as np

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    
    img = inverse_data_transform(img)
    img = img * 10000

    img = img.squeeze().permute(1, 2, 0).cpu().numpy()
    
    img = img.astype(np.uint16) 
    imageio.imwrite(file_directory, img)

def save_checkpoint(state, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename + '.pth.tar')


def load_checkpoint(path, device):
    if device is None:
        return torch.load(path)
    else:
        return torch.load(path, map_location=device)
