from PIL import Image
import numpy as np
import os

import torch
from torch.utils import data


class Imgs2Dataset:
    def convert_rgb2ycbcr(self, dir):
        inputs = {"r": [], "g": [], "b": []}
        outputs = []

        for filename in os.listdir(dir):
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                fp = os.path.join(dir, filename)

                im = Image.open(fp)

                r, g, b = im.split()
                inputs["r"].append(np.array(r))
                inputs["g"].append(np.array(g))
                inputs["b"].append(np.array(b))

                im = im.convert('YCbCr')

                y, cb, cr = im.split()
                y = np.array(y).flatten().tolist()
                cb = np.array(y).flatten().tolist()
                cr = np.array(y).flatten().tolist()
                for ind in range(len(y)):
                    outputs.append([torch.tensor(y[ind]), torch.tensor(
                        cb[ind]), torch.tensor(cr[ind])])

            else:
                continue

        for key in inputs:
            inputs[key] = torch.flatten(torch.Tensor(inputs[key]))
            print(inputs[key].shape)

        print(outputs[0])
        return Dataset(inputs, outputs), len(inputs["r"])


class Dataset(data.Dataset):
    def __init__(self, inputs, outputs):
        self.X = inputs
        self.Y = outputs

    def __len__(self):
        return len(self.X[list(self.X.keys())[0]])

    def __getitem__(self, index):
        return {k: self.X[k][index] for k in self.X}, self.Y[index]
