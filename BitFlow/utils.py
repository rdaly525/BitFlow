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
                r, g, b = np.array(r).flatten(), np.array(
                    g).flatten(), np.array(b).flatten()

                inputs["r"].append(r)
                inputs["g"].append(g)
                inputs["b"].append(b)

                for i in range(len(r)):
                    outputs.append([torch.tensor(0.299 * r[i] + 0.587 * g[i] + 0.144 * b[i]),
                                    torch.tensor(-0.16875 *
                                                 r[i] + -0.33126 * g[i] + 0.5 * b[i]),
                                    torch.tensor(0.5 * r[i] + -0.41869 * g[i] + -0.08131 * b[i])])
        for key in inputs:
            inputs[key] = torch.flatten(torch.Tensor(inputs[key]))

        return Dataset(inputs, outputs), len(inputs["r"])


class Dataset(data.Dataset):
    def __init__(self, inputs, outputs):
        self.X = inputs
        self.Y = outputs

    def __len__(self):
        return len(self.X[list(self.X.keys())[0]])

    def __getitem__(self, index):
        return {k: self.X[k][index] for k in self.X}, self.Y[index]
