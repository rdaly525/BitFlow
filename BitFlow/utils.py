from .node import Input, Output, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select, LookupTable

from PIL import Image
import numpy as np
import os

import graphviz

import torch
from torch.utils import data

from DagVisitor import Visitor
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

import matplotlib.pyplot as plt


class DAGGrapher(Visitor):
    def __init__(self, allroots):
        self.G = nx.DiGraph()
        self.labels = {}
        self.node_colors = []
        self.allroots = allroots

    def maintainGraph(self, node, color, symbol):
        self.G.add_node(node.name)
        self.labels[node.name] = symbol

        children = self.getChildren(node)
        self.node_colors.append(color)

        for c in children:
            self.G.add_edge(c, node.name)

        if (node in self.allroots):
            self.G.add_node(node.name + "_out")
            self.node_colors.append('#30336b')
            self.labels[node.name + "_out"] = "⊙"
            self.G.add_edge(node.name, node.name + "_out")

    def getChildren(self, node):
        children = []
        for child_node in node.children():
            children.append(child_node.name)
        if len(children) == 1:
            return children[0]
        return children

    def generic_visit(self, node: DagNode):
        Visitor.generic_visit(self, node)
        self.maintainGraph(node, '#34495e', " ")

    def visit_Input(self, node: Input):
        Visitor.generic_visit(self, node)
        self.maintainGraph(node, 'black', node.name)

    def visit_Add(self, node: Add):
        Visitor.generic_visit(self, node)
        self.maintainGraph(node, '#c0392b', "+")

    def visit_Mul(self, node: Mul):
        Visitor.generic_visit(self, node)
        self.maintainGraph(node, '#2980b9', "✕")

    def visit_Constant(self, node: Constant):
        Visitor.generic_visit(self, node)
        self.maintainGraph(node, '#7f8c8d', "C")

    def visit_Sub(self, node: Sub):
        Visitor.generic_visit(self, node)
        self.maintainGraph(node, '#27ae60', "—")

    def visit_LookupTable(self, node: LookupTable):
        Visitor.generic_visit(self, node)
        self.maintainGraph(
            node, '#8e44ad', node.func.__name__.replace('np.', "") + "(x)")

    def visit_Round(self, node: Round):
        Visitor.generic_visit(self, node)
        self.maintainGraph(node, '#e67e22', " ")

    def visit_Select(self, node: Select):
        Visitor.generic_visit(self, node)
        self.maintainGraph(node, '#d35400', "W")

    def draw(self):
        pos = graphviz_layout(self.G, prog='dot')

        options = {
            'node_color':  self.node_colors,
            'node_size': 800,
            'width': 3,
            'arrowstyle': '-|>',
            'arrowsize': 12,
            'font_color': 'white',
            'labels': self.labels
        }

        nx.draw_networkx(self.G, pos=pos, arrows=True, **options)

        plt.show()


class LUTGenerator:
    def __init__(self, func, domain, numel=25):
        lut = {}
        delta = (domain[1] - domain[0])/(1. * numel - 1)
        x = domain[0]
        while x <= domain[1] + delta/2.:
            lut[x] = func(x)
            x = x + delta
        self.lut = lut
        self.domain = domain
        self.func = func

    def regen(self, numel, domain):
        lut = {}
        delta = (domain[1] - domain[0])/(1. * numel - 1)
        x = domain[0]
        while x <= domain[1] + delta/2.:
            lut[x] = self.func(x)
            x = x + delta
        self.lut = lut
        self.domain = domain

    def __getitem__(self, x):
        # print(f"Looking for {torch.sin(x)}...")
        if isinstance(x, torch.Tensor):
            x = torch.clamp(x, self.domain[0], self.domain[1])
            if torch.numel(x) == 1:
                closest = min(self.lut.keys(), key=lambda true: abs(true-x))
                # print(f"Found {self.lut[closest]}...")
                return self.lut[closest]
            for (ind, val) in enumerate(x):
                closest = min(self.lut.keys(), key=lambda true: abs(true-val))
                x[ind] = self.lut[closest]
            # print(f"Found {x}...")
            return x

        else:
            x = np.clip(x, self.domain[0], self.domain[1])
            closest = min(self.lut.keys(), key=lambda true: abs(true-x))
            # print(f"Found {self.lut[closest]}...")
            return self.lut[closest]

    def __str__(self):
        out = "{\n"
        for key in self.lut:
            out += f"\t{key} : {self.lut[key]},\n"
        out += "}"
        return out


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

        return DatasetWrapper(inputs, outputs), len(inputs["r"])


class DatasetWrapper(data.Dataset):
    def __init__(self, inputs, outputs):
        self.X = inputs
        self.Y = outputs

    def __len__(self):
        return len(self.X[list(self.X.keys())[0]])

    def __getitem__(self, index):
        return {k: self.X[k][index] for k in self.X}, self.Y[index]


class GeneratedDataset(data.Dataset):
    def __init__(self, model, dataset_size, size_p, size_r, size_output, data_range, true_width, dist):
        self.train_MNIST = True
        self.X = {k: [] for k in data_range}
        self.Y = []
        self.data_range = data_range

        P = torch.Tensor(1, size_p).fill_(true_width)[0]
        R = torch.Tensor(1, size_r).fill_(true_width)[0]
        torch.manual_seed(42)

        for key in data_range:
            # Create random tensor
            input_range = data_range[key]
            if isinstance(input_range, list):
                self.X[key] = (input_range[1] - input_range[0]) * \
                    torch.rand(dataset_size, input_range) + input_range[0]
            else:
                val = 0
                if dist == 1:
                    mean = (input_range[1]+input_range[0])/2
                    std = (mean - input_range[0])/3
                    val = torch.normal(
                        mean=mean, std=std, size=(1, dataset_size)).squeeze()
                elif dist == 2:
                    beta = torch.distributions.beta.Beta(
                        torch.tensor([0.5]), torch.tensor([0.5]))
                    val = (input_range[1] - input_range[0]) * \
                        beta.sample((dataset_size,)).squeeze() + \
                        input_range[0]
                else:
                    val = (input_range[1] - input_range[0]) * \
                        torch.rand(dataset_size) + input_range[0]

                self.X[key] = val

        for i in range(dataset_size):
            inputs = {k: self.X[k][i] for k in data_range}

            inputs["P"] = P
            inputs["R"] = R
            inputs["O"] = torch.Tensor(
                1, size_output).fill_(true_width)[0]
            new_y = model(**inputs)
            self.Y.append(new_y)

    def __len__(self):
        return len(self.X[list(self.data_range.keys())[0]])

    def __getitem__(self, index):
        return {k: self.X[k][index] for k in self.data_range}, self.Y[index]