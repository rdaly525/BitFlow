import torch

from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select, Output, Relu, Reduce
from BitFlow.IA import Interval
from BitFlow.Eval.TorchEval import TorchEval
from BitFlow.AddRoundNodes import AddRoundNodes
from BitFlow.casestudies.caseStudies import caseStudy
from BitFlow.Eval import IAEval, NumEval,AbstractEval

from ..node import DagNode, Select
from ..torch.functions import IntRound
import torch as t

def dot_product(a: DagNode, b:DagNode):
    common = Mul(a,b)
    return Reduce(common,name="dot product")


def matrix_multiply():
    pass

def linear_layer():
    pass
