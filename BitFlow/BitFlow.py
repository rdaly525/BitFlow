from .node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select
from .IA import Interval
from .Eval.IAEval import IAEval
from .Eval.NumEval import NumEval
from .Eval.TorchEval import TorchEval
from .Optimization import BitFlowVisitor, BitFlowOptimizer
from .AddRoundNodes import DAGGrapher, AddRoundNodes
from .utils import GeneratedDataset


import torch
from torch.utils import data

import random
import math
import copy


from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets


from BitFlow.node import Input, Constant, Dag, Add, Sub, Mul, Round, DagNode, Select
from BitFlow.IA import Interval
from BitFlow.Eval.IAEval import IAEval
from BitFlow.Eval.NumEval import NumEval
from BitFlow.Eval.TorchEval import TorchEval

from BitFlow.AddRoundNodes import AddRoundNodes

import torch
from torch.utils import data

import random
import math


class BitFlow:

    def make_model(self, **kwargs):
        return self.evaluator.eval(**kwargs)

    def gen_model(self, dag):
        """ Sets up a given dag for torch evaluation.
        Args: Input dag (should already have Round nodes)
        Returns: A trainable model
        """
        self.transformed_dag = dag
        self.evaluator = TorchEval(dag)
        return self.make_model

    def custom_round(self, P, factor=0.5):
        P = P.tolist()
        for (index, weight) in enumerate(P):
            f, _ = math.modf(weight)
            if f < factor:
                if weight < 0:
                    P[index] = math.ceil(weight)
                else:
                    P[index] = math.floor(weight)
            else:
                if weight < 0:
                    P[index] = math.floor(weight)
                else:
                    P[index] = math.ceil(weight)
        return torch.tensor(P)

    def gen_data(self, model, dataset_size, size_p, size_r, size_output, data_range, true_width=20., dist=0):
        """ Generates ground-truth data from user specifications and model.
        Args:
            model: A dag already set up for torch evaluatiion
            output_precision: The requested number of precision bits on the output node
            num: Number of samples to generate
            size_w: The size of the weight array
            dist: Type of distribution to use
                0 ==> UNIFORM
                1 ==> NORMAL
                2 ==> ARCSINE
            mean, std: statistics for normal distribution to generate data from
        Returns:
            (X, Y): generated data
        """





        return GeneratedDataset(model, dataset_size, size_p, size_r, size_output, data_range, true_width, dist)

    def update_dag(self, dag):
        """
        Args:
            dag: Input dag
        Returns:
            updated dag: A dag which BitFlow can be run on
        """
        P = Input(name="P")
        R = Input(name="R")
        O = Input(name="O")

        rounder = AddRoundNodes(P, R, O)
        roundedDag = rounder.doit(dag)

        return roundedDag, rounder.round_count, rounder.input_count, rounder.output_count, rounder.range_count, rounder.order, rounder.area_weight

    def round_to_precision(self, num, precision):
        if len(precision) > 1:
            num = num.clone()
            scale = 2.0**precision
            for (ind, val) in enumerate(scale):
                num[ind] = num[ind] * val
            num = torch.round(num)
            for (ind, val) in enumerate(scale):
                num[ind] = num[ind] / val
            return num
        else:
            scale = 2.0**precision
            return torch.round(num * scale) / scale

    def is_within_ulp(self, num, truth, precision, should_print=False):
        r = torch.abs(num - self.round_to_precision(truth, precision))
        ulp = 2**-(precision)
        if len(precision) > 1:
            sol = torch.ones(r.shape)
            for (y, row) in enumerate(r):
                for (x, val) in enumerate(row):
                    if val > ulp[y]:
                        if should_print:
                            print(
                                f"guess: {num[y][x]}, true: {truth[y][x]}, ulp: {ulp[y]}")
                        sol[y][x] = 0
            return sol
        else:
            sol = torch.ones(r.shape)
            for (x, val) in enumerate(r):
                if val > ulp[0]:
                    if should_print:
                        print(
                            f"guess: {num[x]}, true: {truth[x]}, ulp: {ulp[0]}")
                    sol[x] = 0
            return sol

            # return(torch.where(r <= ulp, torch.ones(r.shape), torch.zeros(r.shape)))

    def ulp(self, num, truth, precision):
        r = torch.abs(num - self.round_to_precision(truth, precision))
        if len(precision) > 1:
            for (ind, val) in enumerate(precision):
                r[ind] = r[ind] * 2 ** val
            return r
        else:
            r = r * 2 ** precision
            return r

    def calc_ulp(self, name, test_gen, P, R, O, model):
        ulp_err = 0
        total = 0

        max_ulp = -math.inf

        print(f"\n##### {name} ULP ERROR ######")
        for t, (inputs, Y) in enumerate(test_gen):
            inputs["P"] = P
            inputs["R"] = R
            inputs["O"] = O

            res = model(**inputs)
            if isinstance(res, list):
                res = torch.stack(res)
                Y = torch.stack(Y).squeeze()
            else:
                Y = Y.squeeze()

            ulp = self.ulp(res, Y, O)

            if torch.max(ulp) > max_ulp:
                max_ulp = torch.max(ulp)

            ulp_err += torch.sum(ulp)
            total += torch.numel(ulp)

            # if should_print:
            #     indices = (ulp == 0).nonzero()[:, 0].tolist()
            #     for index in indices:
            #         print(
            #             f"guess: {res[index]}, true: {self.round_to_precision(Y[index], O)} ")

        avg_ulp = ulp_err/total
        print(f"AVG ERR: {avg_ulp}")
        print(f"MAX ERR: {max_ulp}")

    def calc_accuracy(self, name, test_gen, P, R, O, model, should_print):
        success = 0
        total = 0
        print(f"\n##### {name} SET ######")
        for t, (inputs, Y) in enumerate(test_gen):
            inputs["P"] = P
            inputs["R"] = R
            inputs["O"] = O

            res = model(**inputs)
            if isinstance(res, list):
                res = torch.stack(res)
                Y = torch.stack(Y).squeeze()
            else:
                Y = Y.squeeze()

            ulp = self.is_within_ulp(res, Y, O, should_print)
            success += torch.sum(ulp)
            total += torch.numel(ulp)

            # if should_print:
            #     indices = (ulp == 0).nonzero()[:, 0].tolist()
            #     for index in indices:
            #         print(
            #             f"guess: {res[index]}, true: {self.round_to_precision(Y[index], O)} ")

        acc = (success * 1.)/total
        print(f"accuracy: {acc}")

    def within_ulp_err(self, num, truth, precision):
        diff = torch.abs(truth - num)

        error = 0
        if len(truth.shape) > 1:
            error = torch.unsqueeze(2 ** -(precision + 1), 1)
            error = error.repeat(1, truth.shape[1])
        else:
            error = 2 ** -(precision + 1)

        return torch.abs(diff - error)

    def hasNegatives(self, tensor):
        vals = tensor < 0
        return vals.any()

    # def filterDAGFromOutputs(self, visitor, outputs):
    #     # Remove output variables from DAG list (these will be our weights)
    #     vars = list(visitor.node_values)
    #     for (i, var) in enumerate(vars):
    #         vars[i] = var.name
    #     filtered_vars = []
    #     for var in vars:
    #         if var not in outputs:
    #             filtered_vars.append(var)

    #     return filtered_vars

    # def calculateRange(self, evaluator, outputs):
    #     node_values = evaluator.node_values
    #     visitor = BitFlowVisitor(node_values)
    #     visitor.run(evaluator.dag)

    #     range_bits = visitor.IBs
    #     return range_bits

    def constructOptimizationFunctions(self, dag, outputs, data_range):
        evaluator = NumEval(dag)


        dag_grapher = DAGGrapher(list(evaluator.dag.roots()))
        dag_grapher.run(evaluator.dag)
        dag_grapher.draw()


        eval_dict = data_range.copy()

        for key in eval_dict:
            if self.train_MNIST:
                eval_dict = {"X": 1.,"weight": 1.,"bias":1.}
                outputs = {'1_concat_add_bias': torch.ones(1,10).fill_(10)}
                # eval_dict = {}
                # for row in range(28):
                #     for col in range(28):
                #         eval_dict[f"input_{row}_{col}"] = 1.
            elif isinstance(eval_dict[key], list):
                for (ind, val) in enumerate(eval_dict[key]):
                    eval_dict[key][ind] = int(max(abs(val[0]),
                                                  abs(val[1])))
            else:
                eval_dict[key] = int(max(abs(eval_dict[key][0]),
                                         abs(eval_dict[key][1])))
        self.eval_dict = eval_dict
        self.outputs = outputs

        print(outputs)
        if not self.train_MNIST:
            evaluator.eval(**eval_dict)

        #range_bits = self.calculateRange(evaluator, outputs)

        bfo = BitFlowOptimizer(evaluator, outputs)
        bfo.calculateInitialValues()
        return bfo



    def createExecutableConstraintFunctions(self, area_fn, error_fn, filtered_vars):

            # exec(f'''def AreaOptimizerFn(P):
            #              return  {area_fn}''', globals())
            #
            # exec(f'''def ErrorOptimizerFn(P):
            #                      return  {error_fn}''', globals())
            # #filtered_vars=['X,X_getitem_0,weight,weight_getitem_(slice(None, None, None), 0),X_getitem_0_mul_weight_getitem_(slice(None, None, None), 0),X_getitem_0_mul_weight_getitem_(slice(None, None, None), 0),weight_getitem_(slice(None, None, None), 1),X_getitem_0_mul_weight_getitem_(slice(None, None, None), 1),X_getitem_0_mul_weight_getitem_(slice(None, None, None), 1),weight_getitem_(slice(None, None, None), 2),X_getitem_0_mul_weight_getitem_(slice(None, None, None), 2),X_getitem_0_mul_weight_getitem_(slice(None, None, None), 2),weight_getitem_(slice(None, None, None), 3),X_getitem_0_mul_weight_getitem_(slice(None, None, None), 3),X_getitem_0_mul_weight_getitem_(slice(None, None, None), 3),weight_getitem_(slice(None, None, None), 4),X_getitem_0_mul_weight_getitem_(slice(None, None, None), 4),X_getitem_0_mul_weight_getitem_(slice(None, None, None), 4),weight_getitem_(slice(None, None, None), 5),X_getitem_0_mul_weight_getitem_(slice(None, None, None), 5),X_getitem_0_mul_weight_getitem_(slice(None, None, None), 5),weight_getitem_(slice(None, None, None), 6),X_getitem_0_mul_weight_getitem_(slice(None, None, None), 6),X_getitem_0_mul_weight_getitem_(slice(None, None, None), 6),weight_getitem_(slice(None, None, None), 7),X_getitem_0_mul_weight_getitem_(slice(None, None, None), 7),X_getitem_0_mul_weight_getitem_(slice(None, None, None), 7),weight_getitem_(slice(None, None, None), 8),X_getitem_0_mul_weight_getitem_(slice(None, None, None), 8),X_getitem_0_mul_weight_getitem_(slice(None, None, None), 8),weight_getitem_(slice(None, None, None), 9),X_getitem_0_mul_weight_getitem_(slice(None, None, None), 9),X_getitem_0_mul_weight_getitem_(slice(None, None, None), 9),_concat,X_getitem_1,weight_getitem_(slice(None, None, None), 0),X_getitem_1_mul_weight_getitem_(slice(None, None, None), 0),X_getitem_1_mul_weight_getitem_(slice(None, None, None), 0),weight_getitem_(slice(None, None, None), 1),X_getitem_1_mul_weight_getitem_(slice(None, None, None), 1),X_getitem_1_mul_weight_getitem_(slice(None, None, None), 1),weight_getitem_(slice(None, None, None), 2),X_getitem_1_mul_weight_getitem_(slice(None, None, None), 2),X_getitem_1_mul_weight_getitem_(slice(None, None, None), 2),weight_getitem_(slice(None, None, None), 3),X_getitem_1_mul_weight_getitem_(slice(None, None, None), 3),X_getitem_1_mul_weight_getitem_(slice(None, None, None), 3),weight_getitem_(slice(None, None, None), 4),X_getitem_1_mul_weight_getitem_(slice(None, None, None), 4),X_getitem_1_mul_weight_getitem_(slice(None, None, None), 4),weight_getitem_(slice(None, None, None), 5),X_getitem_1_mul_weight_getitem_(slice(None, None, None), 5),X_getitem_1_mul_weight_getitem_(slice(None, None, None), 5),weight_getitem_(slice(None, None, None), 6),X_getitem_1_mul_weight_getitem_(slice(None, None, None), 6),X_getitem_1_mul_weight_getitem_(slice(None, None, None), 6),weight_getitem_(slice(None, None, None), 7),X_getitem_1_mul_weight_getitem_(slice(None, None, None), 7),X_getitem_1_mul_weight_getitem_(slice(None, None, None), 7),weight_getitem_(slice(None, None, None), 8),X_getitem_1_mul_weight_getitem_(slice(None, None, None), 8),X_getitem_1_mul_weight_getitem_(slice(None, None, None), 8),weight_getitem_(slice(None, None, None), 9),X_getitem_1_mul_weight_getitem_(slice(None, None, None), 9),X_getitem_1_mul_weight_getitem_(slice(None, None, None), 9),_concat,X_getitem_2,weight_getitem_(slice(None, None, None), 0),X_getitem_2_mul_weight_getitem_(slice(None, None, None), 0),X_getitem_2_mul_weight_getitem_(slice(None, None, None), 0),weight_getitem_(slice(None, None, None), 1),X_getitem_2_mul_weight_getitem_(slice(None, None, None), 1),X_getitem_2_mul_weight_getitem_(slice(None, None, None), 1),weight_getitem_(slice(None, None, None), 2),X_getitem_2_mul_weight_getitem_(slice(None, None, None), 2),X_getitem_2_mul_weight_getitem_(slice(None, None, None), 2),weight_getitem_(slice(None, None, None), 3),X_getitem_2_mul_weight_getitem_(slice(None, None, None), 3),X_getitem_2_mul_weight_getitem_(slice(None, None, None), 3),weight_getitem_(slice(None, None, None), 4),X_getitem_2_mul_weight_getitem_(slice(None, None, None), 4),X_getitem_2_mul_weight_getitem_(slice(None, None, None), 4),weight_getitem_(slice(None, None, None), 5),X_getitem_2_mul_weight_getitem_(slice(None, None, None), 5),X_getitem_2_mul_weight_getitem_(slice(None, None, None), 5),weight_getitem_(slice(None, None, None), 6),X_getitem_2_mul_weight_getitem_(slice(None, None, None), 6),X_getitem_2_mul_weight_getitem_(slice(None, None, None), 6),weight_getitem_(slice(None, None, None), 7),X_getitem_2_mul_weight_getitem_(slice(None, None, None), 7),X_getitem_2_mul_weight_getitem_(slice(None, None, None), 7),weight_getitem_(slice(None, None, None), 8),X_getitem_2_mul_weight_getitem_(slice(None, None, None), 8),X_getitem_2_mul_weight_getitem_(slice(None, None, None), 8),weight_getitem_(slice(None, None, None), 9),X_getitem_2_mul_weight_getitem_(slice(None, None, None), 9),X_getitem_2_mul_weight_getitem_(slice(None, None, None), 9),_concat,X_getitem_3,weight_getitem_(slice(None, None, None), 0),X_getitem_3_mul_weight_getitem_(slice(None, None, None), 0),X_getitem_3_mul_weight_getitem_(slice(None, None, None), 0),weight_getitem_(slice(None, None, None), 1),X_getitem_3_mul_weight_getitem_(slice(None, None, None), 1),X_getitem_3_mul_weight_getitem_(slice(None, None, None), 1),weight_getitem_(slice(None, None, None), 2),X_getitem_3_mul_weight_getitem_(slice(None, None, None), 2),X_getitem_3_mul_weight_getitem_(slice(None, None, None), 2),weight_getitem_(slice(None, None, None), 3),X_getitem_3_mul_weight_getitem_(slice(None, None, None), 3),X_getitem_3_mul_weight_getitem_(slice(None, None, None), 3),weight_getitem_(slice(None, None, None), 4),X_getitem_3_mul_weight_getitem_(slice(None, None, None), 4),X_getitem_3_mul_weight_getitem_(slice(None, None, None), 4),weight_getitem_(slice(None, None, None), 5),X_getitem_3_mul_weight_getitem_(slice(None, None, None), 5),X_getitem_3_mul_weight_getitem_(slice(None, None, None), 5),weight_getitem_(slice(None, None, None), 6),X_getitem_3_mul_weight_getitem_(slice(None, None, None), 6),X_getitem_3_mul_weight_getitem_(slice(None, None, None), 6),weight_getitem_(slice(None, None, None), 7),X_getitem_3_mul_weight_getitem_(slice(None, None, None), 7),X_getitem_3_mul_weight_getitem_(slice(None, None, None), 7),weight_getitem_(slice(None, None, None), 8),X_getitem_3_mul_weight_getitem_(slice(None, None, None), 8),X_getitem_3_mul_weight_getitem_(slice(None, None, None), 8),weight_getitem_(slice(None, None, None), 9),X_getitem_3_mul_weight_getitem_(slice(None, None, None), 9),X_getitem_3_mul_weight_getitem_(slice(None, None, None), 9),_concat,X_getitem_4,weight_getitem_(slice(None, None, None), 0),X_getitem_4_mul_weight_getitem_(slice(None, None, None), 0),X_getitem_4_mul_weight_getitem_(slice(None, None, None), 0),weight_getitem_(slice(None, None, None), 1),X_getitem_4_mul_weight_getitem_(slice(None, None, None), 1),X_getitem_4_mul_weight_getitem_(slice(None, None, None), 1),weight_getitem_(slice(None, None, None), 2),X_getitem_4_mul_weight_getitem_(slice(None, None, None), 2),X_getitem_4_mul_weight_getitem_(slice(None, None, None), 2),weight_getitem_(slice(None, None, None), 3),X_getitem_4_mul_weight_getitem_(slice(None, None, None), 3),X_getitem_4_mul_weight_getitem_(slice(None, None, None), 3),weight_getitem_(slice(None, None, None), 4),X_getitem_4_mul_weight_getitem_(slice(None, None, None), 4),X_getitem_4_mul_weight_getitem_(slice(None, None, None), 4),weight_getitem_(slice(None, None, None), 5),X_getitem_4_mul_weight_getitem_(slice(None, None, None), 5),X_getitem_4_mul_weight_getitem_(slice(None, None, None), 5),weight_getitem_(slice(None, None, None), 6),X_getitem_4_mul_weight_getitem_(slice(None, None, None), 6),X_getitem_4_mul_weight_getitem_(slice(None, None, None), 6),weight_getitem_(slice(None, None, None), 7),X_getitem_4_mul_weight_getitem_(slice(None, None, None), 7),X_getitem_4_mul_weight_getitem_(slice(None, None, None), 7),weight_getitem_(slice(None, None, None), 8),X_getitem_4_mul_weight_getitem_(slice(None, None, None), 8),X_getitem_4_mul_weight_getitem_(slice(None, None, None), 8),weight_getitem_(slice(None, None, None), 9),X_getitem_4_mul_weight_getitem_(slice(None, None, None), 9),X_getitem_4_mul_weight_getitem_(slice(None, None, None), 9),_concat,X_getitem_5,weight_getitem_(slice(None, None, None), 0),X_getitem_5_mul_weight_getitem_(slice(None, None, None), 0),X_getitem_5_mul_weight_getitem_(slice(None, None, None), 0),weight_getitem_(slice(None, None, None), 1),X_getitem_5_mul_weight_getitem_(slice(None, None, None), 1),X_getitem_5_mul_weight_getitem_(slice(None, None, None), 1),weight_getitem_(slice(None, None, None), 2),X_getitem_5_mul_weight_getitem_(slice(None, None, None), 2),X_getitem_5_mul_weight_getitem_(slice(None, None, None), 2),weight_getitem_(slice(None, None, None), 3),X_getitem_5_mul_weight_getitem_(slice(None, None, None), 3),X_getitem_5_mul_weight_getitem_(slice(None, None, None), 3),weight_getitem_(slice(None, None, None), 4),X_getitem_5_mul_weight_getitem_(slice(None, None, None), 4),X_getitem_5_mul_weight_getitem_(slice(None, None, None), 4),weight_getitem_(slice(None, None, None), 5),X_getitem_5_mul_weight_getitem_(slice(None, None, None), 5),X_getitem_5_mul_weight_getitem_(slice(None, None, None), 5),weight_getitem_(slice(None, None, None), 6),X_getitem_5_mul_weight_getitem_(slice(None, None, None), 6),X_getitem_5_mul_weight_getitem_(slice(None, None, None), 6),weight_getitem_(slice(None, None, None), 7),X_getitem_5_mul_weight_getitem_(slice(None, None, None), 7),X_getitem_5_mul_weight_getitem_(slice(None, None, None), 7),weight_getitem_(slice(None, None, None), 8),X_getitem_5_mul_weight_getitem_(slice(None, None, None), 8),X_getitem_5_mul_weight_getitem_(slice(None, None, None), 8),weight_getitem_(slice(None, None, None), 9),X_getitem_5_mul_weight_getitem_(slice(None, None, None), 9),X_getitem_5_mul_weight_getitem_(slice(None, None, None), 9),_concat,X_getitem_6,weight_getitem_(slice(None, None, None), 0),X_getitem_6_mul_weight_getitem_(slice(None, None, None), 0),X_getitem_6_mul_weight_getitem_(slice(None, None, None), 0),weight_getitem_(slice(None, None, None), 1),X_getitem_6_mul_weight_getitem_(slice(None, None, None), 1),X_getitem_6_mul_weight_getitem_(slice(None, None, None), 1),weight_getitem_(slice(None, None, None), 2),X_getitem_6_mul_weight_getitem_(slice(None, None, None), 2),X_getitem_6_mul_weight_getitem_(slice(None, None, None), 2),weight_getitem_(slice(None, None, None), 3),X_getitem_6_mul_weight_getitem_(slice(None, None, None), 3),X_getitem_6_mul_weight_getitem_(slice(None, None, None), 3),weight_getitem_(slice(None, None, None), 4),X_getitem_6_mul_weight_getitem_(slice(None, None, None), 4),X_getitem_6_mul_weight_getitem_(slice(None, None, None), 4),weight_getitem_(slice(None, None, None), 5),X_getitem_6_mul_weight_getitem_(slice(None, None, None), 5),X_getitem_6_mul_weight_getitem_(slice(None, None, None), 5),weight_getitem_(slice(None, None, None), 6),X_getitem_6_mul_weight_getitem_(slice(None, None, None), 6),X_getitem_6_mul_weight_getitem_(slice(None, None, None), 6),weight_getitem_(slice(None, None, None), 7),X_getitem_6_mul_weight_getitem_(slice(None, None, None), 7),X_getitem_6_mul_weight_getitem_(slice(None, None, None), 7),weight_getitem_(slice(None, None, None), 8),X_getitem_6_mul_weight_getitem_(slice(None, None, None), 8),X_getitem_6_mul_weight_getitem_(slice(None, None, None), 8),weight_getitem_(slice(None, None, None), 9),X_getitem_6_mul_weight_getitem_(slice(None, None, None), 9),X_getitem_6_mul_weight_getitem_(slice(None, None, None), 9),_concat,X_getitem_7,weight_getitem_(slice(None, None, None), 0),X_getitem_7_mul_weight_getitem_(slice(None, None, None), 0),X_getitem_7_mul_weight_getitem_(slice(None, None, None), 0),weight_getitem_(slice(None, None, None), 1),X_getitem_7_mul_weight_getitem_(slice(None, None, None), 1),X_getitem_7_mul_weight_getitem_(slice(None, None, None), 1),weight_getitem_(slice(None, None, None), 2),X_getitem_7_mul_weight_getitem_(slice(None, None, None), 2),X_getitem_7_mul_weight_getitem_(slice(None, None, None), 2),weight_getitem_(slice(None, None, None), 3),X_getitem_7_mul_weight_getitem_(slice(None, None, None), 3),X_getitem_7_mul_weight_getitem_(slice(None, None, None), 3),weight_getitem_(slice(None, None, None), 4),X_getitem_7_mul_weight_getitem_(slice(None, None, None), 4),X_getitem_7_mul_weight_getitem_(slice(None, None, None), 4),weight_getitem_(slice(None, None, None), 5),X_getitem_7_mul_weight_getitem_(slice(None, None, None), 5),X_getitem_7_mul_weight_getitem_(slice(None, None, None), 5),weight_getitem_(slice(None, None, None), 6),X_getitem_7_mul_weight_getitem_(slice(None, None, None), 6),X_getitem_7_mul_weight_getitem_(slice(None, None, None), 6),weight_getitem_(slice(None, None, None), 7),X_getitem_7_mul_weight_getitem_(slice(None, None, None), 7),X_getitem_7_mul_weight_getitem_(slice(None, None, None), 7),weight_getitem_(slice(None, None, None), 8),X_getitem_7_mul_weight_getitem_(slice(None, None, None), 8),X_getitem_7_mul_weight_getitem_(slice(None, None, None), 8),weight_getitem_(slice(None, None, None), 9),X_getitem_7_mul_weight_getitem_(slice(None, None, None), 9),X_getitem_7_mul_weight_getitem_(slice(None, None, None), 9),_concat,X_getitem_8,weight_getitem_(slice(None, None, None), 0),X_getitem_8_mul_weight_getitem_(slice(None, None, None), 0),X_getitem_8_mul_weight_getitem_(slice(None, None, None), 0),weight_getitem_(slice(None, None, None), 1),X_getitem_8_mul_weight_getitem_(slice(None, None, None), 1),X_getitem_8_mul_weight_getitem_(slice(None, None, None), 1),weight_getitem_(slice(None, None, None), 2),X_getitem_8_mul_weight_getitem_(slice(None, None, None), 2),X_getitem_8_mul_weight_getitem_(slice(None, None, None), 2),weight_getitem_(slice(None, None, None), 3),X_getitem_8_mul_weight_getitem_(slice(None, None, None), 3),X_getitem_8_mul_weight_getitem_(slice(None, None, None), 3),weight_getitem_(slice(None, None, None), 4),X_getitem_8_mul_weight_getitem_(slice(None, None, None), 4),X_getitem_8_mul_weight_getitem_(slice(None, None, None), 4),weight_getitem_(slice(None, None, None), 5),X_getitem_8_mul_weight_getitem_(slice(None, None, None), 5),X_getitem_8_mul_weight_getitem_(slice(None, None, None), 5),weight_getitem_(slice(None, None, None), 6),X_getitem_8_mul_weight_getitem_(slice(None, None, None), 6),X_getitem_8_mul_weight_getitem_(slice(None, None, None), 6),weight_getitem_(slice(None, None, None), 7),X_getitem_8_mul_weight_getitem_(slice(None, None, None), 7),X_getitem_8_mul_weight_getitem_(slice(None, None, None), 7),weight_getitem_(slice(None, None, None), 8),X_getitem_8_mul_weight_getitem_(slice(None, None, None), 8),X_getitem_8_mul_weight_getitem_(slice(None, None, None), 8),weight_getitem_(slice(None, None, None), 9),X_getitem_8_mul_weight_getitem_(slice(None, None, None), 9),X_getitem_8_mul_weight_getitem_(slice(None, None, None), 9),_concat,X_getitem_9,weight_getitem_(slice(None, None, None), 0),X_getitem_9_mul_weight_getitem_(slice(None, None, None), 0),X_getitem_9_mul_weight_getitem_(slice(None, None, None), 0),weight_getitem_(slice(None, None, None), 1),X_getitem_9_mul_weight_getitem_(slice(None, None, None), 1),X_getitem_9_mul_weight_getitem_(slice(None, None, None), 1),weight_getitem_(slice(None, None, None), 2),X_getitem_9_mul_weight_getitem_(slice(None, None, None), 2),X_getitem_9_mul_weight_getitem_(slice(None, None, None), 2),weight_getitem_(slice(None, None, None), 3),X_getitem_9_mul_weight_getitem_(slice(None, None, None), 3),X_getitem_9_mul_weight_getitem_(slice(None, None, None), 3),weight_getitem_(slice(None, None, None), 4),X_getitem_9_mul_weight_getitem_(slice(None, None, None), 4),X_getitem_9_mul_weight_getitem_(slice(None, None, None), 4),weight_getitem_(slice(None, None, None), 5),X_getitem_9_mul_weight_getitem_(slice(None, None, None), 5),X_getitem_9_mul_weight_getitem_(slice(None, None, None), 5),weight_getitem_(slice(None, None, None), 6),X_getitem_9_mul_weight_getitem_(slice(None, None, None), 6),X_getitem_9_mul_weight_getitem_(slice(None, None, None), 6),weight_getitem_(slice(None, None, None), 7),X_getitem_9_mul_weight_getitem_(slice(None, None, None), 7),X_getitem_9_mul_weight_getitem_(slice(None, None, None), 7),weight_getitem_(slice(None, None, None), 8),X_getitem_9_mul_weight_getitem_(slice(None, None, None), 8),X_getitem_9_mul_weight_getitem_(slice(None, None, None), 8),weight_getitem_(slice(None, None, None), 9),X_getitem_9_mul_weight_getitem_(slice(None, None, None), 9),X_getitem_9_mul_weight_getitem_(slice(None, None, None), 9),_concat,_concat,bias,final_bias = W\n                 return  1 * (X_getitem_0 + X_getitem_0)*(weight_getitem_(slice(None, None, None), 0) + weight_getitem_(slice(None, None, None), 0))+1 * (X_getitem_0 + X_getitem_0)*(weight_getitem_(slice(None, None, None), 1) + weight_getitem_(slice(None, None, None), 1))+1 * (X_getitem_0 + X_getitem_0)*(weight_getitem_(slice(None, None, None), 2) + weight_getitem_(slice(None, None, None), 2))+1 * (X_getitem_0 + X_getitem_0)*(weight_getitem_(slice(None, None, None), 3) + weight_getitem_(slice(None, None, None), 3))+1 * (X_getitem_0 + X_getitem_0)*(weight_getitem_(slice(None, None, None), 4) + weight_getitem_(slice(None, None, None), 4))+1 * (X_getitem_0 + X_getitem_0)*(weight_getitem_(slice(None, None, None), 5) + weight_getitem_(slice(None, None, None), 5))+1 * (X_getitem_0 + X_getitem_0)*(weight_getitem_(slice(None, None, None), 6) + weight_getitem_(slice(None, None, None), 6))+1 * (X_getitem_0 + X_getitem_0)*(weight_getitem_(slice(None, None, None), 7) + weight_getitem_(slice(None, None, None), 7))+1 * (X_getitem_0 + X_getitem_0)*(weight_getitem_(slice(None, None, None), 8) + weight_getitem_(slice(None, None, None), 8))+1 * (X_getitem_0 + X_getitem_0)*(weight_getitem_(slice(None, None, None), 9) + weight_getitem_(slice(None, None, None), 9))+1 * (X_getitem_1 + X_getitem_1)*(weight_getitem_(slice(None, None, None), 0) + weight_getitem_(slice(None, None, None), 0))+1 * (X_getitem_1 + X_getitem_1)*(weight_getitem_(slice(None, None, None), 1) + weight_getitem_(slice(None, None, None), 1))+1 * (X_getitem_1 + X_getitem_1)*(weight_getitem_(slice(None, None, None), 2) + weight_getitem_(slice(None, None, None), 2))+1 * (X_getitem_1 + X_getitem_1)*(weight_getitem_(slice(None, None, None), 3) + weight_getitem_(slice(None, None, None), 3))+1 * (X_getitem_1 + X_getitem_1)*(weight_getitem_(slice(None, None, None), 4) + weight_getitem_(slice(None, None, None), 4))+1 * (X_getitem_1 + X_getitem_1)*(weight_getitem_(slice(None, None, None), 5) + weight_getitem_(slice(None, None, None), 5))+1 * (X_getitem_1 + X_getitem_1)*(weight_getitem_(slice(None, None, None), 6) + weight_getitem_(slice(None, None, None), 6))+1 * (X_getitem_1 + X_getitem_1)*(weight_getitem_(slice(None, None, None), 7) + weight_getitem_(slice(None, None, None), 7))+1 * (X_getitem_1 + X_getitem_1)*(weight_getitem_(slice(None, None, None), 8) + weight_getitem_(slice(None, None, None), 8))+1 * (X_getitem_1 + X_getitem_1)*(weight_getitem_(slice(None, None, None), 9) + weight_getitem_(slice(None, None, None), 9))+1 * (X_getitem_2 + X_getitem_2)*(weight_getitem_(slice(None, None, None), 0) + weight_getitem_(slice(None, None, None), 0))+1 * (X_getitem_2 + X_getitem_2)*(weight_getitem_(slice(None, None, None), 1) + weight_getitem_(slice(None, None, None), 1))+1 * (X_getitem_2 + X_getitem_2)*(weight_getitem_(slice(None, None, None), 2) + weight_getitem_(slice(None, None, None), 2))+1 * (X_getitem_2 + X_getitem_2)*(weight_getitem_(slice(None, None, None), 3) + weight_getitem_(slice(None, None, None), 3))+1 * (X_getitem_2 + X_getitem_2)*(weight_getitem_(slice(None, None, None), 4) + weight_getitem_(slice(None, None, None), 4))+1 * (X_getitem_2 + X_getitem_2)*(weight_getitem_(slice(None, None, None), 5) + weight_getitem_(slice(None, None, None), 5))+1 * (X_getitem_2 + X_getitem_2)*(weight_getitem_(slice(None, None, None), 6) + weight_getitem_(slice(None, None, None), 6))+1 * (X_getitem_2 + X_getitem_2)*(weight_getitem_(slice(None, None, None), 7) + weight_getitem_(slice(None, None, None), 7))+1 * (X_getitem_2 + X_getitem_2)*(weight_getitem_(slice(None, None, None), 8) + weight_getitem_(slice(None, None, None), 8))+1 * (X_getitem_2 + X_getitem_2)*(weight_getitem_(slice(None, None, None), 9) + weight_getitem_(slice(None, None, None), 9))+1 * (X_getitem_3 + X_getitem_3)*(weight_getitem_(slice(None, None, None), 0) + weight_getitem_(slice(None, None, None), 0))+1 * (X_getitem_3 + X_getitem_3)*(weight_getitem_(slice(None, None, None), 1) + weight_getitem_(slice(None, None, None), 1))+1 * (X_getitem_3 + X_getitem_3)*(weight_getitem_(slice(None, None, None), 2) + weight_getitem_(slice(None, None, None), 2))+1 * (X_getitem_3 + X_getitem_3)*(weight_getitem_(slice(None, None, None), 3) + weight_getitem_(slice(None, None, None), 3))+1 * (X_getitem_3 + X_getitem_3)*(weight_getitem_(slice(None, None, None), 4) + weight_getitem_(slice(None, None, None), 4))+1 * (X_getitem_3 + X_getitem_3)*(weight_getitem_(slice(None, None, None), 5) + weight_getitem_(slice(None, None, None), 5))+1 * (X_getitem_3 + X_getitem_3)*(weight_getitem_(slice(None, None, None), 6) + weight_getitem_(slice(None, None, None), 6))+1 * (X_getitem_3 + X_getitem_3)*(weight_getitem_(slice(None, None, None), 7) + weight_getitem_(slice(None, None, None), 7))+1 * (X_getitem_3 + X_getitem_3)*(weight_getitem_(slice(None, None, None), 8) + weight_getitem_(slice(None, None, None), 8))+1 * (X_getitem_3 + X_getitem_3)*(weight_getitem_(slice(None, None, None), 9) + weight_getitem_(slice(None, None, None), 9))+1 * (X_getitem_4 + X_getitem_4)*(weight_getitem_(slice(None, None, None), 0) + weight_getitem_(slice(None, None, None), 0))+1 * (X_getitem_4 + X_getitem_4)*(weight_getitem_(slice(None, None, None), 1) + weight_getitem_(slice(None, None, None), 1))+1 * (X_getitem_4 + X_getitem_4)*(weight_getitem_(slice(None, None, None), 2) + weight_getitem_(slice(None, None, None), 2))+1 * (X_getitem_4 + X_getitem_4)*(weight_getitem_(slice(None, None, None), 3) + weight_getitem_(slice(None, None, None), 3))+1 * (X_getitem_4 + X_getitem_4)*(weight_getitem_(slice(None, None, None), 4) + weight_getitem_(slice(None, None, None), 4))+1 * (X_getitem_4 + X_getitem_4)*(weight_getitem_(slice(None, None, None), 5) + weight_getitem_(slice(None, None, None), 5))+1 * (X_getitem_4 + X_getitem_4)*(weight_getitem_(slice(None, None, None), 6) + weight_getitem_(slice(None, None, None), 6))+1 * (X_getitem_4 + X_getitem_4)*(weight_getitem_(slice(None, None, None), 7) + weight_getitem_(slice(None, None, None), 7))+1 * (X_getitem_4 + X_getitem_4)*(weight_getitem_(slice(None, None, None), 8) + weight_getitem_(slice(None, None, None), 8))+1 * (X_getitem_4 + X_getitem_4)*(weight_getitem_(slice(None, None, None), 9) + weight_getitem_(slice(None, None, None), 9))+1 * (X_getitem_5 + X_getitem_5)*(weight_getitem_(slice(None, None, None), 0) + weight_getitem_(slice(None, None, None), 0))+1 * (X_getitem_5 + X_getitem_5)*(weight_getitem_(slice(None, None, None), 1) + weight_getitem_(slice(None, None, None), 1))+1 * (X_getitem_5 + X_getitem_5)*(weight_getitem_(slice(None, None, None), 2) + weight_getitem_(slice(None, None, None), 2))+1 * (X_getitem_5 + X_getitem_5)*(weight_getitem_(slice(None, None, None), 3) + weight_getitem_(slice(None, None, None), 3))+1 * (X_getitem_5 + X_getitem_5)*(weight_getitem_(slice(None, None, None), 4) + weight_getitem_(slice(None, None, None), 4))+1 * (X_getitem_5 + X_getitem_5)*(weight_getitem_(slice(None, None, None), 5) + weight_getitem_(slice(None, None, None), 5))+1 * (X_getitem_5 + X_getitem_5)*(weight_getitem_(slice(None, None, None), 6) + weight_getitem_(slice(None, None, None), 6))+1 * (X_getitem_5 + X_getitem_5)*(weight_getitem_(slice(None, None, None), 7) + weight_getitem_(slice(None, None, None), 7))+1 * (X_getitem_5 + X_getitem_5)*(weight_getitem_(slice(None, None, None), 8) + weight_getitem_(slice(None, None, None), 8))+1 * (X_getitem_5 + X_getitem_5)*(weight_getitem_(slice(None, None, None), 9) + weight_getitem_(slice(None, None, None), 9))+1 * (X_getitem_6 + X_getitem_6)*(weight_getitem_(slice(None, None, None), 0) + weight_getitem_(slice(None, None, None), 0))+1 * (X_getitem_6 + X_getitem_6)*(weight_getitem_(slice(None, None, None), 1) + weight_getitem_(slice(None, None, None), 1))+1 * (X_getitem_6 + X_getitem_6)*(weight_getitem_(slice(None, None, None), 2) + weight_getitem_(slice(None, None, None), 2))+1 * (X_getitem_6 + X_getitem_6)*(weight_getitem_(slice(None, None, None), 3) + weight_getitem_(slice(None, None, None), 3))+1 * (X_getitem_6 + X_getitem_6)*(weight_getitem_(slice(None, None, None), 4) + weight_getitem_(slice(None, None, None), 4))+1 * (X_getitem_6 + X_getitem_6)*(weight_getitem_(slice(None, None, None), 5) + weight_getitem_(slice(None, None, None), 5))+1 * (X_getitem_6 + X_getitem_6)*(weight_getitem_(slice(None, None, None), 6) + weight_getitem_(slice(None, None, None), 6))+1 * (X_getitem_6 + X_getitem_6)*(weight_getitem_(slice(None, None, None), 7) + weight_getitem_(slice(None, None, None), 7))+1 * (X_getitem_6 + X_getitem_6)*(weight_getitem_(slice(None, None, None), 8) + weight_getitem_(slice(None, None, None), 8))+1 * (X_getitem_6 + X_getitem_6)*(weight_getitem_(slice(None, None, None), 9) + weight_getitem_(slice(None, None, None), 9))+1 * (X_getitem_7 + X_getitem_7)*(weight_getitem_(slice(None, None, None), 0) + weight_getitem_(slice(None, None, None), 0))+1 * (X_getitem_7 + X_getitem_7)*(weight_getitem_(slice(None, None, None), 1) + weight_getitem_(slice(None, None, None), 1))+1 * (X_getitem_7 + X_getitem_7)*(weight_getitem_(slice(None, None, None), 2) + weight_getitem_(slice(None, None, None), 2))+1 * (X_getitem_7 + X_getitem_7)*(weight_getitem_(slice(None, None, None), 3) + weight_getitem_(slice(None, None, None), 3))+1 * (X_getitem_7 + X_getitem_7)*(weight_getitem_(slice(None, None, None), 4) + weight_getitem_(slice(None, None, None), 4))+1 * (X_getitem_7 + X_getitem_7)*(weight_getitem_(slice(None, None, None), 5) + weight_getitem_(slice(None, None, None), 5))+1 * (X_getitem_7 + X_getitem_7)*(weight_getitem_(slice(None, None, None), 6) + weight_getitem_(slice(None, None, None), 6))+1 * (X_getitem_7 + X_getitem_7)*(weight_getitem_(slice(None, None, None), 7) + weight_getitem_(slice(None, None, None), 7))+1 * (X_getitem_7 + X_getitem_7)*(weight_getitem_(slice(None, None, None), 8) + weight_getitem_(slice(None, None, None), 8))+1 * (X_getitem_7 + X_getitem_7)*(weight_getitem_(slice(None, None, None), 9) + weight_getitem_(slice(None, None, None), 9))+1 * (X_getitem_8 + X_getitem_8)*(weight_getitem_(slice(None, None, None), 0) + weight_getitem_(slice(None, None, None), 0))+1 * (X_getitem_8 + X_getitem_8)*(weight_getitem_(slice(None, None, None), 1) + weight_getitem_(slice(None, None, None), 1))+1 * (X_getitem_8 + X_getitem_8)*(weight_getitem_(slice(None, None, None), 2) + weight_getitem_(slice(None, None, None), 2))+1 * (X_getitem_8 + X_getitem_8)*(weight_getitem_(slice(None, None, None), 3) + weight_getitem_(slice(None, None, None), 3))+1 * (X_getitem_8 + X_getitem_8)*(weight_getitem_(slice(None, None, None), 4) + weight_getitem_(slice(None, None, None), 4))+1 * (X_getitem_8 + X_getitem_8)*(weight_getitem_(slice(None, None, None), 5) + weight_getitem_(slice(None, None, None), 5))+1 * (X_getitem_8 + X_getitem_8)*(weight_getitem_(slice(None, None, None), 6) + weight_getitem_(slice(None, None, None), 6))+1 * (X_getitem_8 + X_getitem_8)*(weight_getitem_(slice(None, None, None), 7) + weight_getitem_(slice(None, None, None), 7))+1 * (X_getitem_8 + X_getitem_8)*(weight_getitem_(slice(None, None, None), 8) + weight_getitem_(slice(None, None, None), 8))+1 * (X_getitem_8 + X_getitem_8)*(weight_getitem_(slice(None, None, None), 9) + weight_getitem_(slice(None, None, None), 9))+1 * (X_getitem_9 + X_getitem_9)*(weight_getitem_(slice(None, None, None), 0) + weight_getitem_(slice(None, None, None), 0))+1 * (X_getitem_9 + X_getitem_9)*(weight_getitem_(slice(None, None, None), 1) + weight_getitem_(slice(None, None, None), 1))+1 * (X_getitem_9 + X_getitem_9)*(weight_getitem_(slice(None, None, None), 2) + weight_getitem_(slice(None, None, None), 2))+1 * (X_getitem_9 + X_getitem_9)*(weight_getitem_(slice(None, None, None), 3) + weight_getitem_(slice(None, None, None), 3))+1 * (X_getitem_9 + X_getitem_9)*(weight_getitem_(slice(None, None, None), 4) + weight_getitem_(slice(None, None, None), 4))+1 * (X_getitem_9 + X_getitem_9)*(weight_getitem_(slice(None, None, None), 5) + weight_getitem_(slice(None, None, None), 5))+1 * (X_getitem_9 + X_getitem_9)*(weight_getitem_(slice(None, None, None), 6) + weight_getitem_(slice(None, None, None), 6))+1 * (X_getitem_9 + X_getitem_9)*(weight_getitem_(slice(None, None, None), 7) + weight_getitem_(slice(None, None, None), 7))+1 * (X_getitem_9 + X_getitem_9)*(weight_getitem_(slice(None, None, None), 8) + weight_getitem_(slice(None, None, None), 8))+1 * (X_getitem_9 + X_getitem_9)*(weight_getitem_(slice(None, None, None), 9) + weight_getitem_(slice(None, None, None), 9))+1 * max(_concat + _concat, final_bias + final_bias)')
            exec(f'''def AreaOptimizerFn(W):
                 {','.join(filtered_vars)} = W
                 return  {area_fn}''', globals())

            assert 1 ==0

            # exec(f'''def ErrorConstraintFn(x):
            #      {','.join(filtered_vars)} = x
            #      return  {''}''', globals())

    def initializeData(self, model, training_size, testing_size, num_precision, num_range, output_size, data_range, batch_size, distribution):
        data_params = dict(
            batch_size=batch_size
        )

        # # generate testing/training data
        # training_set = self.gen_data(
        #     model, training_size, num_precision, num_range, output_size, data_range, dist=distribution)
        # train_gen = data.DataLoader(training_set, **data_params)
        # test_set = self.gen_data(
        #     model, testing_size, num_precision, num_range, output_size, data_range, dist=distribution)
        # test_gen = data.DataLoader(test_set, **data_params)

        # LOAD DATA
        training_set = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        test_set = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

        train_gen = torch.utils.data.DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)
        test_gen = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

        return train_gen, test_gen

    def initializeWeights(self, outputs, num_weights, num_range, initial_P, train_range=False):
        # output without grad
        print("WEIGHTS")
        print(outputs.values())


        #O = torch.Tensor(list(outputs.values()))
        O = outputs.values()
        # weights matrix
        P = torch.Tensor(num_weights).fill_(initial_P)
        init_P = P.clone()

        R = torch.Tensor(num_range).fill_(initial_P - 2)
        init_R = R.clone()

        print(P)
        P += 0

        P.requires_grad = True
        if train_range:
            R.requires_grad = True

        return O, P, init_P, R, init_R

    # Loss function
    def compute_loss(self, target, y, P, R, O, iter, filtered_vars, batch_size, epochs, training_size, error_type=1, should_print=True, shouldErrorCheck=False, train_range=False, incorporate_ulp_loss=False):
        """
        Args:
            error_type:
                1 ==> Paper Error
                2 ==> Soft-Loss on ULP
        """
        unpacked_P = []
        for weight in P:
            unpacked_P.append(weight)

        area = 0
        if train_range:
            ldict = {"P": P, "R": R}
            evaluator = NumEval(self.original_dag)
            evaluator.eval(**self.eval_dict)

            node_values = evaluator.node_values
            visitor = BitFlowVisitor(node_values, calculate_IB=False)
            visitor.run(evaluator.dag)

            range_list = [val for val in list(
                visitor.node_values) if val not in evaluator.dag.outputs]

            area_fn = visitor.area_fn
            exec(f"{','.join(filtered_vars)}=P", ldict)
            #ib_vars = [f"{f}_ib" for f in range_list]
            ib_vars = [f"{f}" for f in range_list]
            exec(f"{','.join(ib_vars)}=R", ldict)

            exec(f"area = {area_fn}", ldict)

            area = ldict["area"]
        else:
            area = AreaOptimizerFn(unpacked_P)

        if isinstance(target, list):
            target = torch.stack(target)
            y = torch.stack(y)

        loss = 0
        if error_type == 1:

            # If the error is met, turn off the error. If the error is not met, turn on the error.

            # Calculate erros
            constraint_err = ErrorConstraintFn(unpacked_P)
            ulp_err = torch.sum(self.ulp(y, target, O))

            # Sanity error check
            if shouldErrorCheck:
                if self.hasNegatives(constraint_err):
                    raise ValueError(
                        f"ERR NEGATIVE: {constraint_err}, {P}, {area}")

            # If ulp error is reasonable, relax error constraints
            S = 1./self.area_weight
            decay = 0.95
            if constraint_err > 0 and self.L > 1e-10:
                self.prevL = self.L
                self.L *= decay
            else:
                self.L = self.initL

            loss = (self.L * torch.exp(-1000 * constraint_err) +
                    S * area + torch.sum(torch.exp(-1000 * R)) + torch.sum(torch.exp(-1000 * P)))/(batch_size)

            if incorporate_ulp_loss:
                loss = loss + ulp_err/batch_size

            if train_range:
                loss = loss + self.evaluator.saturation
                self.evaluator.saturation = 0.

        else:
            ulp_error = torch.mean(torch.sum(
                self.within_ulp_err(y, target, O)))
            constraint_err = torch.max(ulp_error.float(), torch.zeros(1))

            constraint_W = 100 * \
                torch.sum(torch.max(-(P) + 0.5, torch.zeros(len(P))))

            loss = (self.R * ulp_error)/batch_size

        # Catch negative values for area and weights
        if shouldErrorCheck:
            if self.hasNegatives(area):
                raise ValueError(f"AREA ERR: {P}, {area}")

            if self.hasNegatives(W):
                raise ValueError(f"WEIGHT ERR: {P}, {area}")

        # Print out model details every so often
        if iter % 1000 == 0 and should_print == True:
            print(
                f"iteration {iter} of {epochs * training_size/batch_size} ({(iter * 100.)/(epochs * training_size/batch_size)}%)")
            print(f"AREA: {area}")
            print(f"ERROR: {constraint_err}")
            print(f"WEIGHTS: {P},\n         {R}")
            if error_type == 1:
                print(f"ERROR CONST: {self.L}")
            print(f"LOSS: {loss}")

        return loss

    def __init__(self, dag, outputs, data_range, training_size=2000, testing_size=200, batch_size=16, lr=1e-4, error_type=1, test_optimizer=True, test_ufb=True, train_range=False, range_lr=1e-4, distribution=0, graph_loss=False, custom_data=None, incorporate_ulp_loss=False):
        self.train_MNIST = True
        torch.manual_seed(42)
        self.original_dag = copy.deepcopy(dag)

        # Run a basic evaluator on the DAG to construct error and area functions
        bfo = self.constructOptimizationFunctions(
            dag, outputs, data_range)

        # Update the dag with round nodes and set up the model for torch training
        dag, num_precision, num_inputs, num_outputs, num_range, ordered_list, area_weight = self.update_dag(
            dag)

        self.area_weight = area_weight

        print('ordered list')
        print(ordered_list)
        filtered_vars = []
        for el in ordered_list:
            if el not in outputs:
                filtered_vars.append(el)
        print(filtered_vars)




        # Generate error and area functions from the visitor
        error_fn = bfo.error_fn
        area_fn = bfo.area_fn
        self.createExecutableConstraintFunctions(
            area_fn, error_fn, filtered_vars)

        model = self.gen_model(dag)

        # create the data according to specifications
        train_gen = None
        test_gen = None
        if custom_data is None:
            train_gen, test_gen = self.initializeData(model, training_size, testing_size,
                                                      num_precision, num_range, num_outputs, data_range, batch_size, distribution)
        else:
            train_gen = custom_data[0]
            test_gen = custom_data[1]

        # initialize the weights and outputs with appropriate gradient toggle
        O, P, init_P, R, init_R = self.initializeWeights(
            outputs, num_precision, num_range, bfo.initial, train_range=train_range)

        if error_type == 1:
            self.L = 1e5
            self.initL = self.L
            self.prevL = self.L

        # store data to object
        self.train_gen = train_gen
        self.test_gen = test_gen
        self.lr = lr
        self.P = P
        self.R = R
        self.O = O
        self.model = model
        self.batch_size = batch_size
        self.training_size = training_size
        self.error_type = error_type
        self.test_optimizer = test_optimizer
        self.test_ufb = test_ufb
        self.init_P = init_P
        self.init_R = init_R
        self.bfo = bfo
        self.train_range = train_range
        self.filtered_vars = filtered_vars
        self.range_lr = range_lr
        self.graph_loss = graph_loss
        self.incorporate_ulp_loss = incorporate_ulp_loss

    def train(self, epochs=10):

        # retrieve initialized data from object
        train_gen = self.train_gen
        test_gen = self.test_gen
        P = self.P
        R = self.R
        O = self.O
        model = self.model
        batch_size = self.batch_size
        training_size = self.training_size
        error_type = self.error_type
        test_optimizer = self.test_optimizer
        test_ufb = self.test_ufb
        init_P = self.init_P
        init_R = self.init_R
        lr = self.lr
        bfo = self.bfo
        train_range = self.train_range
        filtered_vars = self.filtered_vars
        range_lr = self.range_lr
        graph_loss = self.graph_loss
        incorporate_ulp_loss = self.incorporate_ulp_loss

        # Set up optimizer
        opt = torch.optim.AdamW(
            [{"params": P}, {"params": R, "lr": range_lr}], lr=lr)

        lr_decay = 0.5
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=5, gamma=lr_decay)

        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        # Run training process
        loss_values = []
        print("\n##### TRAINING ######")
        iter = 0
        for e in range(epochs):
            for t, (inputs, target_y) in enumerate(train_gen):

                # # Move data to GPU
                #
                # inputs = {k: inputs[k].to(device) for k in inputs}

                #print(inputs)
                print("INDI")
                print(P.shape)
                inputs["P"] = P

                inputs["R"] = R
                inputs["O"] = O
                print(inputs["P"],inputs["R"],inputs["O"])
                y = model(**inputs)

                if isinstance(y, list):
                    y = torch.stack(y)
                    target_y = torch.stack(target_y).squeeze().to(device)

                loss = self.compute_loss(target_y, y, P, R, O, iter, filtered_vars, batch_size, epochs, training_size,
                                         error_type=error_type, should_print=True, train_range=train_range, incorporate_ulp_loss=incorporate_ulp_loss)
                loss_values.append(loss)

                opt.zero_grad()
                loss.backward()
                opt.step()
                iter += 1
            # scheduler.step()

        if graph_loss:
            plt.plot(loss_values)
            plt.show()

        # Show final results for weight and round them
        print(P)
        P = self.custom_round(P, factor=0.1)
        R = self.custom_round(R, factor=0.1)

        # P = torch.ceil(P)
        # R = torch.ceil(R)
        print(f"PRECISION: {P}")
        print(f"RANGE: {R}")
        print(f"ORDER: {filtered_vars}")

        LUTTransformer = LookupTableTransformer(P, R, filtered_vars)
        model = self.gen_model(LUTTransformer.doit(self.transformed_dag))

        self.calc_accuracy("TEST", test_gen, P, R,
                           O, model, False)

        self.calc_ulp("TEST", test_gen, P, R,
                      O, model)

        if test_ufb:
            self.calc_accuracy("UFB TEST", test_gen, init_P, R,
                               O, model, False)

        self.calc_accuracy("TRAIN", train_gen, P, R,
                           O, model, False)

        if test_ufb:
            self.calc_accuracy("UFB TRAIN", train_gen, init_P, R,
                               O, model, False)

        print("\n##### MODEL DETAILS #####")
        print(f"ERROR: {ErrorConstraintFn(P.tolist())}")
        if train_range:
            ldict = {"P": P, "R": R}
            evaluator = NumEval(self.original_dag)
            evaluator.eval(**self.eval_dict)

            node_values = evaluator.node_values
            visitor = BitFlowVisitor(node_values, calculate_IB=False)
            visitor.run(evaluator.dag)

            range_list = [val for val in list(
                visitor.node_values) if val not in evaluator.dag.outputs]

            area_fn = visitor.area_fn
            exec(f"{','.join(filtered_vars)}=P", ldict)
            ib_vars = [f"{f}" for f in range_list]
            # ib_vars = [f"{f}_ib" for f in range_list]
            exec(f"{','.join(ib_vars)}=R", ldict)

            exec(f"area = {area_fn}", ldict)

            area = ldict["area"]
            print(f"AREA: {area}")
        else:
            print(f"AREA: {AreaOptimizerFn(P.tolist())}")

        if test_optimizer:
            print("\n##### FROM OPTIMIZER ######")
            bfo.solve()
            test = list(bfo.fb_sols.values())
            print(test)
            rng = list(bfo.visitor.IBs.values())
            print(rng)
            print(f"ERROR: {ErrorConstraintFn(test)}")
            print(f"AREA: {AreaOptimizerFn(test)}")

            self.calc_accuracy("OPTIMIZER TEST", test_gen, torch.tensor(test), torch.tensor(rng),
                               O, model, False)

            self.calc_ulp("OPTIMIZER TEST", test_gen, torch.tensor(test), torch.tensor(rng),
                          O, model)

        # Save outputs to object
        self.model = model
        self.P = P
        self.R = R
        self.O = O

    @ staticmethod
    def save(fileName, bitflow_object):
        with open(f'{fileName}.pt', 'wb') as output:
            torch.save(bitflow_object, output)

    @ staticmethod
    def load(fileName):
        with open(f'{fileName}.pt', 'rb') as input:
            loaded = torch.load(input)

            loaded.P = loaded.P.float()
            loaded.R = loaded.R.float()

            loaded.P.requires_grad = True
            loaded.R.requires_grad = loaded.train_range

            return loaded