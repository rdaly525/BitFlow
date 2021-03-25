from pathlib import Path
import numpy as np
import importlib

# AHA imports
import magma as m
import fault

# svreal import
from svreal import get_svreal_header

from msdsl import MixedSignalModel, VerilogGenerator, get_msdsl_header

BUILD_DIR = Path(__file__).resolve().parent / 'build'
DOMAIN = np.pi
RANGE = 1.0


def myfunc(x):
    # clip input
    x = np.clip(x, -DOMAIN, +DOMAIN)
    # apply function
    return np.sin(x)


def gen_model(order=0, numel=512):
    # create mixed-signal model
    model = MixedSignalModel('model', build_dir=BUILD_DIR)
    model.add_analog_input('in_')
    model.add_analog_output('out')
    model.add_digital_input('clk')
    model.add_digital_input('rst')

    # create function
    real_func = model.make_function(
        myfunc, domain=[-DOMAIN, +DOMAIN], order=1, numel=numel)

    # apply function
    model.set_from_sync_func(model.out, real_func,
                             model.in_, clk=model.clk, rst=model.rst)

    # write the model
    return model.compile_to_file(VerilogGenerator())


gen_model()
