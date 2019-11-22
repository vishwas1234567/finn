import sys
import os
import subprocess
import tempfile as tmp
import numpy as np

from finn.core.utils import get_by_name
from finn.custom_op.fpgadataflow import HLSCustomOp


class FIFO(HLSCustomOp):
    def make_shape_compatible_op(self, node):
        pass

    def infer_node_datatype(self, node, model):
        pass

    def execute_node(self, node, context, graph):
        # the FIFO passes the input values to the output
        context[node.output[0]] = context[node.input[0]]

    def get_attributes(self, node):
        self.depth = get_by_name(node.attribute, "depth").i

    def global_includes(self, node):
        pass

    def defines(self, node):
        pass

    def read_npy_data(self, node):
        pass

    def strm_decl(self, node):
        pass

    def docompute(self, node):
        pass

    def dataoutstrm(self, node):
        pass

    def save_as_npy(self, node):
        pass
