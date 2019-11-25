import finn.custom_op.registry as registry
from finn.core.datatype import DataType
from finn.transformation import Transformation


def _infer_node_datatype(model, node):
    """Infer output datatype(s) for a particular node. Returns True if any
    changes were made."""
    idtypes = list(map(lambda x: model.get_tensor_datatype(x), node.input))
    odtypes = list(map(lambda x: model.get_tensor_datatype(x), node.output))
    op_type = node.op_type
    if node.domain == "finn":
        # handle DataType inference for CustomOp
        try:
            # lookup op_type in registry of CustomOps
            inst = registry.custom_op[op_type](node)
            inst.infer_node_datatype(model)
        except KeyError:
            # exception if op_type is not supported
            raise Exception("Custom op_type %s is currently not supported." % op_type)
    else:
        if node.op_type == "Sign":
            # always produces bipolar outputs
            model.set_tensor_datatype(node.output[0], DataType.BIPOLAR)
        elif node.op_type == "MatMul":
            if len(list(filter(lambda x: x == DataType.FLOAT32, idtypes))) != 0:
                # node has at least one float input, output is also float
                model.set_tensor_datatype(node.output[0], DataType.FLOAT32)
            else:
                # TODO compute minimum / maximum result to minimize bitwidth
                # use (u)int32 accumulators for now
                has_signed_inp = len(list(filter(lambda x: x.signed(), idtypes))) != 0
                if has_signed_inp:
                    odtype = DataType.INT32
                else:
                    odtype = DataType.UINT32
                model.set_tensor_datatype(node.output[0], odtype)
        else:
            # unknown, assume node produces float32 outputs
            for o in node.output:
                model.set_tensor_datatype(o, DataType.FLOAT32)
    # compare old and new output dtypes to see if anything changed
    new_odtypes = list(map(lambda x: model.get_tensor_datatype(x), node.output))
    graph_modified = new_odtypes != odtypes
    return graph_modified


class InferDataTypes(Transformation):
    """Infer FINN DataType info for all intermediate/output tensors based on
    inputs and node type."""

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        for node in graph.node:
            graph_modified |= _infer_node_datatype(model, node)
        return (model, graph_modified)
