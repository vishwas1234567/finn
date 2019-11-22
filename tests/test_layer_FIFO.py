import numpy as np
import onnx
from onnx import TensorProto, helper

import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper


def test_layer_FIFO():
    inp = helper.make_tensor_value_info("in", TensorProto.FLOAT, [2, 4, 4])
    outp = helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 4, 4])

    FIFO_node = helper.make_node(
        "FIFO",
        ["inp"],
        ["outp"],
        "outp",
        domain="finn",
        backend="fpgadataflow",
        depth=1024,
    )

    graph = helper.make_graph(
        nodes=[FIFO_node], name="FIFO_graph", inputs=[inp], outputs=[outp],
    )
    model = helper.make_model(graph, producer_name="finn-hls-onnx-model")
    model = ModelWrapper(model)

    # set the tensor datatypes (in this case: all to bipolar)
    for tensor in graph.input:
        model.set_tensor_datatype(tensor.name, DataType["BIPOLAR"])
    for tensor in graph.output:
        model.set_tensor_datatype(tensor.name, DataType["BIPOLAR"])

    onnx.save(model.model, "FIFO-model.onnx")

    # generate input data
    input_tensor = np.random.randint(2, size=32)
    input_tensor = (np.asarray(input_tensor, dtype=np.float32)).reshape(2, 4, 4)
    input_dict = {"inp": input_tensor}

    output_dict = oxe.execute_onnx(model, input_dict)

    assert (
        output_dict["outp"] == input_dict["inp"]
    ), "FIFO input and FIFO output do not match!"
