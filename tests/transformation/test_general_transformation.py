from pkgutil import get_data

from finn.core.modelwrapper import ModelWrapper
from finn.transformation.general import GiveUniqueNodeNames


def test_give_unique_node_names():
    raw_m = get_data("finn", "data/onnx/mnist-conv/model.onnx")
    model = ModelWrapper(raw_m)
    model = model.transform(GiveUniqueNodeNames())
    assert model.graph.node[0].name == "Reshape_0"
    assert model.graph.node[1].name == "Conv_0"
    assert model.graph.node[11].name == "Add_2"
