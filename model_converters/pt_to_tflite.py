import sys
import os
sys.path.append(os.getcwd())
from pathlib import Path
from typing import List
from model_converters.util import *
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf


def _export_torch_as_onnx(torch_model, input_shape: List[int], output_path: str, input_node_names: List[str], output_node_names: List[str]):
    if isinstance(torch_model, str):
        torch_model = load_torch_model(torch_model)
    dummy_input = torch.rand(*input_shape, dtype=torch.float32)
    y = torch_model(dummy_input)
    torch_model.fuse()
    torch_model.model[-1].export = True # if export is true, training is false. some layers as bn are disabled.
    torch.onnx.export(torch_model, dummy_input, output_path, verbose=False, opset_version=12, input_names=input_node_names, output_names=output_node_names)
    print(f'Successfully exported ONNX model, saved as {output_path}')
    print("-" * 50)

def _export_onnx_as_frozen_graph(onnx_model_path: str, frozen_graph_path: str):
    onnx_model = onnx.load(onnx_model_path)
    tf_rep = prepare(onnx_model, device='cpu')
    tf_rep.export_graph(frozen_graph_path)
    print(f'Successfully exported frozen graph, saved as {frozen_graph_path}')
    print("-" * 50)

def _export_frozen_graph_as_tflite(frozen_graph_path: str, tflite_model_path: str):
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(open(frozen_graph_path, 'rb').read())
    concrete_func = wrap_frozen_graph(graph_def, inputs=['input:0'], outputs=['output_0:0', 'output_1:0','output_2:0'])
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.experimental_new_converter = True
    converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    open(tflite_model_path, "wb").write(tflite_model)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_quant_model_path = Path(tflite_model_path).parent.joinpath('tflite_model_quant.tflite')
    converter.post_training_quantize=True
    tflite_quant_model = converter.convert()
    open(tflite_quant_model_path, "wb").write(tflite_quant_model)
    print(f'Successfully exported tflite model, saved as {tflite_model_path}')
    print("-" * 50)

def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")
    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    print("Frozen model layers: ")
    layers = [op.name for op in import_graph.get_operations()]
    if print_graph == True:
        for layer in layers:
            print(layer)
    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


if __name__ == "__main__":
    output_dir = 'inference/output'
    torch_model_path = 'runs/exp62/weights/last.pt'
    input_node_names = ['input']
    output_node_names = ['output_0', 'output_1', 'output_2']
    input_shape = [1, 3, 32, 32]
    Path(output_dir).mkdir(exist_ok = True)
    torch_model = load_torch_model(torch_model_path)
    input_shape = regularize_shape(input_shape, int(max(torch_model.stride)))
    onnx_model_path = str(Path(output_dir).joinpath('onnx_model.onnx'))
    frozen_graph_path = str(Path(output_dir).joinpath('frozen_graph.pb'))
    tflite_model_path = str(Path(output_dir).joinpath('tflite_model.tflite'))
    _export_torch_as_onnx(torch_model, input_shape, onnx_model_path, input_node_names, output_node_names)
    _export_onnx_as_frozen_graph(onnx_model_path, frozen_graph_path)
    _export_frozen_graph_as_tflite(frozen_graph_path, tflite_model_path)

