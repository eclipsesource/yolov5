import torch
import tensorflow as tf
from pathlib import Path
from onnx_tf.backend import prepare
from models.yolo import Model
from typing import List
import onnx
import os
from cached_property import cached_property
from tensorflow.python.platform import gfile


class tfliteConverter():
    def __init__(
        self,
        output_dir: str = 'inference/output',
        torch_model_path: str = 'runs/exp18/weights/last.pt',
        input_shape: List[int] = [1, 3, 1056, 1632],
        input_node_names: List[str] = ['input'],
        output_node_names: List[str] = ['output_0', 'output_1', 'output_2']
        ):
        self._torch_model_path = torch_model_path
        self._onnx_model_path = str(Path(output_dir).joinpath('onnx_model.onnx'))
        self._pb_model_path = str(Path(output_dir).joinpath('frozen_graph.pb'))
        self._tflite_model_path = str(Path(output_dir).joinpath('tflite_model.tflite'))
        self._tflite_quant_model_path = str(Path(output_dir).joinpath('tflite_quant_model.tflite'))
        self._input_shape = input_shape
        self._input_node_names = input_node_names
        self._output_node_names = output_node_names

    def convert_torch_to_tflite(self, verbose: bool = False):
        assert Path(self._torch_model_path).is_file(), f'No such file or directory: {self._torch_model_path}'
        try: 
            torch_model = torch.load(self._torch_model_path, map_location=torch.device('cpu'))['model'].float()
            print('PT load success')
        except Exception as e:
            print(f'PT load failure: {e}')
        try:
            self._convert_torch_to_onnx(torch_model)
            onnx_model = onnx.load(self._onnx_model_path)
            onnx.checker.check_model(onnx_model)
            print(f'ONNX export success, saved as {self._onnx_model_path}')
        except Exception as e:
            print(f'ONNX export failure: {e}')
        try:
            self._convert_onnx_to_pb(onnx_model)
            with tf.compat.v1.Session() as sess:
                with gfile.FastGFile(self._pb_model_path, 'rb') as f:
                    graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                tf.import_graph_def(graph_def, name='')
                graph_nodes_names=[n.name for n in graph_def.node]
                if verbose:
                    print(graph_nodes_names)
            print(f'PB export success, saved as {self._pb_model_path}')
        except Exception as e:
            print(f'PB export failure: {e}')
        try:            
            self._convert_pb_to_tflite()
            print(f'TFLITE export success, saved as {self._tflite_model_path}')
        except Exception as e:
            print(f'TFLITE export failure: {e}')
        interpreter = tf.lite.Interpreter(model_path=self._tflite_model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        print(input_details)


    def _convert_torch_to_onnx(self, torch_model):
        torch_model.eval()
        print('Evaluation succeed')
        torch_model.model[-1].export = True
        print('Set export True.')
        dummy_input = torch.rand(*self._input_shape)
        print(dummy_input.shape)
        y = torch_model(dummy_input)
        torch_model.fuse()
        torch.onnx.export(
            torch_model,
            dummy_input,
            self._onnx_model_path,
            verbose=False,
            opset_version=12,
            input_names=self._input_node_names,
            output_names=self._output_node_names
            )

    def _convert_onnx_to_pb(self, onnx_model):
        tf_rep = prepare(onnx_model, device='cpu')
        tf_rep.export_graph(self._pb_model_path)

    def _convert_pb_to_tflite(self):
        converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
            self._pb_model_path,
            self._input_node_names,
            self._output_node_names,
            input_shapes={'input': self._input_shape}
            )
        converter.experimental_new_converter = True
        tflite_model = converter.convert()
        open(self._tflite_model_path, "wb").write(tflite_model)
        #converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        converter.post_training_quantize=True
        tflite_quant_model = converter.convert()
        open(self._tflite_quant_model_path, "wb").write(tflite_quant_model)


if __name__ == "__main__":
    tfliteConverter().convert_torch_to_tflite()
    pass