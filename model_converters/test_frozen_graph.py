import sys
import os
sys.path.append(os.getcwd())
import torch
from pathlib import Path
from nptyping import NDArray
import numpy as np
from models.yolo import Model
import tensorflow as tf


def invoke_torch_model(torch_model_path: str, dummy_input: torch.Tensor):
    assert Path(torch_model_path).is_file(), f'No such file or directory: {torch_model_path}'
    assert dummy_input.dtype is torch.float32, 'Dummy input is not float32.'
    print(f'Found pytorch model at {torch_model_path}. Loading model...')
    torch_model = torch.load(torch_model_path, map_location=torch.device('cpu'))['model'].float()
    torch_model.eval() # turn off special layers for evaluation
    torch_model.model[-1].export = True
    torch_model_output_list = torch_model(dummy_input)
    print(f'Pytorch model output shapes: {[tuple(torch_model_output.shape) for torch_model_output in torch_model_output_list[1]]}')
    return torch_model_output_list

def invoke_frozen_graph(frozen_graph_path: str, dummy_input: NDArray, verbose: bool = False):
    assert Path(frozen_graph_path).is_file(), f'No such file or directory: {frozen_graph_path}'
    with tf.compat.v1.gfile.GFile(frozen_graph_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")
        config = tf.compat.v1.ConfigProto()
        try:
            with tf.compat.v1.Session(config=config) as sess:
                frozen_graph_output_list = sess.run(["output_0:0", "output_1:0", "output_2:0"],feed_dict={'input:0':dummy_input})
                print(f'Frozen graph output shapes: {[frozen_graph_output.shape for frozen_graph_output in frozen_graph_output_list]}')
        except Exception as e:
            tensor_name_list = [tensor.name for tensor in tf.compat.v1.get_default_graph().as_graph_def().node]
            for tensor_name in tensor_name_list:
                print(tensor_name, '\n')
            sys.exit(f'Pytorch model invoking failure: {e}')
        return frozen_graph_output_list

def NrmseBetweenArrays(array_pred: NDArray, array_gt: NDArray):
    upper = np.abs(array_pred - array_gt)
    lower = np.abs(array_gt - np.mean(array_gt))
    return np.mean(upper/lower)

if __name__ == "__main__":
    base_dir = Path('inference/output')
    torch_model_path = 'runs/exp62/weights/last.pt'
    frozen_graph_path = str(base_dir.joinpath('frozen_graph.pb'))
    input_shape = [1, 3, 32, 32]
    dummy_input = np.random.rand(*input_shape).astype(np.float32)
    try:
        torch_model_output_list = invoke_torch_model(torch_model_path, torch.from_numpy(dummy_input))
        print(f'Successfully invoked pytroch model!')
        print('-'*100)
    except Exception as e:
        sys.exit(f'Pytorch model invoking failure: {e}')
    try:
        frozen_graph_output_list = invoke_frozen_graph(frozen_graph_path, dummy_input)
        print(f'Successfully invoked frozen graph!')
        print('-'*100)
    except Exception as e:
        sys.exit(f'Pytorch model invoking failure: {e}')
    loss = sum([
        NrmseBetweenArrays(torch_model_output.detach().numpy(), frozen_graph_output) 
        for torch_model_output, frozen_graph_output in zip(torch_model_output_list, frozen_graph_output_list)
        ]) / len(torch_model_output_list)
    print(
        f'''
        Normalized root mean squared error between pytorch model output and tensorflow frozen graph is: {loss}
        First element in torch model output: {[float(torch_model_output[0, 0, 0, 0, 0]) for torch_model_output in torch_model_output_list]}
        First element in frozen graph output: {[frozen_graph_output[0, 0, 0, 0, 0] for frozen_graph_output in frozen_graph_output_list]}
        ''')
