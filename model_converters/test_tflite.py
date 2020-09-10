import tensorflow as tf
import numpy as np
from pathlib import Path
from nptyping import NDArray
import sys

def invoke_tflite_model(tflite_model_path: str, dummy_input: NDArray):
    assert Path(tflite_model_path).is_file(), f'No such file or directory: {tflite_model_path}'
    assert dummy_input.dtype is np.dtype(np.float32), 'Dummy input is not float32.'
    interpreter = tf.lite.Interpreter(tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_data = dummy_input.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    tflite_output_list = [interpreter.get_tensor(output_detail['index']) for output_detail in output_details]
    print(f'TFLITE model output shapes: {[tflite_output.shape for tflite_output in tflite_output_list]}')
    return tflite_output_list

def invoke_frozen_graph(frozen_graph_path: str, dummy_input: NDArray, verbose: bool = False):
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
    tflite_model_path = str(base_dir.joinpath('tflite_model.tflite'))
    frozen_graph_path = str(base_dir.joinpath('frozen_graph.pb'))
    input_shape = [1, 3, 32, 32]
    dummy_input = np.random.rand(*input_shape).astype(np.float32)

    try:
        tflite_model_output_list = invoke_tflite_model(tflite_model_path, dummy_input)
        print(f'Successfully invoked tflite model!')
        print('-'*100)
    except Exception as e:
        sys.exit(f'TFLITE model invoking failure: {e}')
    try:
        frozen_graph_output_list = invoke_frozen_graph(frozen_graph_path, dummy_input)
        print(f'Successfully invoked frozen graph!')
        print('-'*100)
    except Exception as e:
        sys.exit(f'Pytorch model invoking failure: {e}')
    loss = sum([
        NrmseBetweenArrays(tflite_model_output, frozen_graph_output) 
        for tflite_model_output, frozen_graph_output in zip(tflite_model_output_list, frozen_graph_output_list)
        ]) / len(tflite_model_output_list)
    print(
        f'''
        Normalized root mean squared error between pytorch model output and tensorflow frozen graph is: {loss}
        First element in tflite model output: {[tflite_model_output[0, 0, 0, 0, 0] for tflite_model_output in tflite_model_output_list]}
        First element in frozen graph output: {[frozen_graph_output[0, 0, 0, 0, 0] for frozen_graph_output in frozen_graph_output_list]}
        ''')