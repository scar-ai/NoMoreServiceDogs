from onnxruntime.quantization import shape_inference
import vai_q_onnx
from onnxruntime.quantization.calibrate import CalibrationDataReader

from torch.utils.data import DataLoader
from dataset import DailyPlacesDataset

import time
import random

time_before = time.time()
print('Beginning pre-processing...')

shape_inference.quant_pre_process(
   input_model_path = r"environnementModel\model\resnet50_finetuned.onnx",
   output_model_path = r"environnementModel\model\preprocessed.onnx",
   skip_optimization = False,
   skip_onnx_shape = False,
   skip_symbolic_shape = False,
   auto_merge = True,
   int_max = 2**31 - 1,
   guess_output_rank = False,
   verbose = 0,
   save_as_external_data = False)

time_after = time.time()
print(f'Pre-processing done! ({time_after-time_before}s)')

print("Loading dataset...")
dataset = DailyPlacesDataset(root_dir='dataset\Places')


def split_dataset_into_parts(dataset, num_parts=100):
  random.shuffle(dataset.image_paths)
  random.shuffle(dataset.labels)


  num_elements_per_part = len(dataset) // num_parts
  remainder = len(dataset) % num_parts


  split_datasets = []
  start_index = 0


  for i in range(num_parts):
    end_index = start_index + num_elements_per_part
    if i < remainder:
      end_index += 1


    part_dataset = type(dataset)(dataset.root_dir)
    part_dataset.image_paths = dataset.image_paths[start_index:end_index]
    part_dataset.labels = dataset.labels[start_index:end_index]

    split_datasets.append(part_dataset)

    start_index = end_index

  return split_datasets



dataset_fragments = split_dataset_into_parts(dataset, num_parts=75)
label_map = dataset.getClasses()



test_dataloader = DataLoader(dataset_fragments[0], batch_size=1, shuffle=True)


time_before = time.time()
print('Beginning Quantization...')
class CalibrationDataReader(CalibrationDataReader):
    def __init__(self):
        super().__init__()
        self.iterator = iter(test_dataloader)

    def get_next(self) -> dict:
        try:
            return {"input": next(self.iterator)[0].numpy()}
        except Exception:
            return None



dr = CalibrationDataReader()
vai_q_onnx.quantize_static(
    model_input = r"environnementModel\model\preprocessed.onnx",
    model_output = r"environnementModel\model\quantized.onnx",
    calibration_data_reader = dr,
    calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
    activation_type=vai_q_onnx.QuantType.QInt8,
    weight_type=vai_q_onnx.QuantType.QInt8,
    enable_dpu=True,
    extra_options={"ActivationSymmetric":True,'RemoveQDQConvLeakyRelu':True, 'RemoveQDQConvPRelu':True, 'EnableSubgraph':True,
                   'ConvertLeakyReluToDPUVersion':True, 'ConvertHardSigmoidToDPUVersion':True,
                   'ConvertAvgPoolToDPUVersion':True, 'ConvertReduceMeanToDPUVersion':True, 'ConvertSoftmaxToDPUVersion':True}
)


time_after = time.time()
print(f'Quantization done! ({time_after-time_before}s)')