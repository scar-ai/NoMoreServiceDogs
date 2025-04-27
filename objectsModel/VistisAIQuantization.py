from onnxruntime.quantization import shape_inference
import vai_q_onnx
from onnxruntime.quantization.calibrate import CalibrationDataReader

from torch.utils.data import DataLoader

from dataset import Calib_dataset

import time
import random

time_before = time.time()
print('Beginning pre-processing...')

shape_inference.quant_pre_process(
   input_model_path = r"objectsModel\model\fasterrcnn_resnet50_fpn.onnx",
   output_model_path = r"objectsModel\model\temponnx.onnx",
   skip_optimization = False,
   skip_onnx_shape = False,
   skip_symbolic_shape = True,
   auto_merge = True,
   int_max = 2**31 - 1,
   guess_output_rank = False,
   verbose = 0,
   save_as_external_data = False)

time_after = time.time()
print(f'Pre-processing done! ({time_after-time_before}s)')

time_before = time.time()
print('Beginning Quantization...')



dataset = Calib_dataset(root_dir=r"objectsModel\calib_data")

def split_dataset_into_parts(dataset, num_parts=100):
  random.shuffle(dataset.image_paths)

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

    split_datasets.append(part_dataset)

    start_index = end_index

  return split_datasets


dataset_fragments = split_dataset_into_parts(dataset, num_parts=100)

test_dataloader = DataLoader(dataset_fragments[0], batch_size=1, shuffle=True)

print('\n----------------------------------------------')
print(len(test_dataloader))
print(next(iter(test_dataloader)))
print('\n----------------------------------------------')

class COCOV1CalibrationDataReader(CalibrationDataReader):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        print(len(self.iterator))

    def get_next(self):
        try:
            batch = next(self.iterator)
            return {"input": batch[0].numpy()}
        except StopIteration:
            return None
        
dr = COCOV1CalibrationDataReader(dataloader=test_dataloader)


vai_q_onnx.quantize_static(
    model_input = r"objectsModel\model\temponnx.onnx",
    model_output = r"objectsModel\model\quantized.onnx",
    calibration_data_reader = dr,
    quant_format=vai_q_onnx.QuantFormat.QDQ,
    calibrate_method=vai_q_onnx.CalibrationMethod.MinMax,
    activation_type=vai_q_onnx.QuantType.QUInt8,
    weight_type=vai_q_onnx.QuantType.QInt8,
    enable_dpu=True,
    extra_options= {"ActivationSymmetric":False,"WeightSymmetric":False, 'RemoveQDQConvLeakyRelu':True, 'RemoveQDQConvPRelu':True, 'EnableSubgraph':True, 'ConvertLeakyReluToDPUVersion':True,
                    'ConvertHardSigmoidToDPUVersion':True,'ConvertAvgPoolToDPUVersion':True, 'ConvertReduceMeanToDPUVersion':True,
                    'ConvertSoftmaxToDPUVersion':True}
)

time_after = time.time()
print(f'Quantization done! ({time_after-time_before}s)')