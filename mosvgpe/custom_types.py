from typing import NewType, Tuple

import tensor_annotations.tensorflow as ttf
import tensorflow_probability as tfp
from tensor_annotations import axes
from tensor_annotations.axes import Batch

NumExperts = NewType("NumExperts", axes.Axis)
NumData = NewType("NumData", axes.Axis)
InputDim = NewType("InputDim", axes.Axis)
OutputDim = NewType("OutputDim", axes.Axis)
NumSamples = NewType("NumSamples", axes.Axis)

NumGatingFunctions = NewType("NumGatingFunctions", axes.Axis)

InputData = ttf.Tensor2[NumData, InputDim]
OutputData = ttf.Tensor2[NumData, OutputDim]
Dataset = Tuple[InputData, OutputData]

InputDataBatch = ttf.Tensor2[Batch, InputDim]
OutputDataBatch = ttf.Tensor2[Batch, OutputDim]
DatasetBatch = Tuple[InputDataBatch, OutputDataBatch]

Mean = ttf.Tensor2[NumData, OutputDim]
Variance = ttf.Tensor2[NumData, OutputDim]
MeanAndVariance = Tuple[Mean, Variance]

MeanSamples = ttf.Tensor3[NumSamples, NumData, OutputDim]
VarianceSamples = ttf.Tensor3[NumSamples, NumData, OutputDim]
MeanAndVarianceSamples = Tuple[MeanSamples, VarianceSamples]

MixingProb = ttf.Tensor2[NumData, NumExperts]
MixingProbSamples = ttf.Tensor3[NumSamples, NumData, NumExperts]

ExpertIndicatorCategoricalDist = tfp.distributions.Categorical
GatingFunctionSamples = ttf.Tensor3[NumSamples, NumData, NumExperts]
GatingMean = ttf.Tensor2[NumData, NumExperts]
GatingVariance = ttf.Tensor2[NumData, NumExperts]
GatingMeanAndVariance = Tuple[GatingMean, GatingVariance]
