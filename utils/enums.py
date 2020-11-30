from enum import Enum


class NumpyPadMode(Enum):
    """
    See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
    """

    CONSTANT = "constant"
    EDGE = "edge"
    LINEAR_RAMP = "linear_ramp"
    MAXIMUM = "maximum"
    MEAN = "mean"
    MEDIAN = "median"
    MINIMUM = "minimum"
    REFLECT = "reflect"
    SYMMETRIC = "symmetric"
    WRAP = "wrap"
    EMPTY = "empty"


class GridSampleMode(Enum):
    """
    See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """

    BILINEAR = "bilinear"
    NEAREST = "nearest"


class InterpolateMode(Enum):
    """
    See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
    """

    NEAREST = "nearest"
    LINEAR = "linear"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    TRILINEAR = "trilinear"
    AREA = "area"


class UpsampleMode(Enum):
    """
    See also: https://pytorch.org/docs/stable/nn.html#upsample
    """

    NEAREST = "nearest"
    LINEAR = "linear"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    TRILINEAR = "trilinear"


class BlendMode(Enum):
    """
    See also: :py:class:`monai.data.utils.compute_importance_map`
    """

    CONSTANT = "constant"
    GAUSSIAN = "gaussian"


class PytorchPadMode(Enum):
    """
    See also: https://pytorch.org/docs/stable/nn.functional.html#pad
    """

    CONSTANT = "constant"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    CIRCULAR = "circular"


class GridSamplePadMode(Enum):
    """
    See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """

    ZEROS = "zeros"
    BORDER = "border"
    REFLECTION = "reflection"


class Average(Enum):
    """
    See also: :py:class:`monai.metrics.rocauc.compute_roc_auc`
    """

    MACRO = "macro"
    WEIGHTED = "weighted"
    MICRO = "micro"
    NONE = "none"


class MetricReduction(Enum):
    """
    See also: :py:class:`monai.metrics.meandice.DiceMetric`
    """

    NONE = "none"
    MEAN = "mean"
    SUM = "sum"
    MEAN_BATCH = "mean_batch"
    SUM_BATCH = "sum_batch"
    MEAN_CHANNEL = "mean_channel"
    SUM_CHANNEL = "sum_channel"


class LossReduction(Enum):
    """
    See also:
        - :py:class:`monai.losses.dice.DiceLoss`
        - :py:class:`monai.losses.dice.GeneralizedDiceLoss`
        - :py:class:`monai.losses.focal_loss.FocalLoss`
        - :py:class:`monai.losses.tversky.TverskyLoss`
    """

    NONE = "none"
    MEAN = "mean"
    SUM = "sum"


class Weight(Enum):
    """
    See also: :py:class:`monai.losses.dice.GeneralizedDiceLoss`
    """

    SQUARE = "square"
    SIMPLE = "simple"
    UNIFORM = "uniform"


class Normalisation(Enum):
    """
    See also:
        - :py:class:`monai.networks.nets.ConvNormActi`
        - :py:class:`monai.networks.nets.HighResBlock`
        - :py:class:`monai.networks.nets.HighResNet`
    """

    BATCH = "batch"
    INSTANCE = "instance"


class Activation(Enum):
    """
    See also:
        - :py:class:`monai.networks.nets.ConvNormActi`
        - :py:class:`monai.networks.nets.HighResBlock`
        - :py:class:`monai.networks.nets.HighResNet`
    """

    RELU = "relu"
    PRELU = "prelu"
    RELU6 = "relu6"


class ChannelMatching(Enum):
    """
    See also: :py:class:`monai.networks.nets.HighResBlock`
    """

    PAD = "pad"
    PROJECT = "project"


class Method(Enum):
    """
    See also: :py:class:`monai.transforms.croppad.array.SpatialPad`
    """

    SYMMETRIC = "symmetric"
    END = "end"