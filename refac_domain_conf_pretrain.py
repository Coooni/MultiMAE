"""
Pretraining domain config — single time point
  s1: 2ch, s2: 12ch, elevation: 1ch, soil: 10ch, weather: 7ch, cdl: 1ch
"""
from functools import partial
from multimae.criterion import MaskedMSELoss
from multimae.input_adapters import PatchedInputAdapter
from multimae.output_adapters import SpatialOutputAdapter


DOMAIN_CONF = {
    's1': {
        'channels': 2,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=2),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=2),
        'loss': MaskedMSELoss,
    },
    's2': {
        'channels': 12,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=12),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=12),
        'loss': MaskedMSELoss,
    },
    'elevation': {
        'channels': 1,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=1),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=1),
        'loss': MaskedMSELoss,
    },
    'soil': {
        'channels': 10,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=10),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=10),
        'loss': MaskedMSELoss,
    },
    'weather': {
        'channels': 7,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=7),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=7),
        'loss': MaskedMSELoss,
    },
    'cdl': {
        'channels': 1,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=1),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=1),
        'loss': MaskedMSELoss,
    },
}
