# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------# Based on the timm and MAE-priv code base
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/BUPT-PRIV/MAE-priv
# --------------------------------------------------------


# original
# S2_DEFAULT_MEAN = (867.22485, 966.4503, 1205.7284, 1241.4762, 1663.0093, 2869.3284,
#                    3450.0352, 3570.7214, 3723.5212, 3789.373, 3032.5723, 2149.3071)           
# S2_DEFAULT_STD = (740.3424, 741.81287, 731.1739, 844.05414, 830.8267, 903.8788,
#                   1183.1865, 1150.7161, 1178.2874, 1333.8562, 1004.8264, 1064.1743)
# MODIS_DEFAULT_MEAN = (885.86523, 3551.4817, 481.28528, 850.52325, 3525.8325, 2784.9258, 1665.4954)
# MODIS_DEFAULT_STD = (644.4249, 933.956, 558.6097, 555.3753, 388.08313, 657.5165, 791.2981)


S2_DEFAULT_MEAN = tuple(m / 10000.0 for m in (
                        867.22485, 966.4503, 1205.7284, 1241.4762, 1663.0093, 2869.3284,
                        3450.0352, 3570.7214, 3723.5212, 3789.373, 3032.5723, 2149.3071))
S2_DEFAULT_STD  = tuple(s / 10000.0 for s in (
                        740.3424, 741.81287, 731.1739, 844.05414, 830.8267, 903.8788,
                        1183.1865, 1150.7161, 1178.2874, 1333.8562, 1004.8264, 1064.1743))

MODIS_DEFAULT_MEAN = tuple(m / 10000.0 for m in (885.86523, 3551.4817, 481.28528, 850.52325, 3525.8325, 2784.9258, 1665.4954))
MODIS_DEFAULT_STD  = tuple(s / 10000.0 for s in (644.4249, 933.956, 558.6097, 555.3753, 388.08313, 657.5165, 791.2981))

S1_DEFAULT_MEAN = (0.12468473, 0.02877444)
S1_DEFAULT_STD  = (0.11273567, 0.01697816)




PAD_MASK_VALUE = 254

IMAGE_TASKS = ['modis', 's2', 's1']  

# Data paths
DATA_PATH = '/work/mech-ai-scratch/bgekim/project/imputation/IA_dataset/patches'