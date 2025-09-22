# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------# Based on the timm and MAE-priv code base
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/BUPT-PRIV/MAE-priv
# --------------------------------------------------------


# # 125m
# S2_DEFAULT_MEAN = tuple(m / 10000.0 for m in (
#                         867.22485, 966.4503, 1205.7284, 1241.4762, 1663.0093, 2869.3284,
#                         3450.0352, 3570.7214, 3723.5212, 3789.373, 3032.5723, 2149.3071))
# S2_DEFAULT_STD  = tuple(s / 10000.0 for s in (
#                         740.3424, 741.81287, 731.1739, 844.05414, 830.8267, 903.8788,
#                         1183.1865, 1150.7161, 1178.2874, 1333.8562, 1004.8264, 1064.1743))

# MODIS_DEFAULT_MEAN = tuple(m / 10000.0 for m in (885.86523, 3551.4817, 481.28528, 850.52325, 3525.8325, 2784.9258, 1665.4954))
# MODIS_DEFAULT_STD  = tuple(s / 10000.0 for s in (644.4249, 933.956, 558.6097, 555.3753, 388.08313, 657.5165, 791.2981))

# S1_DEFAULT_MEAN = (0.12468473, 0.02877444)
# S1_DEFAULT_STD  = (0.11273567, 0.01697816)

# # 30m
# S2_DEFAULT_MEAN = tuple(m / 10000.0 for m in (
#                         485.7277, 580.1641, 758.458, 832.11084, 1129.3434, 2026.8511,
#                         2524.7615, 2601.077, 2664.0583, 2746.762, 2127.2568, 1500.915))
# S2_DEFAULT_STD  = tuple(s / 10000.0 for s in (
#                         918.6235, 914.5344, 908.462, 1017.73553, 1116.023, 1473.9512,
#                         1897.0212, 1903.8214, 2011.5074, 2208.987, 1614.9838, 1380.0662))


# MODIS_DEFAULT_MEAN = tuple(m / 10000.0 for m in (885.86523, 3551.4817, 481.28528, 850.52325, 3525.8325, 2784.9258, 1665.4954))
# MODIS_DEFAULT_STD  = tuple(s / 10000.0 for s in (644.4249, 933.956, 558.6097, 555.3753, 388.08313, 657.5165, 791.2981))

# S1_DEFAULT_MEAN = (0.11502027, 0.02443061)
# S1_DEFAULT_STD  = (0.29532233, 0.04774326)

# 30m resolution - new
S2_DEFAULT_MEAN = (0.056615, 0.06887008, 0.09391985, 0.10163148, 0.14381698, 0.25419976,
                   0.30873655, 0.32210684, 0.33362151, 0.34459104, 0.27966201, 0.19123411)
S2_DEFAULT_STD = (0.06765407, 0.06848592, 0.06620852, 0.08064312, 0.0791716, 0.08433618,
                  0.1150472, 0.11199426, 0.11933007, 0.13678636, 0.09880431, 0.10527262)


S1_DEFAULT_MEAN = (0.12419162, 0.02826689)
S1_DEFAULT_STD = (0.41080412, 0.04929494)





PAD_MASK_VALUE = 254

IMAGE_TASKS = ['modis', 's2', 's1']  

# Data paths
DATA_PATH = '/work/mech-ai-scratch/bgekim/project/imputation/IA_dataset/patches'