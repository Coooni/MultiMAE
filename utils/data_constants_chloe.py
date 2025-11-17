# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------# Based on the timm and MAE-priv code base
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/BUPT-PRIV/MAE-priv
# --------------------------------------------------------


# 125m resolution
# S2_DEFAULT_MEAN = (867.22485, 966.4503, 1205.7284, 1241.4762, 1663.0093, 2869.3284,
#                    3450.0352, 3570.7214, 3723.5212, 3789.373, 3032.5723, 2149.3071)           
# S2_DEFAULT_STD = (740.3424, 741.81287, 731.1739, 844.05414, 830.8267, 903.8788,
#                   1183.1865, 1150.7161, 1178.2874, 1333.8562, 1004.8264, 1064.1743)
# MODIS_DEFAULT_MEAN = (885.86523, 3551.4817, 481.28528, 850.52325, 3525.8325, 2784.9258, 1665.4954)
# MODIS_DEFAULT_STD = (644.4249, 933.956, 558.6097, 555.3753, 388.08313, 657.5165, 791.2981)


# # 125m resolution - RGB
# S2_DEFAULT_MEAN = tuple(m / 10000.0 for m in (
#                         1241.4762, 1205.7284, 966.4503))
# S2_DEFAULT_STD  = tuple(s / 10000.0 for s in (
#                         844.05414, 731.1739, 741.81287))
# MODIS_DEFAULT_MEAN = tuple(m / 10000.0 for m in (885.86523, 850.52325, 481.28528))
# MODIS_DEFAULT_STD  = tuple(s / 10000.0 for s in (644.4249, 555.3753, 558.6097))


# 30m resolution
S2_DEFAULT_MEAN = (566.14966, 688.7004, 939.1987, 1016.316, 1438.171, 2541.9949,
                   3087.3604, 3221.064, 3336.2102, 3445.905, 2796.623, 1912.3439)
S2_DEFAULT_STD = (676.5381, 684.8568, 662.08295, 806.42896, 791.71344, 843.35986,
                  1150.4697, 1119.9402, 1193.2979, 1367.8597, 988.0409, 1052.7238)


S1_DEFAULT_MEAN = (0.12419162, 0.02826689)
S1_DEFAULT_STD = (0.41080412, 0.04929494)


SOIL_DEFAULT_MEAN = (260.66113, 0.75120109, 86.993507, 0.69395339, 332.52991, 
                     265.26855, 0.77237779, 19513.955, 18246.875, 181.89001)

SOIL_DEFAULT_STD = (57.849964, 0.15605949, 10.95243, 0.16944358, 70.232323,
                    48.026318, 0.15176231, 12383.65, 11431.456, 29.635946)


ELEVATION_DEFAULT_MEAN = (33169.68)
ELEVATION_DEFAULT_STD = (6685.241)


WEATHER_DEFAULT_MEAN = (47589.148, 1.6730422, 385.21368, 0.11752752, 23.858633, 10.060264, 1341.6506)
WEATHER_DEFAULT_STD  = (5294.75146, 4.4465909, 102.251434, 0.773941696, 8.43982792, 7.71704674, 633.976624)





# PAD_MASK_VALUE = 254

# IMAGE_TASKS = ['modis', 's2', 's1']  

# # Data paths
# DATA_PATH = '/work/mech-ai-scratch/bgekim/project/imputation/IA_dataset/patches'