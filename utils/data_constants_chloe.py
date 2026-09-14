# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------# Based on the timm and MAE-priv code base
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/BUPT-PRIV/MAE-priv
# --------------------------------------------------------



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

