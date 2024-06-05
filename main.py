############################################################################
#         Ensemble model calibration and uncertainty quantification        #
#    in geologic CO2 storage using a spatiotemporal deep learning proxy    #
############################################################################
# Author: Misael M. Morales (github.com/misaelmmorales)                    #
# Co-Authors: Dr. Carlos Torres-Verdin, Dr. Michael Pyrcz - UT Austin      #
# Date: 2024                                                               #
############################################################################
# Copyright (c) 2024, Misael M. Morales                                    #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyesmda
import tensorflow as tf

import keras
import keras.backend as K
from keras import Model
from keras.layers import *