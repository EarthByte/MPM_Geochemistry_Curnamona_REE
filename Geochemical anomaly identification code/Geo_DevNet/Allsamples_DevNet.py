"""

This code performs geochemical anomaly identification on all samples based on the trained DevNet.

"""

import numpy as np
from geo_DevNet import load_model_weight_predict
from geo_utils import aucPerformance, dataLoading
from sklearn.preprocessing import normalize


model_path = './geomodel/devnet_PC8_dataset_0.015cr_64bs_7ko_d.h5'
input_shape = [8]
x_test, y_test = dataLoading('./geodata/PC8_dataset.csv')
x_test=normalize(x_test)

scores = load_model_weight_predict(model_path,
                                   input_shape=input_shape,
                                   x_test=x_test)
AUC_ROC, AUC_PR = aucPerformance(scores, y_test)
np.savetxt('./georesults/PC8_results.csv',scores)