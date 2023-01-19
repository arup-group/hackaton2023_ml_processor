import pandas as pd
from modules.gp import GPR, GPScores
from modules.dataset import Dataset, DatasetType
import numpy as np
# Datasets 
TRAINING_DATASET_PATH = "./datasets/dataset_02_256.xlsx" 
TESTING_DATASET_PATH="./datasets/dataset_02_81.xlsx" 

ONX_EXPORT_PATH = "./onnx_export"

# Dataframes
df_training = pd.read_excel(TRAINING_DATASET_PATH , skiprows=5)
df_testing = pd.read_excel(TESTING_DATASET_PATH, skiprows=5)

test_dataset= Dataset(df=df_testing, input_idxs=[1,2,3,4], output_idx=5).get_IO_dataset()

gp_model= GPR(df_training=df_training, X_idxs=[1,2,3,4], Y_idx=5)


tests = gp_model.estimate(X=np.array([[2,5,5,0.14],[2,5,11.5,0.4]]))
gp_scores : GPScores  = gp_model.get_scores(test_dataset=test_dataset)

print(f"GP score: {gp_scores.default_score}")
print(f"RMSE: {gp_scores.rmse}")
print(f"MAE: {gp_scores.mae}")

gp_model.save_onnx(n_X=4, path=ONX_EXPORT_PATH)