import pandas as pd
from modules.gp import GPR, save_GP_onnx, GPScores
from modules.dataset import Dataset, DatasetType
# Datasets 
TRAINING_DATASET_PATH = "./datasets/dataset_02_256.xlsx" 
TESTING_DATASET_PATH="./datasets/dataset_02_81.xlsx" 

# Dataframes
df_training = pd.read_excel(TRAINING_DATASET_PATH , skiprows=5)
df_testing = pd.read_excel(TESTING_DATASET_PATH, skiprows=5)

test_dataset= Dataset(df=df_testing, input_idxs=[1,2,3,4], output_idx=5).get_IO_dataset(scale=True, dataset_type=DatasetType.TEST)

gp_model= GPR(df_training=df_training, X_idxs=[1,2,3,4], Y_idx=5, scale_X=True)

gp_scores : GPScores  = gp_model.get_scores(test_dataset=test_dataset)

print(f"GP score: {gp_scores.default_score}")
print(f"RMSE: {gp_scores.rmse}")
print(f"MAE: {gp_scores.mae}")