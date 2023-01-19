import pandas as pd
from modules.gp import GPR, GPScores
from modules.dataset import Dataset
import numpy as np
from sklearn.model_selection import train_test_split

DATASET = "./datasets/DataSet04-256.csv" 

df_dataset = pd.read_csv(DATASET)

ONX_EXPORT_PATH = "./onnx_export"
PICKLE_EXPORT_PATH = "./pickle_export"


df_training, df_testing = train_test_split(df_dataset, test_size=0.2)


test_dataset= Dataset(df=np.round(df_testing,2), input_idxs=[1,2,3,4], output_idx=5).get_IO_dataset()

gp_model= GPR(df_training=df_training, X_idxs=[1,2,3,4], Y_idx=5)

# tests = gp_model.estimate(X=np.array([[2,5,12,0.2],[2,5,11.5,0.25],[2,5,12,0.2]]))
gp_scores : GPScores  = gp_model.get_scores(test_dataset=test_dataset)

print(f"GP score: {gp_scores.default_score}")
print(f"RMSE: {gp_scores.rmse}")
print(f"MAE: {gp_scores.mae}")

# Export Onnx
# gp_model.save_onnx(n_X=4, path=ONX_EXPORT_PATH)

gp_model.save_gp_model(path=PICKLE_EXPORT_PATH)