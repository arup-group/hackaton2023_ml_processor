import pandas as pd
from modules.gp import GPR, save_onnx, get_GP_score
from modules.dataset import Dataset, DatasetType
# Datasets 
TRAINING_DATASET_PATH = "./datasets/dataset_02_256.xlsx" 
TESTING_DATASET_PATH="./datasets/dataset_02_81.xlsx" 

# Dataframes
df_training = pd.read_excel(TRAINING_DATASET_PATH , skiprows=5)
df_testing = pd.read_excel(TESTING_DATASET_PATH, skiprows=5)

gp_model = GPR(df_training=df_training, X_idxs=[1,2,3,4], Y_idx=5, scale_X=True).get_model()

test_dataset= Dataset(df=df_testing, input_idxs=[1,2,3,4], output_idx=5).get_IO_dataset(scale=True, dataset_type=DatasetType.TEST)

gp_score = get_GP_score(gp_model=gp_model, test_dataset=test_dataset)

print(gp_score)





# save_onnx(gp_model=gp_model, n_params=4)
print("Ciao")
# gpr = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=9)
# gpr.fit(training_dataset.X,training_dataset.Y)
# print("ciao")
# n_input_params = 4
# training_dataset = handler.dataset_importer(DATASET_PATH)
# test_dataset = handler.dataset_importer()

# # Scale the data
# scaler = StandardScaler()

# X = scaler.fit_transform(np.array(training_dataset[['p1', 'p2', 'p3', 'p4']].copy()))
# Y_rf = np.array(training_dataset[['rf']].copy())

# X_train, X_test, y_train, y_test=train_test_split(X,Y_rf,test_size=0.2)

# Ytest_assessed = 

# kernel = Matern(nu=5/2)
# # Define Gaussian process regressor
# gpr = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=9)

# # Fit the model to the deata
# gpr.fit(X_train,y_train)

# x_test = scaler.transform([[3,9.67, 7.33,0.226]])
# y_pred, sigma = gpr.predict(x_test, return_std=True)

# print(x_test)
# print(y_pred)
# print(sigma)
# print("ciao")

# ts = [np.round(x,2) for x in np.arange(0.14,0.4,0.02)]
# testing_points = [[[5,12,12,t]] for t in ts]
# rf=[gpr.predict(scaler.transform(params), return_std=True) for params in testing_points]

# mean = [v[0][0] for v in rf]
# std = [v[1][0] for v in rf]

# TESTING_DATASET_PATH="./datasets/dataset_02_81.xlsx" 
# testing_dataset = handler.dataset_importer(TESTING_DATASET_PATH)
# # Xtest = scaler.fit_transform(np.array(testing_dataset[['p1', 'p2', 'p3', 'p4']].copy()))
# # Ytest = np.array(testing_dataset[['rf']].copy())

# score = gpr.score(X_test,y_test)
# print(score)
