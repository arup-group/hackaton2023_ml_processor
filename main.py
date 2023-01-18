from modules import handler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.preprocessing import StandardScaler
import numpy as np
# Dataset handler
DATASET_PATH = "./datasets/dataset_01_81.xlsx" 

n_input_params = 4
training_dataset = handler.dataset_importer(DATASET_PATH)


# Scale the data
scaler = StandardScaler()

X = scaler.fit_transform(np.array(training_dataset[['p1', 'p2', 'p3', 'p4']].copy()))


Y_rf = np.array(training_dataset[['rf']].copy())

# kernel = 1.0 * RBF(
#     length_scale=[
#         10**-5,
#         10**-5,
#         10**-5,
#         10**-5
#         ],
#     length_scale_bounds=
#     [
#         [10**-30, 10**15],
#         [10**-30, 10**15],
#         [10**-30, 10**15],
#         [10**-30, 10**15]
#     ]
# )
kernel =RBF()*Matern()*RationalQuadratic()
# Define Gaussian process regressor
gpr = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=9)

# Fit the model to the deata
gpr.fit(X,Y_rf)

x_test = scaler.transform([[3,9.67, 7.33,0.313]])
y_pred, sigma = gpr.predict(x_test, return_std=True)

print(x_test)
print(y_pred)
print(sigma)
print("ciao")