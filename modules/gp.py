from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from modules.dataset import Dataset
import pandas as pd
from sklearn.gaussian_process.kernels import Matern
from typing import List, Optional
from modules.dataset import Dataset
import numpy as np
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn

class GPR(GaussianProcessRegressor):
    def __init__(
        self, 
        df_training: pd.DataFrame,
        X_idxs: List[int], 
        Y_idx: int,
        scale_X: Optional[bool] = True,
        kernel=Matern(nu=5/3), 
        alpha=1e-10, optimizer='fmin_l_bfgs_b',
        n_restarts_optimizer=0, 
        normalize_y=False, 
        copy_X_train=True,
        
        ):
        super().__init__(
            kernel=kernel, 
            alpha=alpha, 
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer, 
            normalize_y=normalize_y,
            copy_X_train=copy_X_train
            )
        self.df_training= df_training
        self.X_idxs=X_idxs
        self.Y_idx=Y_idx
        self.scale_X=scale_X
        
        self.training_dataset= Dataset(df=df_training, input_idxs=self.X_idxs, output_idx=Y_idx).get_IO_dataset(scale=self.scale_X)
        self.gpr=None
    def get_model(self):
        self.gpr = GaussianProcessRegressor(
            kernel=self.kernel, 
            alpha=self.alpha, 
            optimizer=self.optimizer,
            n_restarts_optimizer=self.n_restarts_optimizer, 
            normalize_y=self.normalize_y,
            copy_X_train=self.copy_X_train
            )
        self.gpr.fit(self.training_dataset.X,self.training_dataset.Y)
        
        return self.gpr
    
def save_onnx(gp_model: GaussianProcessRegressor, n_params: int):
    initial_type = [('float_input', FloatTensorType([None, n_params]))]
    onx = convert_sklearn(gp_model, initial_types=initial_type)
    with open("gp_model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    
