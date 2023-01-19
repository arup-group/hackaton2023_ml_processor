from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from modules.dataset import Dataset
import pandas as pd
from sklearn.gaussian_process.kernels import Matern
from typing import List, Optional
from modules.dataset import Dataset, DatasetType
import numpy as np
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error
from datetime import datetime

@dataclass
class GPScores:
    default_score: float
    rmse: float
    mae: float
    
class GPR(GaussianProcessRegressor):
    def __init__(
        self, 
        df_training: pd.DataFrame,
        X_idxs: List[int], 
        Y_idx: int,
        scale_X: Optional[bool] = True,
        kernel=Matern(nu=5/2), 
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
        
        self.training_dataset= Dataset(
            df=df_training, 
            input_idxs=self.X_idxs, 
            output_idx=Y_idx).get_IO_dataset(
                scale=self.scale_X, 
                dataset_type=DatasetType.TRAIN
                )
            
        self._get_model()
        
    def _get_model(self):
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
    


    def get_scores(self, test_dataset: Dataset) -> GPScores:
        
        # Default GP score
        gp_score = np.round(self.gpr.score(test_dataset.X, test_dataset.Y),2)
        
        # Predict Mean value from the GP 
        y = self.gpr.predict(test_dataset.X, return_std = False)
        
        # RMSE  (Root mean squared error)
        rmse = np.round(np.sqrt(mean_squared_error(y,test_dataset.Y)),3)
        
        # MAE Mean Absolute Error
        mae = np.round(mean_squared_error(y, test_dataset.Y),2) 
        
        return GPScores(
            default_score=gp_score,
            rmse =rmse,
            mae=mae
        )

    def save_onnx(self, n_X: int, path: str = None)->None:
        
        initial_type = [('float_input', FloatTensorType([None, n_X]))]
        onx = convert_sklearn(self.gpr, initial_types=initial_type)
        
        now = datetime.now()
        date_time_str = now.strftime("%m_%d_%Y_%H_%M_%S")
        
        export_path = f"gp_model_{date_time_str}.onnx"
        
        if path:
            export_path = f"{path}/{export_path}"

        with open(export_path, "wb") as f:
            f.write(onx.SerializeToString())
        
        return
    
