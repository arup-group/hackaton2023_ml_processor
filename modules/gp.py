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
from enum import Enum
from nptyping import NDArray
from sklearn.preprocessing import StandardScaler
class ConfidenceEnum(Enum):
    C68 = "68"
    C95 = "95"
    C99 = "99"
    
@dataclass
class GPScores:
    default_score: float
    rmse: float
    mae: float
@dataclass
class GPConfidenceInterval:
    confidence: ConfidenceEnum
    bounds: List[float]
@dataclass
class GPEstimate:
    mean: float
    std : float
    cv: float
    confidence_intervals: List[GPConfidenceInterval]
    

class GPR(GaussianProcessRegressor):
    def __init__(
        self, 
        df_training: pd.DataFrame,
        X_idxs: List[int], 
        Y_idx: int,
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
        self.scaler = StandardScaler()
        
        self.training_dataset= Dataset(
            df=df_training, 
            input_idxs=self.X_idxs, 
            output_idx=Y_idx).get_IO_dataset()
            
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
        X_transformed = self.scaler.fit_transform(self.training_dataset.X)
        self.gpr.fit(X_transformed, self.training_dataset.Y)
        return self.gpr
    


    def get_scores(self, test_dataset: Dataset) -> GPScores:
        
        X_transformed = self.scaler.transform(test_dataset.X)
        # Default GP score
        gp_score = np.round(self.gpr.score(X_transformed, test_dataset.Y),2)
        
        # Predict Mean value from the GP 
        y = self.gpr.predict(X_transformed, return_std = False)
        
        # RMSE  (Root mean squared error)
        rmse = np.round(np.sqrt(mean_squared_error(y,test_dataset.Y)),3)
        
        # MAE Mean Absolute Error
        mae = np.round(mean_squared_error(y, test_dataset.Y),2) 
        
        return GPScores(
            default_score=gp_score,
            rmse =rmse,
            mae=mae
        )
        
    def estimate(self, X: NDArray) -> List[GPEstimate]:
        
        X_transformed = self.scaler.transform(X)
            
        result: List[GPEstimate] = []
        
        # GP prediction
        mean, std = self.gpr.predict(X_transformed, return_std = True)
        
        for mu, sigma in zip(mean, std):
            
            confidence_intervals: List[GPConfidenceInterval] = []
            for n_sigma, confidence in zip([1, 2, 3], [68, 95, 99.7]):
                
                confidence_intervals.append(
                    GPConfidenceInterval(
                        confidence = confidence,
                        bounds = [np.round(mu-n_sigma*sigma,3),np.round(mu + n_sigma*sigma,3)]
                    )
                )
            result.append(
                GPEstimate(
                    mean=np.round(mu,3),
                    std=np.round(sigma,5),
                    cv=np.round(std/mean,5),
                    confidence_intervals=confidence_intervals
                )
            )

        return result
    
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
    
