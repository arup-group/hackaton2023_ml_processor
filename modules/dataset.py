import pandas as pd
from dataclasses import dataclass
from nptyping import NDArray, Bool
from typing import List
from sklearn.preprocessing import StandardScaler
from enum import Enum

class DatasetType(Enum):
    TRAIN = "TRAIN"
    TEST = "TEST"
    
@dataclass
class IODataset:
    scale: Bool 
    X: NDArray
    Y: NDArray

class Dataset():
    
    def __init__(self, df: pd.DataFrame, input_idxs: List[int], output_idx: int):
        self.df=df
        self.input_idxs = input_idxs
        self.output_idx=output_idx
        
    def get_IO_dataset(
        self, 
        scale: bool, 
        dataset_type: DatasetType,
        ) -> IODataset:
        
        # Inputs
        X = self.df.loc[:, self.df.columns[self.input_idxs]].values
        # Output
        Y = pd.DataFrame = self.df.loc[:, self.df.columns[[self.output_idx]]].values
        
        if scale:
            
            scaler = StandardScaler()
            
            if dataset_type == DatasetType.TRAIN:
                X = scaler.fit_transform(X)
            
            if dataset_type == DatasetType.TEST:
                scaler.fit(X)
                X = scaler.transform(X)
        
        return IODataset(
            scale=scale,
            X=X,
            Y=Y
        )

