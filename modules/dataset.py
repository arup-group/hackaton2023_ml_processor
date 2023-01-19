import pandas as pd
from dataclasses import dataclass
from nptyping import NDArray
from typing import List

@dataclass
class IODataset:
    X: NDArray
    Y: NDArray

class Dataset():
    
    def __init__(self, df: pd.DataFrame, input_idxs: List[int], output_idx: int):
        self.df=df
        self.input_idxs = input_idxs
        self.output_idx=output_idx
        
    def get_IO_dataset(
        self, 
        ) -> IODataset:
        
        # Inputs
        X = self.df.loc[:, self.df.columns[self.input_idxs]].values
        # Output
        Y = self.df.loc[:, self.df.columns[[self.output_idx]]].values
        
        return IODataset(
            X=X,
            Y=Y
        )
    

