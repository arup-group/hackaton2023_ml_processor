import pandas as pd
from dataclasses import dataclass
from nptyping import NDArray, Bool
from typing import Any, Optional,List, Tuple
from sklearn.preprocessing import StandardScaler


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
        
    def get_IO_dataset(self, scale: Optional[Bool] = True) -> IODataset:
        
        # Inputs
        X = self.df.loc[:, self.df.columns[self.input_idxs]].values
        # Output
        Y = pd.DataFrame = self.df.loc[:, self.df.columns[[self.output_idx]]].values
        
        if scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        return IODataset(
            scale=scale,
            X=X,
            Y=Y
        )
            
            
            
        
        # training_dataset[['p1', 'p2', 'p3', 'p4']].copy()
        
        
        # # Import dataframe
        
        # df = pd.read_excel(path, skiprows=5)
        

        # # Rename columns
        # # Current mapping
        # # p1 -> Load
        # # p2 -> Span in X-Dir
        # # p3 -> Span in Y-Dir
        # # p4 -> Slab thickness
        
        # df.rename(
        #     columns={
        #         df.columns[0] : 'p1',
        #         df.columns[1] : 'p2',
        #         df.columns[2] : 'p3',
        #         df.columns[3] : 'p4',
        #         df.columns[4] : 'rf',
        #         df.columns[5] : 'f1',
        #         df.columns[6] : 'f2',
        #         df.columns[7] : 'f3',
        #     },
        #     inplace=True
        # )
        
        
        # return df
