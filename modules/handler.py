import pandas as pd



def dataset_importer(path : str):
    
    # Import dataframe
    
    df = pd.read_excel(path, skiprows=5)
    
    # Drop column zero
    df.drop(df.columns[[0,1]], axis=1, inplace=True)

    # Rename columns
    # Current mapping
    # p1 -> Load
    # p2 -> Span in X-Dir
    # p3 -> Span in Y-Dir
    # p4 -> Slab thickness
    
    df.rename(
        columns={
            df.columns[0] : 'p1',
            df.columns[1] : 'p2',
            df.columns[2] : 'p3',
            df.columns[3] : 'p4',
            df.columns[4] : 'rf',
            df.columns[5] : 'f1',
            df.columns[6] : 'f2',
            df.columns[7] : 'f3',
        },
        inplace=True
    )
    
    
    return df
