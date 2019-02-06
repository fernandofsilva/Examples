#%%
import pandas as pd
import numpy as np
import os

def split_data(path_to_file, path_to_save, size=0.7):
    """
    Argument:
    String -- String the path to read the dataset
    String -- String the path to save the train and test
    Int    -- Integer with size of the train data
    
    Returns:
    None
    """

    dataset = pd.read_csv(path_to_file)                   # Read Csv file
    dataset_size = dataset.shape[0]                       # Get the file length

    mask = np.random.choice(a=dataset_size,               # random select the
                            size=int(dataset_size*size),  # elements according
                            replace=False)                # the size selectec
    
    sub_mask = set(range(dataset_size))-set(mask)         # select the diff

    train = dataset.loc[mask]                             # split the train
    test = dataset.loc[sub_mask]                          # split the test
    
    # Save the train file
    train.to_csv(path_or_buf=os.path.join(path_to_save, 'train.csv'),
                 index=False)

    # Save the test file
    test.to_csv(path_or_buf=os.path.join(path_to_save, 'test.csv'),
                index=False)



#%%
split_data(os.path.join('/home/esssfff/Documents',
                        'WA_Fn-UseC_-Telco-Customer-Churn.csv'),
           '/home/esssfff/Documents/')

