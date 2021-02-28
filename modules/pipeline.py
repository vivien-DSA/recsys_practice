###############################################LOAD THE libraries ########################################################

import numpy as np
from pathlib import Path
# import datatable as dt

from collections import defaultdict



###############################################LOAD THE DATA ########################################################

def load_data(filepath_data : Path, filename):
    # The independent variable on the train( predictors)
    train_dt =  dt.fread(filepath_data/ filename)
    train = train_dt.to_pandas()
    return train

######################################### CREATE MODEL AND PREDICT ########################################################


