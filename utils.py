from copy import deepcopy
import random

def maketrainData(inputData):
    _local_data = deepcopy(inputData)
    rowIndex = random.sample(list(range(len(_local_data))), len(_local_data))
    
    for i in range(len(rowIndex)):
        _local_data[i][random.randint(0, _local_data.shape[1] - 1)] = -1
    
    return _local_data