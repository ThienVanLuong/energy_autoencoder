import numpy as np

# generate data, one-hot vector
def generate_one_hot_vectors(M, data_size, get_label=False):
    """
    Generate one hot vectors which are used as training data or testing data.
    
    Parameters:
    -----
        M: int, dimension of one-hot vectors, i.e, number of classes/categoraries
        data_size: int, number of one-hot vectors generated
    
    Return:
        data: shape(data_size,M) array
        
    """
    label = np.random.randint(M, size=data_size)
    data = []
    for i in label:
        temp = np.zeros(M)
        temp[i] = 1
        data.append(temp)
    data = np.array(data)
    
    if not get_label:
        return data
    else:
        return data, label