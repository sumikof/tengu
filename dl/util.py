

def train_test_split(df, test_size=0.1, n_prev = 100):
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    X_train, y_train = _load_data(df.iloc[0:ntrn], n_prev)
    X_test, y_test = _load_data(df.iloc[ntrn:], n_prev)

    return (X_train, y_train), (X_test, y_test)

def _load_data(data, n_prev = 100):
    """
    data should be pd.DataFrame()
    """
    import numpy as np
    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].values)
        docY.append(data.iloc[i+n_prev].values)
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def standard_0_1(df):
    xmax = df.max()
    xmin = df.min()

    return  (df - xmin )/(xmax -xmin)
