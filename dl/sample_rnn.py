import random
import pandas as pd
import math
import numpy as np

from dl import util


def odlprint():
    import matplotlib.pyplot as plt
    random.seed(0)
    # 乱数の係数
    random_factor = 0.05
    # サイクルあたりのステップ数
    steps_per_cycle = 80
    # 生成するサイクル数
    number_of_cycles = 50

    df = pd.DataFrame(np.arange(steps_per_cycle * number_of_cycles + 1), columns=["t"])
    df["sin_t"] = df.t.apply(
        lambda x: math.sin(x * (2 * math.pi / steps_per_cycle) + random.uniform(-1.0, +1.0) * random_factor))
    df[["sin_t"]].head(steps_per_cycle * 2).plot()
    #    plt.show()

    length_of_sequences = 100
    print(df[["sin_t"]].tail(5))
    print(df[["sin_t"]].shape)
    (X_train, y_train), (X_test, y_test) = util.train_test_split(df[["sin_t"]],test_size=0.1, n_prev=length_of_sequences)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    import dl.oanda_keras as odl
    odl.odl(X_train,y_train,X_test,y_test)
