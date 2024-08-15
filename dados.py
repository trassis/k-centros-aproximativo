import pandas as pd
import numpy as np

def banknote_data():
    df = pd.read_csv('datasets/banknote-authentication/data_banknote_authentication.txt', delimiter=',', header=None)
    pontos = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    K = len(np.unique(labels))
    
    return [K, pontos, labels]
    
banknote_data()
