import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder

def ret_(df):
    df = df.dropna()
    pontos = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels) 
    K = len(np.unique(labels))

    return [K, pontos, labels]

def dados_rice():
    df = arff.loadarff('./datasets/rice-cammeo-and-osmancik/Rice_Cammeo_Osmancik.arff')
    df = pd.DataFrame(df[0])
    return ret_(df)

def dados_banknote():
    df = pd.read_csv('datasets/banknote-authentication/data_banknote_authentication.txt', delimiter=',', header=None)
    return ret_(df) 

def dados_wine_red():
    df = pd.read_csv('datasets/wine-quality/winequality-red.csv', delimiter=';')
    return ret_(df)

def dados_wine_white():
    df = pd.read_csv('datasets/wine-quality/winequality-white.csv', delimiter=';')
    return ret_(df)

def dados_abalone():
    df = pd.read_csv('datasets/abalone/abalone.data', delimiter=',', header=None)
    df[0] = df[0].map({'F':0, 'M':1})
    return ret_(df)

def dados_eletrical():
    df = pd.read_csv('datasets/eletrical-stability/eletrical.csv')
    return ret_(df)

def dados_raisin():
    df = arff.loadarff('./datasets/raisin/Raisin_Dataset/Raisin_Dataset.arff')
    df = pd.DataFrame(df[0])
    return ret_(df)

def dados_shopping():
    df = pd.read_csv('datasets/online-shopping/online_shoppers.csv')
    df.drop('Month', axis=1, inplace=True)
    df.drop('Revenue', axis=1, inplace=True)
    df.drop('Weekend', axis=1, inplace=True)
    return ret_(df)

def dados_yeast():
    df = pd.read_csv('datasets/yeast/yeast.data', delim_whitespace=True)
    df.drop(df.columns[0], axis=1, inplace=True)
    return ret_(df)

def dados_optical():
    df = pd.read_csv('datasets/optical_recognition_of_handwritten_digits/optdigits.tra')
    return ret_(df)
