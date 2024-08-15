import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12345) 

# K = num de centros
# N = média de pontos ao redor de cada centro
# SCALE = limite superior para o valor absoluto de cada coordenada
# STD = variância ao redor dos centros
def gerar_dados(K, N, SCALE, STD):
    centros = np.random.rand(K, 2) * SCALE  

    pontos = []
    labels = []
    for i in range(K):
        num_novos = np.random.randint(N/2, 2*N)
        cov_matrix = np.eye(2) * STD
        novos_pontos = np.random.multivariate_normal(centros[i], cov_matrix, num_novos)

        pontos.append(novos_pontos)
        labels.append(np.full(num_novos, i))

    # Não colocamos os centros na lista de pontos
    """
    for i, pt in enumerate(centros):
        pontos.append(pt)
        labels.append(np.array([i]))
    """

    pontos = np.vstack(pontos)
    labels = np.concatenate(labels)
    
    return [pontos, labels]

def imprimir_pontos(pontos, labels, num_conjunto=0):
    plt.figure(figsize=(3, 3))
    K = len(np.unique(labels))
    for i in range(K):
        plt.scatter(pontos[labels == i, 0], pontos[labels == i, 1], label=f'Cluster {i+1}', s=3)
    plt.title(f'Pontos do conjunto {num_conjunto}')
    plt.show()

parametros = [
    # Com 5 centros, variância aumentando
    [ 5, 150, 10, 0.01 ],
    [ 5, 150, 10, 0.1 ],
    [ 5, 150, 10, 0.5 ],
    [ 5, 150, 10, 1.0 ],
    [ 5, 150, 10, 10.0 ],
    # Com 7 centros, variância aumentando
    [ 7, 200, 20, 0.01 ],
    [ 7, 200, 20, 0.1 ],
    [ 7, 200, 20, 0.5 ],
    [ 7, 200, 20, 1.0 ],
    [ 7, 200, 20, 10.0 ]
]

def dados_normal_1():
    K, N, SCALE, STD = parametros[0]
    pontos, labels = gerar_dados(K, N, SCALE, STD)
    return [K, pontos, labels]

def dados_normal_2():
    K, N, SCALE, STD = parametros[1]
    pontos, labels = gerar_dados(K, N, SCALE, STD)
    return [K, pontos, labels]

def dados_normal_3():
    K, N, SCALE, STD = parametros[2]
    pontos, labels = gerar_dados(K, N, SCALE, STD)
    return [K, pontos, labels]

def dados_normal_4():
    K, N, SCALE, STD = parametros[3]
    pontos, labels = gerar_dados(K, N, SCALE, STD)
    return [K, pontos, labels]

def dados_normal_5():
    K, N, SCALE, STD = parametros[4]
    pontos, labels = gerar_dados(K, N, SCALE, STD)
    return [K, pontos, labels]

def dados_normal_6():
    K, N, SCALE, STD = parametros[5]
    pontos, labels = gerar_dados(K, N, SCALE, STD)
    return [K, pontos, labels]

def dados_normal_7():
    K, N, SCALE, STD = parametros[6]
    pontos, labels = gerar_dados(K, N, SCALE, STD)
    return [K, pontos, labels]

def dados_normal_8():
    K, N, SCALE, STD = parametros[7]
    pontos, labels = gerar_dados(K, N, SCALE, STD)
    return [K, pontos, labels]

def dados_normal_9():
    K, N, SCALE, STD = parametros[8]
    pontos, labels = gerar_dados(K, N, SCALE, STD)
    return [K, pontos, labels]

def dados_normal_10():
    K, N, SCALE, STD = parametros[9]
    pontos, labels = gerar_dados(K, N, SCALE, STD)
    return [K, pontos, labels]
