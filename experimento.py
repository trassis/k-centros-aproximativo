import numpy as np
import random
import time
import sklearn as skl
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

random.seed(1234)

class experimento:
    # pontos é um np array com pontos nas linhas
    # resposta é um np array com as labels dos pontos
    def __init__(self, K, pontos, labels, p=1):
        # Parâmetro da distancia de Minkowski
        self.p = p
        self.N = pontos.shape[0]
        self.K = K
        
        self.pontos = pontos
        self.labels = labels
        
        # Calcula a matriz de distância usando norma p 
        t_inicial = time.time()
        self._calcular_distancias()
        t_final = time.time()
        print("Tempo para calcular matriz de dist: ", t_final - t_inicial)
        
    def _calcular_distancias(self):
        # Calculando dist matrix
        self.dist_matrix = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                self.dist_matrix[i][j] = np.sum(np.abs(self.pontos[i] - self.pontos[j]) ** self.p) ** (1 / self.p)
                
        """
        # Salvando na memória
        np.save('test_dist.npy', self.dist_matrix)
        
        # Carregando pre computado da memória
        self.dist_matrix = np.load('test_dist.npy')
        """
                
    # Constrói uma solução que possui raio, no máximo 2*r
    # Note que pode ser retornada uma lista de centros com tamanho > K
    def aprox_raio(self, r):
        centros = []
        ok = [False]*self.N
        i_lista = list(range(self.N))
        random.shuffle(i_lista)
        for i in range(len(i_lista)):
            if ok[i_lista[i]]:
                continue
            centros.append(i_lista[i])
            for j in range(i+1, self.N):
                if ok[i_lista[j]]:
                    continue
                if self.dist_matrix[i_lista[i]][i_lista[j]] <= 2*r:
                    ok[i_lista[j]] = True

        return centros

    # Retorna solução aproximada no máximo 2*ótimo com busca binária
    def aprox_refinamento(self, largura=1e-6):
        lo = 0.0
        hi = 0.0
        for i in range(len(self.pontos)):
            for j in range(i+1, len(self.pontos)):
                hi = max(hi, self.dist_matrix[i][j])

        largura_final = largura*(hi - lo)
        
        ultima_solucao = None
        while abs(hi - lo) > largura_final:
            mid = (lo+hi)/2
            solucao = self.aprox_raio(mid)
            if len(solucao) <= self.K:
                ultima_solucao = solucao
                hi = mid
            else:
                lo = mid

        centros = ultima_solucao
        return centros 

    # Retorna solução aproximada no máximo 2*ótimo com método incremental
    def aprox_incremental(self):
        if self.K >= self.N:
            return list(range(self.N))
        
        start = random.randint(0, self.N-1)
        centros = [ start ]
        
        falta = list(range(self.N))
        falta.remove(start)
        while len(centros) < self.K:
            idx = -1 
            mxdist_matrix = -1
            for i in falta:
                idist_matrix = 0
                for c in centros:
                    idist_matrix = max(idist_matrix, self.dist_matrix[i][c])
                if idist_matrix > mxdist_matrix:
                    idx = i
                    mxdist_matrix = idist_matrix

            centros.append(idx)
            falta.remove(idx)

        return centros
    
    def aprox_kmeans(self):
        kmeans = KMeans(n_clusters=self.K, n_init=10, random_state=random.randint(0, 1000))
        resposta = kmeans.fit_predict(self.pontos)
        centros = kmeans.cluster_centers_
        return [ centros, resposta ]
    
    def _calcular_particao(self, centros):
        resposta = [-1]*self.N
        for i in range(self.N):
            mn = 0
            for j in range(1, len(centros)):
                if self.dist_matrix[i][centros[j]] < self.dist_matrix[i][mn]:
                    mn = j
            resposta[i] = mn
        return resposta
    
    def _calcular_raio(self, resposta):
        raio = 0
        for i in range(self.N):
            c = resposta[i]
            if self.dist_matrix[i][c] > raio:
                raio = self.dist_matrix[i][c]
        return raio

    def teste_incremental(self):
        NUM_IT = 30
        tempo = np.zeros(NUM_IT)
        raio = np.zeros(NUM_IT)
        silhueta = np.zeros(NUM_IT)
        rand = np.zeros(NUM_IT)
        for i in range(NUM_IT):
            inicial = time.time()
            centros = self.aprox_incremental()
            resposta = self._calcular_particao(centros)
            final = time.time()
            
            tempo[i] = final - inicial
            raio[i] = self._calcular_raio(resposta)
            try:
                silhueta[i] = skl.metrics.silhouette_score(self.pontos, resposta)
            except:
                silhueta[i] = -1

            rand[i] = skl.metrics.adjusted_rand_score(self.labels, resposta)
            
        dados_tempo = [ np.mean(tempo), np.std(tempo) ]
        dados_raio = [ np.mean(raio), np.std(raio) ]
        dados_silhueta = [ np.mean(silhueta), np.std(silhueta) ]
        dados_rand = [ np.mean(rand), np.std(rand) ]
        
        print("Teste incremental feito:")
        print("\t\t Média \t\t\t Desvio padrão")
        print(f"tempo: \t\t {dados_tempo[0]} \t {dados_tempo[1]}")
        print(f"raio: \t\t {dados_raio[0]} \t {dados_raio[1]}")
        print(f"silhueta: \t {dados_silhueta[0]} \t {dados_silhueta[1]}")
        print(f"rand: \t\t {dados_rand[0]} \t {dados_rand[1]}")
        
    def teste_kmeans(self):
        NUM_IT = 30
        tempo = np.zeros(NUM_IT)
        raio = np.zeros(NUM_IT)
        silhueta = np.zeros(NUM_IT)
        rand = np.zeros(NUM_IT)
        for i in range(NUM_IT):
            inicial = time.time()
            centros, resposta = self.aprox_kmeans()
            final = time.time()
            
            tempo[i] = final - inicial
            raio[i] = self._calcular_raio(resposta)
            try:
                silhueta[i] = skl.metrics.silhouette_score(self.pontos, resposta)
            except:
                silhueta[i] = -1
            rand[i] = skl.metrics.adjusted_rand_score(self.labels, resposta)
            
        dados_tempo = [ np.mean(tempo), np.std(tempo) ]
        dados_raio = [ np.mean(raio), np.std(raio) ]
        dados_silhueta = [ np.mean(silhueta), np.std(silhueta) ]
        dados_rand = [ np.mean(rand), np.std(rand) ]
        
        print("Teste kmeans feito:")
        print("\t\t Média \t\t\t Desvio padrão")
        print(f"tempo: \t\t {dados_tempo[0]} \t {dados_tempo[1]}")
        print(f"raio: \t\t {dados_raio[0]} \t {dados_raio[1]}")
        print(f"silhueta: \t {dados_silhueta[0]} \t {dados_silhueta[1]}")
        print(f"rand: \t\t {dados_rand[0]} \t {dados_rand[1]}")
        
    def teste_refinamento(self, largura):
        NUM_IT = 30
        tempo = np.zeros(NUM_IT)
        raio = np.zeros(NUM_IT)
        silhueta = np.zeros(NUM_IT)
        rand = np.zeros(NUM_IT)
        for i in range(NUM_IT):
            inicial = time.time()
            centros = self.aprox_refinamento(largura)
            resposta = self._calcular_particao(centros)
            final = time.time()
            
            tempo[i] = final - inicial
            raio[i] = self._calcular_raio(resposta)
            try:
                silhueta[i] = skl.metrics.silhouette_score(self.pontos, resposta)
            except:
                silhueta[i] = -1
            rand[i] = skl.metrics.adjusted_rand_score(self.labels, resposta)
            
        dados_tempo = [ np.mean(tempo), np.std(tempo) ]
        dados_raio = [ np.mean(raio), np.std(raio) ]
        dados_silhueta = [ np.mean(silhueta), np.std(silhueta) ]
        dados_rand = [ np.mean(rand), np.std(rand) ]
        
        print(f"Teste refinamento com largura {largura:.0%} feito:")
        print("\t\t Média \t\t\t Desvio padrão")
        print(f"tempo: \t\t {dados_tempo[0]} \t {dados_tempo[1]}")
        print(f"raio: \t\t {dados_raio[0]:.15f} \t {dados_raio[1]}")
        print(f"silhueta: \t {dados_silhueta[0]} \t {dados_silhueta[1]}")
        print(f"rand: \t\t {dados_rand[0]} \t {dados_rand[1]}")
    
    def imprimir_experimento(self):
        plt.figure(figsize=(3, 3))
        scatter = plt.scatter(self.pontos[:, 0], self.pontos[:, 1], c=self.labels, cmap='viridis', s=2)
        plt.title("Visualização do Experimento")
        plt.show()
    
    def executar_testes(self):
        self.teste_incremental()
        print()
        self.teste_kmeans()
        print()
        for largura in [ 0.01, 0.05, 0.10, 0.15, 0.20 ]:
            self.teste_refinamento(largura)
            print()
