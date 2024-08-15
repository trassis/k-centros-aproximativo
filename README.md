# O que tem que ser feito:

Implementar 2 algoritmos aproximados para k-centros.
- Intervalo de raio ótimo é refinado até uma largura definida
- Centros escolhidos para maximizar a distância entre os centros previamente escolhidos.

Implementar a métrica de Minkowski para p=1 e p=2

Conjuntos de dados:
- Dados da UCIMLR: 10 conjuntos com no mínimo 700 exemplos.  
- Dados de scikit
- Dados da média

Para cada conjunto, deve ser feito 30 testes para cada algoritmo.
(A matriz de distância so deve ser computada uma vez).
Cada execução deve guardar tempo de execução, o raio da solução e a avaliação da solução pelas métricas: silhueta e índice de Rand ajustado.

Algoritmo de refinamento tem um caso especial.


O resultado final deve ser agregado em uma tabela com respectivas médias e desvios padrão por exeperimento e por algoritmo

