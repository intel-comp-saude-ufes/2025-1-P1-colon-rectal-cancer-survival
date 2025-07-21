# Modelagem de Sobrevida em Câncer Colorretal via Classificação Supervisionada 

O câncer colorretal (CCR) é uma das principais causas de mortalidade global, 
com crescente incidência no Brasil. Este estudo investiga a aplicação de algoritmos 
de aprendizado de máquina (MLP, Random Forest e XGBoost) para predição da 
sobrevida de pacientes com CCR, utilizando dados do Registro Hospitalar de 
Câncer do Estado de São Paulo (RHC-SP). Foram definidos quatro cenários 
preditivos, categorizando a sobrevida em intervalos clínicos significativos. 
O pré-processamento incluiu discretização de variáveis contínuas e recodificação 
de atributos categóricos. Os modelos foram avaliados quanto à acurácia, 
F1-score, precisão, recall e AUC. Os resultados indicam acurácia superior a 
85\% para predição de sobrevida em 1 ano e cerca de 76\% para 5 anos. 
Em um cenário multiclasse, a acurácia caiu para 60\%, com maior desempenho 
nas classes extremas (sobrevida menor que 1 ano e sobrevida maior que 5 anos). 
Estes achados destacam o potencial dos modelos para auxiliar decisões 
clínicas no manejo do CCR.

# Apresentação do Trabalho

Um vídeo com a discussão da metodologia e resultados pode ser acessado em [link](https://drive.google.com/file/d/1DqzOIF4ToQ-FDEHRwJkUYorke53B8MeR/view).

# Artigo do Trabalho

O artigo contendo a discussão completa da metodologia e resultados pode ser acessado 
na raiz do repositório, no arquivo **ColonRectalCancerSurvival.pdf**.

# Organização do Repositório

Este repositório contém os scripts, dados e resultados relacionados à avaliação de diferentes modelos de machine learning (MLP, Random Forest, XGBoost) aplicados à previsão de sobrevida com base em dados clínicos e epidemiológicos.

- **mlp_gridsearch.py**: Script Python para treinamento de redes neurais MLP com `GridSearchCV` e geração de modelos otimizados e métricas.

- **randomforest_gridsearch.py**: Script para treinamento de modelos Random Forest com `GridSearchCV` e geração de modelos otimizados e métricas.

- **xgboost_gridsearch.py**: Script que realiza treinamento de modelos XGBoost com `GridSearchCV` e geração de modelos otimizados e métricas.

- **decision_making**: Pasta com código para o método **A-TOPSIS** [link](https://github.com/paaatcha/decision-making), utilizado para avaliação dos modelos

- **data/**: Diretório contendo os dados brutos e scripts de pré-processamento.
  - **dataset_clean.csv**: Base de dados gerada por [link](https://www.nature.com/articles/s41598-023-35649-9), usada como base para a geração dos datasets neste trabalho.
  - **dictionary_RHC-1.pdf**: Dicionário de variáveis com descrições dos campos presentes no dataset.
  - **s41598-023-35649-9_Maria_Paula_Curado.pdf**: Artigo científico base que fundamenta a análise realizada.
  - **preprocessing1.R** a **preprocessing4.R**: Scripts em R utilizados para preparação e transformação dos dados, incluindo discretização e limpeza.
  - **datasets_gerados/**: Contém os datasets finais gerados para os diferentes cenários de análise.
    - **dataset_cenario1.csv** a **dataset_cenario4.csv**: Bases pré-processadas correspondentes a diferentes cenários de predição.

- **resultados/**: Diretório com os resultados gerados pelos modelos aplicados a cada cenário.
  - **dataset_cenario1/** a **dataset_cenario4/**: Resultados organizados por cenário.
    - **_mlp/**: Contém saídas do modelo MLP.
      - **curva_roc.png**: Curva ROC gerada para o modelo.
      - **cv_resultados.csv**: Resultados de validação cruzada.
      - **matriz_confusao.png**: Matriz de confusão para o melhor modelo.
      - **melhor_modelo.pkl**: Melhor modelo MLP treinado salvo.
      - **metricas_avaliacao.txt**: Métricas finais de avaliação (F1, acurácia, etc.).
    - **_randomforest/**: Resultados análogos aos da pasta _mlp, mas para o modelo Random Forest.
    - **_xgboost/**: Resultados análogos aos da pasta _mlp, mas para o modelo XGBoost.
  - **curva_roc_cenarios_123_integrados_randomforest.png**: Curva ROC consolidada dos cenários 1, 2 e 3 para Random Forest.
  - **curva_roc_cenarios_123_integrados_xgboost.png**: Curva ROC consolidada dos cenários 1, 2 e 3 para XGBoost.
  - **curva_roc_cenarios_123_integrados_mlp.png**: Curva ROC consolidada dos cenários 1, 2 e 3 para MLP.

# Instalação de Dependências

Para execução dos scripts de treinamento dos modelos, é preciso instalar os seguintes 
pacotes:

``````
pip install pandas scikit-learn numpy matplotlib joblib
``````

# Execução

- Execução do treinamento e geração de modelos/métricas para MLP:

``````
python mlp_gridsearch.py
``````

- Execução do treinamento e geração de modelos/métricas para XGboost:

``````
python xgboost_gridsearch.py
``````

- Execução do treinamento e geração de modelos/métricas para Random Forest:

``````
python randomforest_gridsearch.py
``````
## Autores

* **Leandro Furlam Turi** - [https://github.com/leandrofturi](https://github.com/leandrofturi)
* **Daniel Ribeiro Trindade** - [https://github.com/danielrt](https://github.com/danielrt)
