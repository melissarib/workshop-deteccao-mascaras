import os
import numpy as np
import cv2 as cv
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import warnings

# Pandas, que serve para análise exploratória de dados;
# ela nos permite ler, manipular, agregar e plotar os 
# dados em poucos passos por meio de *dataframes*. 
# Isso significa que depois do computador identificar 
# que as imagens são matrizes, a gente consegue mexer 
# nessas matrizes por dentro.


def load_dataframe():
    '''
    Carrega um dataframe Pandas com as imagens para o treinamento do modelo
    '''
    dados = {
        "ARQUIVO": [],
        "ROTULO": [],
        "ALVO": [],
        "IMAGEM": [],
    }

    com_mascara = os.listdir(f"imagens{os.sep}maskon")
    sem_mascara = os.listdir(f"imagens{os.sep}maskoff")

    for arquivo in com_mascara:
        dados["ARQUIVO"].append(f"imagens{os.sep}maskon{os.sep}{arquivo}")
        dados["ROTULO"].append(f"Com mascara")
        dados["ALVO"].append(1)
        img = cv.cvtColor(cv.imread(f"imagens{os.sep}maskon{os.sep}{arquivo}"), cv.COLOR_BGR2GRAY).flatten()
        dados["IMAGEM"].append(img)
        
    for arquivo in sem_mascara:
        dados["ARQUIVO"].append(f"imagens{os.sep}maskoff{os.sep}{arquivo}")
        dados["ROTULO"].append(f"Sem mascara")
        dados["ALVO"].append(0)
        img = cv.cvtColor(cv.imread(f"imagens{os.sep}maskoff{os.sep}{arquivo}"), cv.COLOR_BGR2GRAY).flatten()
        dados["IMAGEM"].append(img)
        
    dataframe = pd.DataFrame(dados)

    return dataframe


# O reconhecimento de padrões do aprendizado de máquina ocorre de acordo
# com nos dados que damos pra essa máquina analisar. No nosso caso,
# separamos as imagens pra máquina poder aprender o que é o que 
# e depois treinar sobre isso. Conhece o jogo "Eu sou...?"


def train_test(dataframe):
    '''
    Divide o dataframe em conjunto de treino e teste
    '''
    X = list(dataframe["IMAGEM"])
    y = list(dataframe["ALVO"])

    return train_test_split(X, y, train_size=0.40, random_state=13)

# Encontrar um caminho para condensar a informação contida em várias
# variáveis originais em um conjunto menor. Queremos ficar só com
# "os principais componentes" das nossas imagens.

def pca_model(X_train):
    '''
    PCA para extração de features das imagens
    '''
    pca = PCA(n_components=30)
    pca.fit(X_train)
    
    return pca


# Setando informações para o modelo classificador
# identificar se uma pessoa está ou não portando uma máscara.

def knn(X_train, y_train):

    warnings.filterwarnings("ignore")
    '''
    Modelo K-Nearest Neighbors
    '''
    grid_params = {
    "n_neighbors": [2, 3, 5, 11, 19, 23, 29],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattam", "cosine", "l1", "l2"]
    }

    knn_model = GridSearchCV(KNeighborsClassifier(), grid_params, refit=True)

    knn_model.fit(X_train, y_train)

    return knn_model

