import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

base = pd.read_csv('corona_tested_individuals_subset_ver_eua.xlsb.csv')
base = base.drop('test_date', axis=1)
base = base.drop('age_60_and_above', axis=1)
base = base.drop('gender', axis=1)
base = base.drop('test_indication', axis=1)
base = base.loc[base.corona_result != 'Other']

sintomas_previsores = base.iloc[:,0:5].values
resultado_teste = base.iloc[:,5].values


def criarRede(optimizer, loos, kernel_initializer, activation, neurons):
    classificador = Sequential()
    classificador.add(Dense(units = neurons, activation = activation,
                        kernel_initializer = kernel_initializer, input_dim = 4))
    classificador.add(Dropout(0.1))
    classificador.add(Dense(units = neurons, activation = activation,
                        kernel_initializer = kernel_initializer))
    classificador.add(Dropout(0.1))
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    classificador.compile(optimizer = optimizer, loss = loos,
                      metrics = ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede)
parametros = {'batch_size': [2],
              'epochs': [5],
              'optimizer': ['adamax', 'adam'],
              'loos': ['binary_crossentropy'],
              'kernel_initializer': ['random_uniform'],
              'activation': ['relu'],
              'neurons': [3,6]}
grid_search = GridSearchCV(estimator = classificador,
                           param_grid = parametros,
                           scoring = 'accuracy',
                           cv = 5)
grid_search = grid_search.fit(sintomas_previsores,resultado_teste)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_

print(melhor_precisao,melhores_parametros)