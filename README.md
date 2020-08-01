# HERE YOU CAN FIND ALL THE PYTHON CODE USED TO CREATE, TRAIN AND IMPROVE THE PATHOGEN's NEURAL NETWORK.
# The database used was taken from that site: https://data.gov.il/dataset/covid-19/resource/74216e15-f740-4709-adb7-a6fb0955a048, which is the official data site of the government of Israel.
# As this database did not have other symptoms presented by the people who performed the coronavirus test, only 5 predictive attributes were used.
# The predictive attributes are: Cough, Fever, Sore Throat, Headache and Shortness of breath.
# As we didn't have much time to test different parameters, we only studied variation in the amount of neurons used in the hidden layers.
# The other parameters were chosen through readings of articles that spoke of neural networks used for similar classifications.
# When this network is actually used commercially, we will carry out a new training using more parameters to increase the effectiveness of the prediction, which at the moment is between 92% to 94% correct.

```import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


#PREPARING THE DATABASE FOR TRAINING, REMOVING COLUMNS THAT ARE NOT NECESSARY.
base = pd.read_csv('corona_tested_individuals_subset_ver_eua.xlsb.csv')
base = base.drop('test_date', axis=1)
base = base.drop('age_60_and_above', axis=1)
base = base.drop('gender', axis=1)
base = base.drop('test_indication', axis=1)
base = base.loc[base.corona_result != 'Other']
sintomas_previsores = base.iloc[:,0:5].values
resultado_teste = base.iloc[:,5].values

#CREATING THE ARTIFICIAL NEURAL NETWORK FUNCTION
def criarRede(optimizer, loos, kernel_initializer, activation, neurons):
    classificador = Sequential()
    classificador.add(Dense(units = neurons, activation = activation,
                        kernel_initializer = kernel_initializer, input_dim = 5))
    classificador.add(Dropout(0.1))
    classificador.add(Dense(units = neurons, activation = activation,
                        kernel_initializer = kernel_initializer))
    classificador.add(Dropout(0.1))
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    classificador.compile(optimizer = optimizer, loss = loos,
                      metrics = ['binary_accuracy'])
    return classificador

#Using GridSearch to find the best parameters for the neural network using this database.

classificador = KerasClassifier(build_fn = criarRede)
parametros = {'batch_size': [2],
              'epochs': [5],
              'optimizer': ['adamax','adam'],
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

#EFFECTIVELY CREATING THE NEURAL NETWORK AND SAVING THE BEST WEIGHTS FOUND, AS WELL AS ALL ITS STRUCTURE.

classificador = Sequential()
classificador.add(Dense(units = 3, activation = 'relu',
                        kernel_initializer = 'RandomUniform', input_dim = 5))
classificador.add(Dropout(0.1))
classificador.add(Dense(units = 3, activation = 'relu',
                        kernel_initializer = 'RandomUniform'))
classificador.add(Dropout(0.1))
classificador.add(Dense(units = 1, activation = 'sigmoid'))
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])
classificador.fit(sintomas_previsores,resultado_teste,batch_size=2, epochs=5)

r_n_covid19 = classificador.to_json()
with open('IA_PATHOGEN.json','w') as json_file:
    json_file.write(r_n_covid19)
classificador.save_weights('IA_PATHOGEN.h5')
