import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout

base = pd.read_csv('corona_tested_individuals_subset_ver_eua.xlsb.csv')
base = base.drop('test_date', axis=1)
base = base.drop('age_60_and_above', axis=1)
base = base.drop('gender', axis=1)
base = base.drop('test_indication', axis=1)
base = base.loc[base.corona_result != 'Other']

sintomas_previsores = base.iloc[:, 0:5].values
resultado_teste = base.iloc[:,5].values


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