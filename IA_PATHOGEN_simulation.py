import numpy as np
from keras.models import model_from_json


#LOADING THE NEURAL NETWORK
arch = open('IA_PATHOGEN.json', 'r')
struct_network = arch.read()
arch.close()

clasf = model_from_json(struct_network)
clasf.load_weights('IA_PATHOGEN.h5')

#CLASSIFYING A SYMPTOM
novos_sintomas = np.array([[1,1,0,1,1]]) # Cough = 1, Fever = 1, Sore Throat = 0, Shortness Of Breath = 1 and Headache = 1. Yes = 1 and No = 0
resultado_covid = clasf.predict(novos_sintomas)
resultado_covid = (resultado_covid > 0.5)

if resultado_covid:
    print("From the symptoms presented, it is possible that you have COVID 19. Please look for the nearest health Center.")
    print("A partir dos sintomas apresentados, é possível que você tenha o COVID 19. Procure o centro de saúde mais próximo.")
else:
    print("From the symptoms presented, it is possible that you do not have COVID19. In the meantime, continue to assess your symptoms. If there is a worsening in them, look for the nearest health center.")
    print("A partir dos sintomas apresentados, é possível que você não tenha COVID19. Entretanto, continue avaliando os seus sintomas. Se houver uma piora nos mesmos, procure o posto de saúde mais próximo.")

