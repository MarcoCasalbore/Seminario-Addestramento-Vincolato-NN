# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:51:51 2024

@author: marco
"""

#codice per il prof. Stefano Lucidi


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import time

#Qui genero i dati di input e definisco la funzione sinc da approssimare 

def sinc_function(x1, x2):
    return np.sin(np.sqrt(x1**2 + x2**2)) / np.sqrt(x1**2 + x2**2)


np.random.seed(0)
num_samples = 15
x1 = np.random.uniform(-10, 10, num_samples)
x2 = np.random.uniform(-10, 10, num_samples)
y = sinc_function(x1, x2)

#qui inizializzo la rete neurale in questione MLP [2-10-10-1]

model = keras.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(10, activation='tanh'),
    layers.Dense(10, activation='tanh'),
    layers.Dense(1, activation='linear')
])

#Da qui faccio compilare il modello con l'ottimizzatore ADAM ed ho aggiunto delle righe di testo per stampare il tempo di ottimizzazione
#Per confrontare con l'altro metodo le prestazioni

model.compile(optimizer='adam', loss='mean_squared_error')

start_time = time.time()
history = model.fit(np.column_stack((x1, x2)), y, epochs=1000, verbose=0)
end_time = time.time()  

loss = model.evaluate(np.column_stack((x1, x2)), y, verbose=0)
training_time = end_time - start_time  

print("Loss del modello Keras+ADAM:", loss)
print(f"Tempo di addestramento: {training_time:.4f} secondi")

def predict(inputs):
    return model(inputs)


#Da qui in poi sono solo righe di codice per la stampa della funzinone approssimata


x1_grid = np.linspace(-10, 10, 100)
x2_grid = np.linspace(-10, 10, 100)
x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid)
inputs_mesh = np.column_stack((x1_mesh.flatten(), x2_mesh.flatten()))

#Qui ho fatto diversi tentativi per rimuovere diversi warning che mi dava il codice in compilazione riguardanti la forma delle variabili in input
#A quanto pare ho risolto definendo con una funzione prima la predizione e poi richiamandola qui in basso per la moltiplicazione tensoriale

#predictions = model.predict(inputs_mesh).reshape(x1_mesh.shape)
#predictions = model.predict(inputs_mesh, batch_size=1024).reshape(x1_mesh.shape)
inputs_tensor = tf.convert_to_tensor(inputs_mesh, dtype=tf.float32)
predictions = predict(inputs_tensor).numpy().reshape(x1_mesh.shape)


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_mesh, x2_mesh, predictions, cmap='viridis')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Predicted y')
ax.set_title('Funzione approssimata con Keras+ADAM')
plt.show()
