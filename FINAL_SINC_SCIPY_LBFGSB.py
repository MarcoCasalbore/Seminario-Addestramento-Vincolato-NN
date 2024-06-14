# -*- coding: utf-8 -*-
"""
Created on Thu May 16 18:02:45 2024

@author: marco
"""

#codice per il prof. Stefano Lucidi

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
import logging

#Se desidera vedere le iterazioni ed il calcolo della f.obj. durante l'algoritmo BFGS disabiliti la riga sotto
logging.getLogger('scipy').setLevel(logging.ERROR)

#Il termine di regolarizzazione della funzione di costo, se lo si aumenta si regolarizza l'approssimazione
#cambiamenti di prestazioni apprezzabili anche nella minimizzazione

lambda_reg = 0.02  

#Da qui in poi costruisco la rete neurale in metodo classico, se talvolta nota delle parentesi commentate alla fine delle righe
#Sono quelle che mi sono servite per fare i conti a mano della dimensione di alcune matrici
#Dato che ricorrevo in problemi dimensionali nel calcolo di alcuni prodotti
#La rete neurale in questione è un MLP [2-10-10-1]

class NeuralNetwork:
    def __init__(self):
        self.input_layer_size = 2
        self.hidden_layer_size_1 = 10
        self.hidden_layer_size_2 = 10
        self.output_layer_size = 1
        
        #mutando con il seed ho ottenuto risultati leggermente diversi tra loro

        np.random.seed(1)
        self.W1 = np.random.randn(self.input_layer_size, self.hidden_layer_size_1)
        self.b1 = np.ones((1, self.hidden_layer_size_1))
        self.W2 = np.random.randn(self.hidden_layer_size_1, self.hidden_layer_size_2)
        self.b2 = np.ones((1, self.hidden_layer_size_2))
        self.W3 = np.random.randn(self.hidden_layer_size_2, self.output_layer_size)
        self.b3 = np.ones((1, self.output_layer_size))
        
    def tanh(self, z):
        return np.tanh(z)

    def identity(self, z):
        return z

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.tanh(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        y_pred = self.identity(self.z3)
        return y_pred

    def cost_function(self, params, X, y): 
        self.W1 = np.reshape(params[:20], (self.input_layer_size, self.hidden_layer_size_1))
        self.b1 = np.reshape(params[20:30], (1, self.hidden_layer_size_1))
        self.W2 = np.reshape(params[30:130], (self.hidden_layer_size_1, self.hidden_layer_size_2))
        self.b2 = np.reshape(params[130:140], (1, self.hidden_layer_size_2))
        self.W3 = np.reshape(params[140:150], (self.hidden_layer_size_2, self.output_layer_size))
        self.b3 = np.reshape(params[150], (1, self.output_layer_size))
        y_pred = self.forward(X)
        
        E = 0.5 * np.sum((y - y_pred) ** 2)

        E_reg = E + (lambda_reg / 2) * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2) + np.sum(self.W3 ** 2))

        return E_reg

    def cost_function_gradient(self, params, X, y):
        self.W1 = np.reshape(params[:20], (self.input_layer_size, self.hidden_layer_size_1))
        self.b1 = np.reshape(params[20:30], (1, self.hidden_layer_size_1))
        self.W2 = np.reshape(params[30:130], (self.hidden_layer_size_1, self.hidden_layer_size_2))
        self.b2 = np.reshape(params[130:140], (1, self.hidden_layer_size_2))
        self.W3 = np.reshape(params[140:150], (self.hidden_layer_size_2, self.output_layer_size))
        self.b3 = np.reshape(params[150], (1, self.output_layer_size))

        y_pred = self.forward(X)

        delta4 = -(y - y_pred)
        dEdW3 = np.dot(self.a2.T, delta4)
        dEdb3 = np.sum(delta4, axis=0)

        delta3 = np.dot(delta4, self.W3.T) * (1 - self.a2 ** 2)
        dEdW2 = np.dot(self.a1.T, delta3)
        dEdb2 = np.sum(delta3, axis=0)

        delta2 = np.dot(delta3, self.W2.T) * (1 - self.a1 ** 2)
        dEdW1 = np.dot(X.T, delta2)
        dEdb1 = np.sum(delta2, axis=0)

        dEdW1 += lambda_reg * self.W1
        dEdW2 += lambda_reg * self.W2
        dEdW3 += lambda_reg * self.W3

        return np.concatenate((dEdW1.ravel(), dEdb1.ravel(), dEdW2.ravel(), dEdb2.ravel(), dEdW3.ravel(), dEdb3.ravel()))

#metodo di ottimizzazione L-BFGS-B con condizioni di Wolfe forte
#ho aggiunto una riga di codice dentro la routine per contare il tempo

    def train(self, X, y):
        initial_params = np.concatenate((self.W1.ravel(), self.b1.ravel(), self.W2.ravel(), self.b2.ravel(), self.W3.ravel(), self.b3.ravel()))
        options = {'maxiter': 10000, 'disp': False, 'gtol': 1e-5, 'ftol': 2.2e-9}
        
        start_time = time.time()  # Inizio del timer
        result = minimize(self.cost_function, initial_params, jac=self.cost_function_gradient, args=(X, y), method='L-BFGS-B', options=options)
        end_time = time.time()  # Fine del timer

        self.W1 = np.reshape(result.x[:20], (self.input_layer_size, self.hidden_layer_size_1))
        self.b1 = np.reshape(result.x[20:30], (1, self.hidden_layer_size_1))
        self.W2 = np.reshape(result.x[30:130], (self.hidden_layer_size_1, self.hidden_layer_size_2))
        self.b2 = np.reshape(result.x[130:140], (1, self.hidden_layer_size_2))
        self.W3 = np.reshape(result.x[140:150], (self.hidden_layer_size_2, self.output_layer_size))
        self.b3 = np.reshape(result.x[150], (1, self.output_layer_size))

        training_time = end_time - start_time  
        print(f"Tempo di addestramento: {training_time:.4f} secondi")
        print("Risultato del modello Scipy+L-BFGS-B:")
        print(result)

#Qui genero i dati di input e definisco la funzione sinc da approssimare 
#np.abs mi è servito solo per stabilizzare il calcolo del gradiente perché faceva impazzire l'algoritmo applicare 
#minimize ad una funzione sqrt (con abs lo rassicuro che siano solo valori positivi)

X = np.random.uniform(-10, 10, (15, 2))
y = np.sin(np.sqrt(X[:, 0]**2 + X[:, 1]**2)) / np.sqrt(X[:, 0]**2 + X[:, 1]**2)
y = y.reshape(-1, 1)

# Creazione e addestramento della rete neurale

nn = NeuralNetwork()
nn.train(X, y)




#Da qui in poi sono solo righe di codice per la stampa della funzinone approssimata

x_visualization = np.linspace(-10, 10, 100)
y_visualization = np.linspace(-10, 10, 100)
X_visualization, Y_visualization = np.meshgrid(x_visualization, y_visualization)
X_visualization_flat = X_visualization.ravel()
Y_visualization_flat = Y_visualization.ravel()
X_visualization_2d = np.vstack([X_visualization_flat, Y_visualization_flat]).T
Z_predicted = nn.forward(X_visualization_2d).reshape(X_visualization.shape)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_visualization, Y_visualization, Z_predicted, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Funzione approssimata con Scipy+L-BFGS-B')
plt.show()
