import pandas as pd
import matplotlib.pyplot as plt
from numpy import argmax

dataset = pd.read_csv('Wine_quality.csv')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
print(dataset.describe())
print('\n*******************************************************************************************************************************************************************************************\n')
print(dataset.corr())
dataset.hist()
plt.show()
x = dataset.iloc[:, :11]
y = dataset.iloc[:, 11].values

dataset.loc[dataset['quality'] < 6, 'quality'] = 0
dataset.loc[dataset['quality'] == 6, 'quality'] = 1
dataset.loc[dataset['quality'] > 6, 'quality'] = 2

x = dataset.iloc[:, :11]
y = dataset.iloc[:, 11].values

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings("ignore")

x = dataset.iloc[:, :11].values
from sklearn.model_selection import train_test_split
X_obucavajuci, X_testirajuci, Y_obucavajuci, Y_testirajuci = train_test_split(x, y, test_size=0.20, random_state=1)
n_features = X_obucavajuci.shape[1]

model = Sequential()

model.add(Dense(11, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(10, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(6, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_obucavajuci, Y_obucavajuci, epochs=1700, batch_size=800, verbose=2)

print("")
loss, acc = model.evaluate(X_obucavajuci, Y_obucavajuci, verbose=2)
loss, acc = model.evaluate(X_testirajuci, Y_testirajuci, verbose=2)

print("--------------- TESTIRAJUCI SKUP ----------------")
row = X_testirajuci[0].tolist()
predikcija = model.predict([row])
print('Izvršena je predkcija: %s (u pitanju je klasa=%d)' % (predikcija, argmax(predikcija)+5))
print(X_testirajuci[0].tolist())
print("Realna klasa=" + str(Y_testirajuci[0]+5))

row = X_testirajuci[1].tolist()
predikcija = model.predict([row])
print('Izvršena je predkcija: %s (u pitanju je klasa=%d)' % (predikcija, argmax(predikcija)+5))
print(X_testirajuci[1].tolist())
print("Realna klasa=" + str(Y_testirajuci[1]+5))

row = X_testirajuci[2].tolist()
predikcija = model.predict([row])
print('Izvršena je predkcija: %s (u pitanju je klasa=%d)' % (predikcija, argmax(predikcija)+5))
print(X_testirajuci[2].tolist())
print("Realna klasa=" + str(Y_testirajuci[2]+5))

row = X_testirajuci[3].tolist()
predikcija = model.predict([row])
print('Izvršena je predkcija: %s (u pitanju je klasa=%d)' % (predikcija, argmax(predikcija)+5))
print(X_testirajuci[3].tolist())
print("Realna klasa=" + str(Y_testirajuci[3]+5))

row = X_testirajuci[4].tolist()
predikcija = model.predict([row])
print('Izvršena je predkcija: %s (u pitanju je klasa=%d)' % (predikcija, argmax(predikcija)+5))
print(X_testirajuci[4].tolist())
print("Realna klasa=" + str(Y_testirajuci[4]+5))

row = X_testirajuci[5].tolist()
predikcija = model.predict([row])
print('Izvršena je predkcija: %s (u pitanju je klasa=%d)' % (predikcija, argmax(predikcija)+5))
print(X_testirajuci[5].tolist())
print("Realna klasa=" + str(Y_testirajuci[5]+5))

row = X_testirajuci[6].tolist()
predikcija = model.predict([row])
print('Izvršena je predkcija: %s (u pitanju je klasa=%d)' % (predikcija, argmax(predikcija)+5))
print(X_testirajuci[6].tolist())
print("Realna klasa=" + str(Y_testirajuci[6]+5))

row = X_testirajuci[7].tolist()
predikcija = model.predict([row])
print('Izvršena je predkcija: %s (u pitanju je klasa=%d)' % (predikcija, argmax(predikcija)+5))
print(X_testirajuci[7].tolist())
print("Realna klasa=" + str(Y_testirajuci[7]+5))

row = X_testirajuci[8].tolist()
predikcija = model.predict([row])
print('Izvršena je predkcija: %s (u pitanju je klasa=%d)' % (predikcija, argmax(predikcija)+5))
print(X_testirajuci[8].tolist())
print("Realna klasa=" + str(Y_testirajuci[8]+5))

row = X_testirajuci[9].tolist()
predikcija = model.predict([row])
print('Izvršena je predkcija: %s (u pitanju je klasa=%d)' % (predikcija, argmax(predikcija)+5))
print(X_testirajuci[9].tolist())
print("Realna klasa=" + str(Y_testirajuci[9]+5))
