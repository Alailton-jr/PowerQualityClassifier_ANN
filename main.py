import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
import numpy as np
from sklearn.preprocessing import StandardScaler

from src import generate_data
from src import compute_stft, compute_cwt, compute_st, compute_hht

def get_model(input_shape, n_classes):
    model = Sequential([
        Input(input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='valid'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='valid'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_cnn(X, labels, n_epochs=100, batch_size=4):
    
    # One-Hot Adjusment
    labels, y, inv = list(np.unique(labels, return_index=True, return_inverse=True))
    maskSort = np.argsort(y)
    labels = labels[maskSort]
    maping = {v: i for i, v in enumerate(maskSort)}
    y = np.vectorize(lambda x: maping[x])(inv)
    Y = np.zeros((y.shape[0], np.max(y)+1))
    Y[np.arange(y.shape[0]), y] = 1

    # Shuffle the Data
    perm = np.random.permutation(X.shape[0])
    X = X[perm]
    Y = Y[perm]

    # Normalize the Data
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1,1)).reshape(X.shape)
    X = X[:, :, :, np.newaxis] 
    # Y = Y[:, :, np.newaxis]

    # Split the data in Train and Test (80% Train, 20% Test)
    mask = np.random.choice([True, False], size=(X.shape[0]), p=[0.8, 0.2])

    # Create the Model
    cnn_model = get_model(X[0].shape, Y[0].shape[0])
    cnn_model.summary()

    # Train the Model
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='accuracy',
        min_delta=0, # minimum improvement
        patience=6, # how many epochs to wait
        restore_best_weights=True # keep the best model
    )
    cnn_model.fit(X[mask], Y[mask], epochs=n_epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping])

    # Evaluate the Model
    cnn_model.evaluate(X[~mask], Y[~mask])

    return cnn_model

if __name__ == '__main__':

    data, labels = generate_data(
        n_samples=1000,
        duration=1/60,
        fs=15360,
        t_ini=0
    )

    stft_data = compute_stft(data)
    stft_model = train_cnn(stft_data, labels)

    cwt_data = compute_cwt(data)
    cwt_model = train_cnn(cwt_data, labels)

    st_data = compute_st(data)
    st_model = train_cnn(st_data, labels)

    hht_data = compute_hht(data)
    hht_model = train_cnn(hht_data, labels)



