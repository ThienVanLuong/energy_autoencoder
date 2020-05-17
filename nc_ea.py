"""
Implementation of NC-EA (energy autoencoder concept) with single user. Detail is in the paper:
- T. V. Luong, Y. Ko, N. A. Vien, M. Matthaiou and H. Q. Ngo, "Deep Energy Autoencoder for Noncoherent Multicarrier MU-SIMO Systems," 
in IEEE Transactions on Wireless Communications, doi: 10.1109/TWC.2020.2979138.  
  
- Created by Thien Van Luong, Research Fellow at University of Southampton, UK.
- Contact: thien.luong@soton.ac.uk, https://tvluong.wordpress.com 
- Requirements: Tensorflow 1.x
"""

import tensorflow as tf
import numpy as np
from keras.layers import Lambda, Add, Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from utils import generate_one_hot_vectors

"""=======NC-EA system parameters======="""
N = 4                   # number of sub-carriers per block
M = 4                   # number of messages s
m = int(np.log2(M))     # number of bits per message s/codeword x
L = 1                   # number of receive antennas (training)
L_test = L              # number of receive antennas (testing)

# network config, train, test data
norm_type = 1               # type of encoder normalization: 0 per data sample norm, 1 per batch norm
SNRdB_train = 20            # training SNR in dB, 
SNRdB_test = SNRdB_train    # tesing SNR in dB, to get results for different SNRs, please change SNRdB_train
b_size = 128                # batch_size
n_epoch = 100               # number of epochs
l_rate = 0.001              # learning rate
hidden_layer_dec = np.array([16])   # hidden layer setup of decoder, len(hidden_layer_dec) = number of hidden layers
f_loss = 'mean_squared_error'       # loss function: categorical_crossentropy/ mean_squared_error
act_enc = 'tanh'                    # activation of encoder
act_dec = 'tanh'                    # activation of decoder
train_size = 20000                  # number of training samples
test_size = 100000                  # number of testing samples

# other parameters
R = m / N                               # bit rate ~ spectral efficiency
snr_train = 10 ** (SNRdB_train / 10.0)  # training SNR
power_ratio = 1
noise_std = np.sqrt(1 / (2 * R * snr_train * power_ratio))
n_hidden_layer_dec = hidden_layer_dec.shape[0]

# channel layer function for training
def channel(Z):
    Y = 0
    for l in range(L):
        H_R = K.random_normal(K.shape(Z), mean=0, stddev=1) / np.sqrt(2)
        H_I = K.random_normal(K.shape(Z), mean=0, stddev=1) / np.sqrt(2)
        real = H_R * Z
        imag = H_I * Z
        noise_r = K.random_normal(K.shape(real), mean=0, stddev=noise_std)
        noise_i = K.random_normal(K.shape(imag), mean=0, stddev=noise_std)

        real = Add()([real, noise_r])
        imag = Add()([imag, noise_i])
        Y = Y + (K.pow(real, 2) + K.pow(imag, 2))/L
        
    return Y

# channel layer function for testing
def channel_test(Z, noise_std, test_size):
    with tf.compat.v1.Session() as sess:
        Y = 0
        for l in range(L_test):
            H_R = np.random.normal(0, 1, (test_size, N)) / np.sqrt(2)
            H_I = np.random.normal(0, 1, (test_size, N)) / np.sqrt(2)
    
            real = H_R * Z
            imag = H_I * Z
    
            noise_r = K.random_normal(K.shape(real), mean=0, stddev=noise_std)
            noise_i = K.random_normal(K.shape(imag), mean=0, stddev=noise_std)
    
            real = real + noise_r
            imag = imag + noise_i
            Y = Y + (K.pow(real, 2) + K.pow(imag, 2))/L_test
        Y = sess.run(Y)
    
    return Y

# train data, one-hot vector
train_data = generate_one_hot_vectors(M, train_size, get_label=False)

"""energy autoencoder model -  NC-EA"""
# encoder network
X = Input(shape=(M,))
enc = Dense(N, activation=act_enc)(X) 
if norm_type == 0:
    Z = Lambda(lambda x: np.sqrt(N) * K.l2_normalize(x, axis=1))(enc)
else:
    Z = Lambda(lambda x: x / tf.sqrt(tf.reduce_mean(tf.square(x))))(enc)

# received signal through channel layer
Y = Lambda(lambda x: channel(x))(Z)

# decoder network
dec = Dense(hidden_layer_dec[0], activation=act_dec)(Y)
if n_hidden_layer_dec > 1:
    for n in range(n_hidden_layer_dec - 1):
        dec = Dense(hidden_layer_dec[n + 1], activation=act_dec)(dec)
X_hat = Dense(M, activation='softmax')(dec)  # estimate of X

# training
AE = Model(X, X_hat)
AE.compile(optimizer=Adam(lr=l_rate), loss=f_loss, metrics=['accuracy'])  # categorical_crossentropy mean_squared_error
H = AE.fit(train_data, train_data, epochs=n_epoch, batch_size=b_size, verbose=2)

# model encoder and decoder
encoder = Model(X, Z)
X_enc = Input(shape=(N,))
deco = AE.layers[-n_hidden_layer_dec - 1](X_enc)  # first layer of decoder
for n in range(n_hidden_layer_dec):  # hidden and last layers of decoder
    deco = AE.layers[-n_hidden_layer_dec + n](deco)
decoder = Model(X_enc, deco)

# generate learned codewords x
test_label = np.arange(M)
test_data = []
for i in test_label:
    temp = np.zeros(M)
    temp[i] = 1
    test_data.append(temp)
test_data = np.array(test_data)
U = encoder.predict(test_data) # codewords x
print('==>leaned codewords: \n', np.round(U,2))

# generate test data
test_data, test_label = generate_one_hot_vectors(M, test_size, get_label=True)
test_bit = (((test_label[:, None] & (1 << np.arange(m)))) > 0).astype(int)

# BLER calculation and plot
EbNodB_range = list(np.linspace(SNRdB_test, SNRdB_test, 1))
BLER = [None] * len(EbNodB_range)
BER = [None] * len(EbNodB_range)
for n in range(0, len(EbNodB_range)):
    EbNo = 10 ** (EbNodB_range[n] / 10.0)
    noise_std = np.sqrt(1 / (2 * R * EbNo * power_ratio))

    no_errors = 0
    Z = encoder.predict(test_data)
    Y = channel_test(Z, noise_std, test_size)
    X_hat = decoder.predict(Y)
    pred_output = np.argmax(X_hat, axis=1)
    
    re_bit = (((pred_output[:, None] & (1 << np.arange(m)))) > 0).astype(int)
    bit_errors = (re_bit != test_bit).sum()
    BER[n] = bit_errors / test_size / m

    no_errors = (pred_output != test_label)
    no_errors = no_errors.astype(int).sum()
    BLER[n] = no_errors / test_size

    print('SNR:', EbNodB_range[n], 'BLER:', BLER[n])
    print('SNR:', EbNodB_range[n], 'BER:', BER[n])
