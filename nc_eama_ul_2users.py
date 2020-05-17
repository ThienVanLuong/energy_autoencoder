"""
Implementation of NC-EAMA uplink(energy autoencoder concept) with 2 users. Detail is in the paper:
- T. V. Luong, Y. Ko, N. A. Vien, M. Matthaiou and H. Q. Ngo, "Deep Energy Autoencoder for Noncoherent Multicarrier MU-SIMO Systems," 
in IEEE Transactions on Wireless Communications, doi: 10.1109/TWC.2020.2979138.  
  
- Created by Thien Van Luong, Research Fellow at University of Southampton, UK.
- Contact: thien.luong@soton.ac.uk, https://tvluong.wordpress.com 
- Requirements: Tensorflow 2.0, Keras 2.3.1
"""

import tensorflow as tf
import numpy as np
from keras.layers import Lambda, Add, Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.losses import mse
from utils import generate_one_hot_vectors

"""=======NC-EAMA uplink 2-users system parameters======="""
J = 2                   # number of users
N = 4                   # number of sub-carriers per block
M = 2                   # number of messages s
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
hidden_layer_dec = np.array([8,16])   # hidden layer setup of decoder, len(hidden_layer_dec) = number of hidden layers
f_loss = 'mean_squared_error'       # loss function: categorical_crossentropy/ mean_squared_error
act_enc = 'tanh'                    # activation of encoder
act_dec = 'tanh'                    # activation of decoder
train_size = 20000                  # number of training samples
test_size = 100000                  # number of testing samples
loss_scale = 5                      # loss scaling factor, lambda in paper

# other parameters
R = m / N                               # bit rate ~ spectral efficiency
snr_train = 10 ** (SNRdB_train / 10.0)  # training SNR
power_ratio = 1
noise_std = np.sqrt(1 / (2 * R * snr_train * power_ratio))
n_hidden_layer_dec = hidden_layer_dec.shape[0]

def channel(Z1, Z2):
    Y = 0
    for l in range(L):
        H1_R = K.random_normal(K.shape(Z1),mean=0,stddev=1)/np.sqrt(2)
        H1_I = K.random_normal(K.shape(Z1),mean=0,stddev=1)/np.sqrt(2)
        H2_R = K.random_normal(K.shape(Z2),mean=0,stddev=1)/np.sqrt(2)
        H2_I = K.random_normal(K.shape(Z2),mean=0,stddev=1)/np.sqrt(2)

        real = H1_R * Z1 + H2_R * Z2
        imag = H1_I * Z1 + H2_I * Z2

        noise_r = K.random_normal(K.shape(real),mean=0,stddev=noise_std)
        noise_i = K.random_normal(K.shape(imag),mean=0,stddev=noise_std)

        real = Add()([real, noise_r])
        imag = Add()([imag, noise_i])

        Y = Y + (K.pow(real,2)+K.pow(imag,2))/L

    return Y

def channel_test(Z, noise_std, test_size):
    with tf.compat.v1.Session() as sess:
        Y = 0
        for l in range(L_test):
            H1_R = np.random.normal(0, 1, (test_size, N))/np.sqrt(2)
            H1_I = np.random.normal(0, 1, (test_size, N))/np.sqrt(2)
            H2_R = np.random.normal(0, 1, (test_size, N))/np.sqrt(2)
            H2_I = np.random.normal(0, 1, (test_size, N))/np.sqrt(2)
    
            real = H1_R * Z1 + H2_R * Z2
            imag = H1_I * Z1 + H2_I * Z2
    
            noise_r = K.random_normal(K.shape(real), mean=0, stddev=noise_std)
            noise_i = K.random_normal(K.shape(imag), mean=0, stddev=noise_std)
    
            real = real + noise_r
            imag = imag + noise_i
    
            Y = Y + (K.pow(real,2)+K.pow(imag,2))/L_test
    
        Y = sess.run(Y)
    return Y

# train data, one-hot vector
train_data_1 = generate_one_hot_vectors(M, train_size, get_label=False)
train_data_2 = generate_one_hot_vectors(M, train_size, get_label=False)
train_data = np.concatenate([train_data_1, train_data_2], axis=1)

"""energy autoencoder model -  NC-EAMA uplink 2 users"""
# encoder of user 1
X1 = Input(shape=(M,))
enc1 = Dense(N, use_bias=True, activation=act_enc)(X1) # be careful
if norm_type == 0:
    Z1 = Lambda(lambda x: np.sqrt(N) * K.l2_normalize(x, axis=1))(enc1)
else:
    Z1 = Lambda(lambda x: x/tf.sqrt(tf.reduce_mean(tf.square(x))))(enc1)
    
# encoder of user 2
X2 = Input(shape=(M,))
enc2 = Dense(N, use_bias=True, activation=act_enc)(X2) # be careful
if norm_type == 0:
    Z2 = Lambda(lambda x: np.sqrt(N) * K.l2_normalize(x, axis=1))(enc2)
else:
    Z2 = Lambda(lambda x: x/tf.sqrt(tf.reduce_mean(tf.square(x))))(enc2)

# received signal through channel layer
Y = Lambda(lambda x: channel(x[0],x[1]))([Z1, Z2])

# decoder
dec = Dense(hidden_layer_dec[0], activation=act_dec)(Y)
if n_hidden_layer_dec>1:
    for n in range(n_hidden_layer_dec-1):
        dec = Dense(hidden_layer_dec[n+1], activation=act_dec)(dec)

X1_hat = Dense(M, activation='softmax')(dec) # estimate of X1
X2_hat = Dense(M, activation='softmax')(dec)

AE = Model(inputs = [X1,X2], outputs = [X1_hat,X2_hat])

# loss design
l1 = mse(X1,X1_hat)
l2 = mse(X2,X2_hat)
recon_loss = l1+l2 # mse = K.mean(K.square(y_true - y_pred))
dev_loss = K.pow(l1-recon_loss/2,2)+K.pow(l2-recon_loss/2,2)
total_loss = K.mean(recon_loss + loss_scale*dev_loss)

# training
AE.add_loss(recon_loss)
AE.compile(optimizer=Adam(lr=l_rate))
AE.fit([train_data_1,train_data_2],
       epochs=n_epoch, batch_size=b_size, verbose=2)

# model encoder and decoder
encoder = Model(inputs = [X1,X2], outputs = [Z1,Z2])
encoder_1 = Model(X1,Z1)
encoder_2 = Model(X2,Z2)
AE1 = Model(inputs = [X1,X2], outputs = X1_hat)
AE2 = Model(inputs = [X1,X2], outputs = X2_hat)

X_enc = Input(shape=(N,))
deco_1 = AE1.layers[-n_hidden_layer_dec-1](X_enc) # first layer of decoder
deco_2 = AE2.layers[-n_hidden_layer_dec-1](X_enc) # first layer of decoder

for n in range(n_hidden_layer_dec): # hidden and last layers of decoder
    deco_1 = AE1.layers[-n_hidden_layer_dec+n](deco_1)
    deco_2 = AE2.layers[-n_hidden_layer_dec+n](deco_2)

decoder_1 = Model(X_enc, deco_1)
decoder_2 = Model(X_enc, deco_2)

# learned constellations/codewords
test_label = np.arange(M)
test_data = []
for i in test_label:
    temp = np.zeros(M)
    temp[i] = 1
    test_data.append(temp)
test_data = np.array(test_data)
U1 = encoder_1.predict(test_data)
print('Leaned codewords of user 1: \n', U1)
U2 = encoder_2.predict(test_data)
print('Leaned codewords of user 2: \n', U2)

# generate test data
test_data_1, test_label_1 = generate_one_hot_vectors(M, test_size, get_label=True)
test_data_2, test_label_2 = generate_one_hot_vectors(M, test_size, get_label=True)
test_data = np.concatenate([test_data_1, test_data_2], axis=1)

# BLER calculation and plot
EbNodB_range = list(np.linspace(SNRdB_test, SNRdB_test, 1))
BLER = [None] * len(EbNodB_range)   # overall BLER
BLER1 = [None] * len(EbNodB_range)  # BLER of user 1
BLER2 = [None] * len(EbNodB_range)  # BLER of user 2

for n in range(0, len(EbNodB_range)):
    EbNo = 10 ** (EbNodB_range[n] / 10.0)
    noise_std = np.sqrt(1/(2*R*EbNo))

    no_errors = 0
    Z1,Z2 = encoder.predict([test_data_1, test_data_2])
    Y = channel_test([Z1, Z2], noise_std,test_size)
    X1_hat = decoder_1.predict(Y)
    X2_hat = decoder_2.predict(Y)
    pred_output_1 = np.argmax(X1_hat, axis=1)
    pred_output_2 = np.argmax(X2_hat, axis=1)
    no_errors = (pred_output_1 != test_label_1).astype(int).sum() + (pred_output_2 != test_label_2).astype(int).sum()

    no_errors_1 = (pred_output_1!=test_label_1).astype(int).sum()
    no_errors_2 = (pred_output_2!=test_label_2).astype(int).sum()

    BLER[n] = no_errors/test_size/J
    BLER1[n] = no_errors_1/test_size
    BLER2[n] = no_errors_2/test_size
    print('SNR:', EbNodB_range[n], 'BLER:', BLER[n], 'BLER1:', BLER1[n], 'BLER2:', BLER2[n])
