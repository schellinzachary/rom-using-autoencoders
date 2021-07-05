import numpy as np
import matplotlib.pyplot as plt

rho_predict_lin = np.load('/home/bapu/BA/Neural_Network/Results/rho_predict_lin.npy')
rho_samples = np.load('/home/bapu/BA/Neural_Network/Results/rho_samples.npy')
rho_predict = np.load('/home/bapu/BA/Neural_Network/Results/rho_predict.npy')

plt.ion()
plt.figure(1)
for i in range(241):   
    plt.title('Convolutional Autoencoder')
    plt.plot(rho_predict[i,:],label='Prediction')
    plt.plot(rho_samples[i,:],label='original')
    plt.legend(loc='upper right')
    plt.xlabel('x')
    plt.ylabel('Dichte')
    plt.ylim([1.3,2.7])
    plt.draw()
    plt.pause(0.001)
    plt.clf()
plt.ion()
plt.figure()
for i in range(241):  
    plt.title('Linear Autoencoder') 
    plt.plot(rho_predict_lin[i,:],label='Prediction')
    plt.plot(rho_samples[i,:],label='Original')
    plt.legend(loc='upper right')
    plt.xlabel('x')
    plt.ylabel('Dichte')
    plt.ylim([1.3,2.7])
    plt.draw()
    plt.pause(0.001)
    plt.clf()



plt.figure()
plt.xlabel('x')
plt.ylabel('Dichte')
plt.plot(rho_predict[8,:],label='Prediction')
plt.plot(rho_samples[8,:],label='original')
plt.show()

plt.figure()
plt.xlabel('x')
plt.ylabel('Dichte')
plt.plot(rho_predict_lin[8,:],label='Prediction')
plt.plot(rho_samples[8,:],label='Original')
plt.show()