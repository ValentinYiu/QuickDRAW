import numpy as np
import matplotlib.pyplot as plt

losses_validation = np.load('draw_validation_loss6.npz')
losses_train = np.load('draw_train_loss6.npz')

Lxsv = losses_validation['Lxs_validation'][2:]
Lzsv = losses_validation['Lzs_validation'][2:]

Lxst = losses_train['Lxs_train'][2:]
Lzst = losses_train['Lzs_train'][2:]

f=plt.figure()
plt.plot(Lxsv,label='Validation')
plt.plot(Lxst,label='Training')
plt.xlabel('iterations')
plt.legend()
prefix = 'reconstruction'
plt.title('Reconstruction Loss Lx')
plt.savefig('%s_loss.png' % (prefix))
plt.close()
'''
f=plt.figure()
plt.plot(Lzsv,label='Validation')
plt.plot(Lzst,label='Training')
plt.xlabel('iterations')
plt.title('Latent Loss Lz')
plt.legend()
prefix = 'latent'
plt.savefig('%s_loss.png' % (prefix))
plt.close()
'''