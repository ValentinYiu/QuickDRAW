# takes data saved by DRAW model and generates animations
# Example Usage: python plot_data.py testing_final draw_data.npz

import matplotlib
import sys
import numpy as np

interactive=False # set to False if you want to write images to file

if not interactive:
	matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot as plt

mask = np.vectorize(lambda x: x*0.25 if x<0.98 else 1.)
def xrecons_grid(X,B,A):
	"""
	plots canvas for single time step
	X is x_recons, (batch_size x img_size)
	assumes features = BxA images
	batch is assumed to be a square number
	"""
	padsize=1
	padval=.5
	ph=B+2*padsize
	pw=A+2*padsize
	batch_size=X.shape[0]
	N=int(np.sqrt(batch_size))
	X=X.reshape((N,N,B,A))
	img=np.ones((N*ph,N*pw))*padval
	for i in range(N):
		for j in range(N):
			startr=i*ph+padsize
			endr=startr+B
			startc=j*pw+padsize
			endc=startc+A
			img[startr:endr,startc:endc]=X[i,j,:,:]
			img[startr:endr,startc:endc] = mask(img[startr:endr,startc:endc])
			if i+j ==0:
				print(img[startr:endr,startc:endc])
	return img
	

if __name__ == '__main__':
	prefix=sys.argv[1]
	out_file=sys.argv[2]
	out = np.load(out_file)
	C = out['canvases']
	[Lxs,Lzs] = out['loss']
	print(Lxs)
	print(Lzs)
	T,batch_size,img_size=C.shape
	X=1.0/(1.0+np.exp(-C)) # x_recons=sigmoid(canvas)
	X = 1.0-X



	B=A=int(np.sqrt(img_size))
	if interactive:
		f,arr=plt.subplots(1,T)
	for t in range(T):
		img=xrecons_grid(X[t,:,:],B,A)
		if interactive:
			arr[t].matshow(img,cmap=plt.cm.gray)
			arr[t].set_xticks([])
			arr[t].set_yticks([])
		else:
			plt.matshow(img,cmap=plt.cm.gray)
			imgname='%s_%d.png' % (prefix,t) # you can merge using imagemagick, i.e. convert -delay 10 -loop 0 *.png mnist.gif
			plt.savefig(imgname)
			print(imgname)
	f=plt.figure()
	plt.plot(Lxs,label='Reconstruction Loss Lx')
	plt.plot(Lzs,label='Latent Loss Lz')
	plt.xlabel('iterations')
	plt.legend()
	if interactive:
		plt.show()
	else:
		plt.savefig('%s_loss.png' % (prefix))
