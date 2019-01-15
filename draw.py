#!/usr/bin/env python

""""
Simple implementation of http://arxiv.org/pdf/1502.04623v2.pdf in TensorFlow

Example Usage: 
	python draw.py --data_dir=/tmp/draw --read_attn=True --write_attn=True

Author: Eric Jang

#############################################################################

Changes by Valentin Yiu and Pranita Shrestha

Kept structure model with tensorflow (implementation was strictly following equations from the paper)

Changes:
- Fine tuning of the hyperparameters
- Adapting to GPU usage
- Training for Quick, Draw! Doodle dataset over minibatches of data
- Dividing training into epoch and minibatches
- Drawing doodles using inputs from validation set
- Outputing loss for both training and validation sets
- Fix loss function

"""

#Touch only hyperparameters and Run training
import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
import os
import time

tf.flags.DEFINE_string("data_dir", "", "")
tf.flags.DEFINE_boolean("read_attn", True, "enable attention for reader")
tf.flags.DEFINE_boolean("write_attn",True, "enable attention for writer")
FLAGS = tf.flags.FLAGS

## MODEL PARAMETERS ## 

A,B = 256,256 # image width,height
img_size = B*A # the canvas size

#Hyperparameters

#Keep fixed
enc_size = 256 # number of hidden units / output size in LSTM
dec_size = 256

read_n = 30 # read glimpse grid width/height, originally 5
write_n = 30 # write glimpse grid width/height
T = 20 #Generation sequence length, originally 10
z_size=20 # QSampler output size
batch_size=100 # training minibatch size, must be a square for output
learning_rate=1e-4 # learning rate for optimizern originally 1e-3
eps=1e-8 # epsilon for numerical stability
epoch = 30

#Write/read sizes
read_size = 2*read_n*read_n if FLAGS.read_attn else 2*img_size
write_size = write_n*write_n if FLAGS.write_attn else img_size

#Handy functions for working with the batches for quickdraw instead of the mnist integrated functions.
def next_batch(data,batch_size):
        #Creates a random batch of size batch_size from the array data
        indexes = np.random.choice(len(data),batch_size,replace = False)
        return [data[i] for i in indexes]

def shuffle_data(data):
        indexes = np.random.permutation(len(data))
        return [data[i] for i in indexes]
        
## BUILD MODEL ## 

DO_SHARE=None # workaround for variable_scope(reuse=True)

x = tf.placeholder(tf.float32,shape=(batch_size,img_size)) # input (batch_size * img_size)
e=tf.random_normal((batch_size,z_size), mean=0, stddev=1) # Qsampler noise
lstm_enc = tf.contrib.rnn.LSTMCell(enc_size, state_is_tuple=True) # encoder Op
lstm_dec = tf.contrib.rnn.LSTMCell(dec_size, state_is_tuple=True) # decoder Op

def linear(x,output_dim):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """
    w=tf.get_variable("w", [x.get_shape()[1], output_dim]) 
    b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x,w)+b

def filterbank(gx, gy, sigma2,delta, N):
    grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
    mu_x = gx + (grid_i - N / 2 - 0.5) * delta # eq 19
    mu_y = gy + (grid_i - N / 2 - 0.5) * delta # eq 20
    a = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, -1])
    b = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, -1])
    mu_x = tf.reshape(mu_x, [-1, N, 1])
    mu_y = tf.reshape(mu_y, [-1, N, 1])
    sigma2 = tf.reshape(sigma2, [-1, 1, 1])
    Fx = tf.exp(-tf.square(a - mu_x) / (2*sigma2))
    Fy = tf.exp(-tf.square(b - mu_y) / (2*sigma2)) # batch x N x B
    # normalize, sum over A and B dims
    Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keepdims=True),eps)
    Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keepdims=True),eps)
    return Fx,Fy

def attn_window(scope,h_dec,N):
    with tf.variable_scope(scope,reuse=DO_SHARE):
        params=linear(h_dec,5)
    # gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(1,5,params)
    gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(params,5,1)
    gx=(A+1)/2*(gx_+1)
    gy=(B+1)/2*(gy_+1)
    sigma2=tf.exp(log_sigma2)
    delta=(max(A,B)-1)/(N-1)*tf.exp(log_delta) # batch x N
    return filterbank(gx,gy,sigma2,delta,N)+(tf.exp(log_gamma),)

## READ ## 
def read_no_attn(x,x_hat,h_dec_prev):
    return tf.concat([x,x_hat], 1)

def read_attn(x,x_hat,h_dec_prev):
    Fx,Fy,gamma=attn_window("read",h_dec_prev,read_n)
    def filter_img(img,Fx,Fy,gamma,N):
        Fxt=tf.transpose(Fx,perm=[0,2,1])
        img=tf.reshape(img,[-1,B,A])
        glimpse=tf.matmul(Fy,tf.matmul(img,Fxt))
        glimpse=tf.reshape(glimpse,[-1,N*N])
        return glimpse*tf.reshape(gamma,[-1,1])
    x=filter_img(x,Fx,Fy,gamma,read_n) # batch x (read_n*read_n)
    x_hat=filter_img(x_hat,Fx,Fy,gamma,read_n)
    return tf.concat([x,x_hat], 1) # concat along feature axis

read = read_attn if FLAGS.read_attn else read_no_attn

## ENCODE ## 
def encode(state,input):
    """
    run LSTM
    state = previous encoder state
    input = cat(read,h_dec_prev)
    returns: (output, new_state)
    """
    with tf.variable_scope("encoder",reuse=DO_SHARE):
        return lstm_enc(input,state)

## Q-SAMPLER (VARIATIONAL AUTOENCODER) ##

def sampleQ(h_enc):
    """
    Samples Zt ~ normrnd(mu,sigma) via reparameterization trick for normal dist
    mu is (batch,z_size)
    """
    with tf.variable_scope("mu",reuse=DO_SHARE):
        mu=linear(h_enc,z_size)
    with tf.variable_scope("sigma",reuse=DO_SHARE):
        logsigma=linear(h_enc,z_size)
        sigma=tf.exp(logsigma)
    return (mu + sigma*e, mu, logsigma, sigma)

## DECODER ## 
def decode(state,input):
    with tf.variable_scope("decoder",reuse=DO_SHARE):
        return lstm_dec(input, state)

## WRITER ## 
def write_no_attn(h_dec):
    with tf.variable_scope("write",reuse=DO_SHARE):
        return linear(h_dec,img_size)

def write_attn(h_dec):
    with tf.variable_scope("writeW",reuse=DO_SHARE):
        w=linear(h_dec,write_size) # batch x (write_n*write_n)
    N=write_n
    w=tf.reshape(w,[batch_size,N,N])
    Fx,Fy,gamma=attn_window("write",h_dec,write_n)
    Fyt=tf.transpose(Fy,perm=[0,2,1])
    wr=tf.matmul(Fyt,tf.matmul(w,Fx))
    wr=tf.reshape(wr,[batch_size,B*A])
    #gamma=tf.tile(gamma,[1,B*A])
    return wr*tf.reshape(1.0/gamma,[-1,1])

write=write_attn if FLAGS.write_attn else write_no_attn

## STATE VARIABLES ## 

cs=[1]*T # sequence of canvases
mus,logsigmas,sigmas=[0]*T,[0]*T,[0]*T # gaussian params generated by SampleQ. We will need these for computing loss.
# initial states
h_dec_prev=tf.zeros((batch_size,dec_size))
enc_state=lstm_enc.zero_state(batch_size, tf.float32)
dec_state=lstm_dec.zero_state(batch_size, tf.float32)

## DRAW MODEL ## 

# construct the unrolled computational graph
with tf.device('/gpu:0'): #Delete this line to run with vanilla Tensorflow
        for t in range(T):
                c_prev = tf.zeros((batch_size,img_size)) if t==0 else cs[t-1]
                x_hat=x-tf.sigmoid(c_prev) # error image
                r=read(x,x_hat,h_dec_prev)
                h_enc,enc_state=encode(enc_state,tf.concat([r,h_dec_prev], 1))
                z,mus[t],logsigmas[t],sigmas[t]=sampleQ(h_enc)
                h_dec,dec_state=decode(dec_state,z)
                cs[t]=c_prev+write(h_dec) # store results
                h_dec_prev=h_dec
                DO_SHARE=True # from now on, share variables

## LOSS FUNCTION ## 

def binary_crossentropy(t,o):
    return -(t*tf.log(o+eps) + (1.0-t)*tf.log(1.0-o+eps))

# reconstruction term appears to have been collapsed down to a single scalar value (rather than one per item in minibatch)
x_recons=tf.nn.sigmoid(cs[-1])

# after computing binary cross entropy, sum across features then take the mean of those sums across minibatches
Lx=tf.reduce_sum(binary_crossentropy(x,x_recons),1) # reconstruction term
Lx=tf.reduce_mean(Lx)

kl_terms=[0]*T
for t in range(T):
    mu2=tf.square(mus[t])
    sigma2=tf.square(sigmas[t])
    logsigma=logsigmas[t]
    kl_terms[t]=0.5*tf.reduce_sum(mu2+sigma2-2*logsigma-1,1) # each kl term is for one recurrence batch
KL=tf.add_n(kl_terms) # Summing for 1:T
Lz=tf.reduce_mean(KL) # average over minibatches

cost=Lx+Lz

## OPTIMIZER ## 

optimizer=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
grads=optimizer.compute_gradients(cost)
for i,(g,v) in enumerate(grads):
    if g is not None:
        grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
train_op=optimizer.apply_gradients(grads)



## RUN TRAINING ## 
#Names placeholder for file saving
name_data_batches = 'data_batches2/batch_'
name_save = "model/drawmodelfinal_"
name_loss = "draw_loss_"
name_drawing = "draw_data_all_"

#Initial loading for shape
train_data = np.load(name_data_batches + '0.npy')


#Add data to fetch
fetches=[]
fetches.extend([Lx,Lz,train_op])


sess=tf.InteractiveSession()

saver = tf.train.Saver(max_to_keep=5000) # saves variables learned during training, keep number of 
tf.global_variables_initializer().run()
#last_save = None
#saver.restore(sess, last_save) # to restore from model, uncomment

#Timing
start_time = time.time()
train_iters = len(train_data)//batch_size

num_minibatches = 7
Lxs=[0]*train_iters*epoch*num_minibatches
Lzs=[0]*train_iters*epoch*num_minibatches
Lxs_mean = [0]*epoch*num_minibatches
Lzs_mean = [0]*epoch*num_minibatches

###Starting training
print('Training start')
for e in range(epoch):
        for k in range(num_minibatches):
                train_data = np.load(name_data_batches + str(k) + '.npy')
                xtrains = shuffle_data(train_data)
                for i in range(train_iters):
                        xtrain = xtrains[i*batch_size:(i+1)*batch_size]
                        feed_dict={x:xtrain}
                        results=sess.run(fetches,feed_dict)
                        Lxs[(e*num_minibatches+k)*train_iters + i],Lzs[(e*num_minibatches+k) * train_iters +i],_=results
                        iter_time = time.time()

                #Keep the mean of loss over each minibatch (= loss on minibatch)
                Lxs_mean[e*num_minibatches + k] = np.mean(Lxs[(e*num_minibatches+k)*train_iters:(e*num_minibatches+k+1)*train_iters])
                Lzs_mean[e*num_minibatches + k] = np.mean(Lzs[(e*num_minibatches+k)*train_iters:(e*num_minibatches+k+1)*train_iters])

                print("epoch=%d : Lx: %f Lz: %f time(s): %f" % (e,Lxs_mean[e*num_minibatches + k],Lzs_mean[e*num_minibatches + k], iter_time - start_time))
                ckpt_file= name_save + str(e) + "_" +  str(k) + ".ckpt"
                print("Model saved in file: %s" % saver.save(sess,ckpt_file))

out_file = os.path.join(FLAGS.data_dir,name_loss +"train" +".npz")
np.savez(out_file,Lxs_train = Lxs_mean,Lzs_train = Lzs_mean)

## TRAINING FINISHED ## 
print('Training finished')
#Validation

#Drawing with the fully trained model
num_classes = 10
validation_data = np.load(name_data_batches + str(num_minibatches) + '.npy')
xvalidations = validation_data[:batch_size]

#Take the first 10 of each class as validation data, draws all the classes
for i in range(num_classes):
        for j in range(10):
                xvalidations[10*i+j] = validation_data[500*i + j]


feed_dict = {x:xvalidations}
print('Starting drawing')
canvases,L_x,L_z=sess.run([cs,Lx,Lz],feed_dict)
canvases=np.array(canvases) # T x batch x img_size
out_file=os.path.join(FLAGS.data_dir,name_drawing + str(num_minibatches) + ".npz")

np.savez(out_file,canvases = canvases, loss = [[L_x],[L_z]])
print("Outputs saved in file: %s" % out_file)

#Computing the validation loss over the epoch by loading the attention data each time
#Taking the same validation data for each iteration shouldn't change the loss since the training set and validation set are separate
#Alternatively, load all the batch 7, then shuffle to get a new batch_size at each iteration

validation_data = np.load(name_data_batches + str(num_minibatches) + '.npy')[:batch_size]
feed_dict = {x:validation_data}
Lxs_validation=[0]*epoch*num_minibatches
Lzs_validation=[0]*epoch*num_minibatches

print('Starting computing validation loss')
for e in range(epoch):
        for k in range(num_minibatches):
                saver.restore(sess, name_save + str(e) + "_" + str(k) + ".ckpt")
                Lxs_validation[e*num_minibatches + k], Lzs_validation[e*num_minibatches + k] = sess.run([Lx,Lz],feed_dict)

out_file = os.path.join(FLAGS.data_dir,name_loss + "validation" + ".npz")
np.savez(out_file,Lxs_validation = Lxs_validation,Lzs_validation = Lzs_validation)
print('Done')
sess.close()