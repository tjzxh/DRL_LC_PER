import numpy as np
from keras import regularizers
from keras import initializers
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras import layers
#from keras.engine.training import collect_trainable_weights
from keras.layers import Dense, Flatten, Input, merge, Lambda
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size)
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size,action_dim):
        print("Now we build the model")
        S = Input(shape=[state_size])   
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        #Steering = Dense(1,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)  
        #Acceleration = Dense(1,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-4, name=name), kernel_regularizer=regularizers.l2(0.01))(h1)
        Acceleration = Dense(1, activation='tanh', use_bias=True, kernel_initializer=initializers.VarianceScaling(scale=1e-4, mode='fan_in', distribution='normal', seed=None), bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(h1)
        LaneChanging = Dense(1, activation='sigmoid', use_bias=True, kernel_initializer=initializers.VarianceScaling(scale=1e-4, mode='fan_in', distribution='normal', seed=None), bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(h1)
        #Brake = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1) 
        V = merge([LaneChanging,Acceleration],mode='concat')
        model = Model(input=S,output=V)
        return model, model.trainable_weights, S

