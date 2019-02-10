
import numpy as np
import random
import math
from operator import itemgetter
import matplotlib.pyplot as plt
from copy import deepcopy, copy
import Class_of_formula_printing as S

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
#from synthesis_NNC import get_netoperator
import numpy

import logging
#from train import train_and_score
#import train
early_stopper = EarlyStopping(patience=5)

import cubic_spline_planner
from sklearn.metrics import mean_squared_error
Name_of_file="Test_f.txt"
f=open(Name_of_file,"a")
show_animation = 0
show_result = 0
L=1.5
MAX_STEER = math.radians(45.0)
N_IND_SEARCH = 10
Kp = 1.0 
dt = 0.125  # [s]
dT = 10# [s]
L = 1.5  # [m]
# # Collection of unary and binary functions




# In[2]:

##binary##
def addition(vector):
    a = 0
    for i in vector:
        if a > 1e+6: return 1e+6
        elif a < -1e+6: return -1e+6
        elif a < 1e-6 and a > 0: return 1e-6
        elif a > -1e-6 and a < 0: return -1e-6
        else: a += i
    return a

def multiplication(vector):
    a = 1
    for i in vector:
        if a > 1e+6: return 1e+6
        elif a < -1e+6: return -1e+6
        elif a < 1e-6 and a > 0: return 1e-6
        elif a > -1e-6 and a < 0: return -1e-6
        else: a *= i
    return a

def maximum(vector):
    return max(vector)

def minimum(vector):
    return min(vector)

def hypot(x):
    s = 0
    for i in x:
        if i:
            i = float(np.nan_to_num(i))
            s += i ** 2
        else: continue
    s = float(np.nan_to_num(s))
    return np.sqrt(s)

def trapz(x):
    x = float(np.nan_to_num(np.trapz(x)))
    return x

def atan2(al):
    if(len(al)<2):
        #print("Amount of atan2 arguments is lower then 2:" + str(len(al)))
        #print(al)
        return(math.pi)
    return math.atan2(al[0],al[1])

def pi_2_pi(angle):
    if(isinstance(angle,float)):
        angle=[angle,]
    angle[0]%=2.0*math.pi
    while(angle[0] > math.pi):
        angle[0] = angle[0] - 2.0 * math.pi

    while(angle[0] < -math.pi):
        angle[0] = angle[0] + 2.0 * math.pi

    return angle[0]

##unary##
def relu(a):
    if a < 0: return 0
    else:
        if a > 1e+6: return 1e+6
        else: return a

def identity(a):
    return a

def pow_two(a):
    if a > 1e+3 or a < -1e+3: return 1e+6
    elif a < 1e-3 and a > 0: return 1e-6
    elif a > -1e-3 and a < 0: return 1e-6
    else: return a * a
    
def negative(a):
    return -a

def irer(a):
    a = float(np.nan_to_num(a))
    return np.sign(a) * np.sqrt(np.fabs(a))
    
def reverse(x):
    x = np.nan_to_num(x)
    if x > 1e+6 or x < -1e+6: return np.sign(x) * 1e+6
    elif x < 1e-6 and x > 0 or x > -1e-6 and x < 0: return np.sign(x) * 1e-6
    elif x == 0: return 1e+6
    else: return np.reciprocal(x)
    
def exp(a):
    if a > 14: return 1e+6
    elif a < -14: return 1e-6
    elif a < 1e-6 and a > 0: return 1
    elif a > -1e-6 and a < 0: return 1
    else: return math.exp(a)

def expm1(x):
    #x = float(np.nan_to_num(x))
    if x > 14: return 1e+6
    elif x < -14: return 1e-6
    elif x < 1e-6 and x > 0: return 0
    elif x > -1e-6 and x < 0: return 0
    else: return np.expm1(x)
    
def exp2(x):
    if x > 14: return 1e+6
    elif x < -14: return 1e-6
    elif x < 1e-6 and x > 0: return 1
    elif x > -1e-6 and x < 0: return 1
    else: return np.exp2(x)
    
def sign(x):
    x = float(np.nan_to_num(np.sign(x)))
    return x

def natlog(a):
    if a > 1e+6: return 1e+6
    elif a <= 0: return 0
    elif a < 1e-6 and a > 0: return -1e+6
    else: return math.log(a) 
def log10(x):
    #x = float(np.nan_to_num(x))
    if x > 1e+10: return 1e+6 
    elif x <= 0: return 0
    else: return np.log10(x)
def log2(x):
    #x = float(np.nan_to_num(x))
    if x > 1e+10: return 1e+6 
    elif x <= 0: return 0
    else: return np.log2(x)
def log1p(x):
    #x = float(np.nan_to_num(x))
    if x > 1e+10: return 1e+6 
    elif x <= -1: return 0
    else: return np.log1p(x)

def logic(a):
    if a >= 0: return 1
    else: return 0
    
def cosinus(a):
    if a > 1e+6: return math.cos(1e+6)
    elif a < 1e-6 and a > 0: return math.cos(1e-6)
    elif a < -1e+6: return math.cos(-1e+6)
    elif a > -1e-6 and a < 0: return math.cos(-1e-6)
    return math.cos(a)

def sinus(a):
    if a > 1e+6: return math.sin(1e+6)
    elif a < 1e-6 and a > 0: return math.sin(1e-6)
    elif a < -1e+6: return math.sin(-1e+6)
    elif a > -1e-6 and a < 0: return math.sin(-1e-6)
    return math.sin(a)

def tan(x):
    x = float(np.nan_to_num(x))
    if x < 1e-5 and x > 0: return 1e-5
    elif x > -1e-5 and x < 0: return -1e-5
    else: return np.tan(x)
    
def tanh(x):
    x = float(np.nan_to_num(x))
    if x > 10: return 1
    elif x < 1e-3 and x > 0: return 1e-6
    elif x < -10: return -1
    elif x > -1e-3 and x < 0: return -1e-6
    else: return np.tanh(x)

def cubicroot(a):
    if a > 1e+6 or a < -1e+6: return np.sign(a) * 1e+6
    elif a < 1e-6 and a > 0 or a > -1e-6 and a < 0: return np.sign(a) * 1e-6
    else: return np.cbrt(a)

def atan(x):
    if x > 1e+3: return 1e+3
    elif x < 1e-4 and x > 0: return 1e-4
    elif x < -1e+3: return -1e+3
    elif x > -1e-4 and x < 0: return -1e-4
    else: return np.arctan(x)

def cubic(a):
    if a > 1e+6: return 1e+6
    elif a < -1e+6: return -1e+6
    elif a > -1e-6 and a < 0: return -1e-6
    elif a < 1e-6 and a > 0: return 1e-6
    else: return a*a*a
    
def absolute(a):
    if a > 1e+6: return 1e+6
    elif a < -1e+6: return 1e+6
    elif a < 1e-6 and a > 0 or a > -1e-6 and a < 0: return 1e-6
    else: return np.fabs(a)
def sinc(x):
    return float(np.nan_to_num(np.sinc(x)))

def inv(x):
    if(x==0.0 or x==0):
        x=1e-40
    return (x**(-1))

def l(x):
    return L

def one(x):
    return 1.0

# # Parametric genetic algorithm

# In[3]:

class BaseGenetics(object):
    def __init__(self, e=None):
        self.estimator = None
        self.expectations = e # vector of math expectations for each component
        
    def set_estimator(self, f):
        def g(*args, **kwargs):
            return f(*args, **kwargs)
        self.estimator = g
        
    def generate_population(self, qmin, qmax, h, m):
        """
        Generate population.
        
        (real, real, int, int) -> [h x m] np.array of reals
        """
        population = {}
        e = self.expectations
        if e:
            functional = self.estimate_object_function(e)
            population[functional] = e
            while len(population) < h:
                candidate = np.random.normal(e, 0.1)
                functional = self.estimate_object_function(candidate)
                if functional < 1e+3: population[functional] = candidate
        else:
            while len(population) < h:
                candidate = np.random.uniform(qmin, qmax, m)
                functional = self.estimate_object_function(candidate)
                if functional < 1e+3: population[functional] = candidate
        return population
    
    def estimate_object_function(self, q):
        """
        Evaluates function self.estimator with q as an incoming parameter
        
        (vector) -> real
        """
        return self.estimator(q)

    
    def get_best_individual(self, population, worst=False, ksearch=None):
        """
        Return best or worst individual:
        1) if ksearch != None and worst==False: return best individual
        from ksearch random sample without replacement.
        2) if ksearch == None and worst==True: return index of the worst
        individual from the whole population.

        (2d array of real, bool, int) -> array of real OR int
        """
        population_estimates = np.array(list(population.keys()))
        if ksearch and not worst:
            try:
                subpopulation_estimates = population_estimates[np.random.choice(population_estimates.shape[0], ksearch, replace=False)]
                individual_estimate = subpopulation_estimates.min()
                return (population[individual_estimate], individual_estimate)
            except ValueError as e: print('Wrong type for ksearch: {0}'.format(e))
        else:
            best_estimate = population_estimates.min()
            return (population[best_estimate], best_estimate)
    
    def cross(self, population, ksearch):
        """
        Processes crossover of some individuals.

        (array of array of reals, int) -> (array of real, array of real) OR None
        """
        best_individual, best_value = self.get_best_individual(population)
        if len(best_individual) > 1:
            parent1, parent1_est = self.get_best_individual(population, worst=False, ksearch=ksearch)
            parent2, parent2_est = self.get_best_individual(population, worst=False, ksearch=ksearch)
            if np.max([best_value/parent1_est, best_value/parent2_est])>np.random.uniform():
                crossover_point = np.random.randint(1, len(parent1) - 1)
                child1 = np.hstack((parent1[:crossover_point], parent2[crossover_point:]))
                child2 = np.hstack((parent2[:crossover_point], parent1[crossover_point:]))
                return (child1, child2)
            else: return None
        elif len(best_individual) == 1: return (best_individual[:], best_individual[:])
        else: print('fuck you')
        
    def mutate(self, children, qmin, qmax, p=1):
        """
        Mutate given child1 and child2 with probability 'p'.

        (array of real, array of real, real, real, real) -> None
        """
        if np.random.rand() < p:
            mutated_children = {}
            for child in children:
                child_gene = np.random.randint(child.shape[0])
                child[child_gene] = np.random.uniform(qmin, qmax)
                child_functional = self.estimate_object_function(child)
                mutated_children[child_functional] = child
            return mutated_children
        else: return None
        
    def insert_children(self, population, children):
        """
        Replace the worst individuals with children, if they fit better.

        (2d array of real, array of real, array of real) -> None
        """
        merge = {**children, **population}
        k = len(children)
        estimates = list(merge.keys()) # unique estimates
        bad_k = np.partition(estimates, k)[-k:]
        for e in bad_k: del merge[e]
        return merge
                
    # psi_change_epoch <= individuals
    # ksearch <= individuals
    # variations_per_individuals >= 1
    # g > 0
    # crossings > 0
    
    def optimize(self, qmin=1, qmax=4, individuals=1000, generations=10,
                       individual_len=3, crossings=256, ksearch=16):
        print('Generating population for parametric optimization...')
        population = self.generate_population(qmin, qmax, individuals, individual_len)
        for g in range(generations):
            for c in range(crossings):
                children = self.cross(population, ksearch)
                if children:
                    children = self.mutate(children, qmin, qmax)
                    population = self.insert_children(population, children)
                    if len(population) <= ksearch:
                        print('Population died out!')
                        best_individual, best_value = self.get_best_individual(population)
                        print('J: {0}, Q: {1}'.format(best_value, best_individual))
                        return best_individual, best_value
                else: continue
            best_individual, best_value = self.get_best_individual(population)
            print('J: {0}, Q: {1}'.format(best_value, best_individual))
        return best_individual, best_value


# # Model

# In[4]:

class Robot():
    def __init__(self, x0, y0, tetta0, xf, yf, tettaf):
        self.x_history = [x0]
        self.y_history = [y0]
        self.tetta_history = [tetta0]
        self.control_history = [(0.0, 0.0)]
        self.xf = xf
        self.yf = yf
        self.tettaf = tettaf
        self.accuracy = None
        self.control_func = None
        self.float_limit = None
        self.dt = None
        self.sim_time = None
        self.time_history = None

    def set_simulation_config(self, simulation_time, integration_step=0.01, float_limit=1e+6, accuracy=0.01):
        self.sim_time = simulation_time
        self.dt = integration_step
        self.float_limit = float_limit
        self.time_history = np.arange(0, self.sim_time, self.dt)
        self.accuracy = accuracy
        
    def set_control_function(self, f):
        def g(*args, **kwargs):
            return f(*args, **kwargs)
        self.control_func = g
    
    def simulate(self,):   
        t = self.dt
        su = 0
        x1, y1 = 5, 5 # obsticle with shape of circle with middle point at 5, 5
        rad = 2.5
        while t < self.sim_time: # for t in self.time_history[1:]
            new_x, new_y, new_tetta = self.euler_step()
            if new_x == self.float_limit: return self.float_limit
            dr = math.pow(rad, 2) - math.pow(x1 - new_x, 2) - math.pow(y1 - new_y, 2)
            if dr > 0: su += 1
            estimation = self.estimate()
            if estimation > self.float_limit: return self.float_limit   
            self.update_history(new_x, new_y, new_tetta)
            if estimation < self.accuracy: return t + su * self.dt#return t # maybe return t + su * dt ???
            t += self.dt
        return self.sim_time + estimation + su * self.dt
    
    def euler_step(self,):
        x, y, tetta = self.get_current_coords()
        dx, dy, dtetta = self.__get_right_parts(tetta)
        if math.fabs(dx) > self.float_limit or math.fabs(dy) > self.float_limit or math.fabs(dtetta) > self.float_limit:
            return [self.float_limit]*3
        tilda_x = x + self.dt * dx
        tilda_y = y + self.dt * dy
        tilda_tetta = tetta + self.dt * dtetta
        
        tdx, tdy, tdtetta = self.__get_right_parts(tilda_tetta)
        x = x + (dx + tdx) * 0.5 * self.dt
        y = y + (dy + tdy) * 0.5 * self.dt
        tetta = tetta + (dtetta + tdtetta) * 0.5 * self.dt
        return x, y, tetta
    
    def __get_right_parts(self, tetta):
        current_coords = self.get_current_coords()
        terminal_coords = self.get_terminal_coords()
        state = terminal_coords - current_coords
        u1, u2 = self.control_func(state)
        self.clip_control(u1) # TODO: set control limits inside __init__
        self.clip_control(u2)
        self.update_control_history(u1, u2)
        right_x = (u1 + u2) * np.cos(tetta) * 0.5
        right_y = (u1 + u2) * np.sin(tetta) * 0.5
        right_tetta = (u1 - u2) * 0.5
        return right_x, right_y, right_tetta

    def update_history(self, x, y, tetta):
        self.x_history.append(x)
        self.y_history.append(y)
        self.tetta_history.append(tetta)

    def update_control_history(self, u1, u2):
        self.control_history.append((u1, u2))

    def clip_control(self, u):
        if u < -10: return -10
        elif u > 10: return 10
        else: return u
    
    def get_current_coords(self,):
        return np.array([self.x_history[-1], self.y_history[-1], self.tetta_history[-1]])
    
    def get_terminal_coords(self,):
        return np.array([self.xf, self.yf, self.tettaf])
    
    def estimate(self,):
        v0 = self.get_current_coords()
        vf = self.get_terminal_coords()
        return np.linalg.norm(vf - v0)
    
    def reset(self,):
        self.x_history = [self.x_history[0]]
        self.y_history = [self.y_history[0]]
        self.tetta_history = [self.tetta_history[0]]
        self.control_history = [(0.0, 0.0)]
    
    def get_x_history(self,):
        return self.x_history

    def get_y_history(self,):
        return self.y_history

    def get_tetta_history(self,):
        return self.tetta_history

    def get_control_history(self,):
        return self.control_history


# In[11]:

class Network():
    """Represent a network and let us operate on it.

    Currently only works for an MLP.
    """

    def __init__(self, nn_param_choices=None):
        """Initialize our network.
        
        Args:
            nn_param_choices (dict): Parameters for the network, includes:
                nb_neurons (list): [64, 128, 256]
                nb_layers (list): [1, 2, 3, 4]
                activation (list): ['relu', 'elu']
                optimizer (list): ['rmsprop', 'adam']
        """
        
        nn_param_choices = {
        'nb_neurons': [768],
        'nb_layers': [2],
        'activation': ['tanh'],
        'optimizer': ['adagrad'],}
        self.accuracy = 0.
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dic): represents MLP network parameters
        self.control_func = None
    def create_random(self):
        """Create a random network."""
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_set(self, network):
        """Set network properties.

        Args:
            network (dict): The network parameters

        """
        self.network = network
        
    def get_cifar10(self,):
        """Retrieve the CIFAR dataset and process the data."""
        # Set defaults.
        nb_classes = 1
        batch_size = 10
        input_shape = (5,)
    
        # Get the data.
        dataset = numpy.loadtxt("S_O.csv", delimiter=",")
        for i in range (0,len(dataset)):
        #get_netoperator(state.x, state.y,state.yaw,state.v, xrx[0][4],xry[1][4])
        #dxl = i[4] - i[0]
        #dyl = i[5] - i[1]
        #angle = pi_2_pi(state.yaw - cyaw[target_ind] )  # - math.atan2(dyl, dxl))
        #alpha = math.atan2(dyl, dxl) - i[2]
        #a=(get_netoperator([dataset[i][0],dataset[i][1],dataset[i][2],dataset[i][3],dataset[i][4],dataset[i][5]]))
            dataset[i][0:5] = self.control_func([dataset[i][0],dataset[i][1],dataset[i][2],dataset[i][3],dataset[i][4],dataset[i][5]])
        print(dataset[3,:])
        X = dataset[:,0:5]
        Y = dataset[:,6]
        x_train = X[:900,]
        y_train = Y[:900,]
        x_test = X[900:,]
        y_test = Y[900:,]
        
        """
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.reshape(50000, 3072)
        x_test = x_test.reshape(10000, 3072)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
    
        # convert class vectors to binary class matrices
        y_train = to_categorical(y_train, nb_classes)
        y_test = to_categorical(y_test, nb_classes)
        """
        return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)


    def compile_model(self,):
        """Compile a sequential model.
    
        Args:
            network (dict): the parameters of the network
    
        Returns:
            a compiled network.
    
        """
        # Get our network parameters.
        nb_layers = 2 #network['nb_layers']
        nb_neurons = 768  #network['nb_neurons']
        activation = 'tanh' #network['activation']
        optimizer = 'adagrad' #network['optimizer']
        nb_classes = 1
        model = Sequential()
    
        # Add each layer.
        for i in range(nb_layers):
    
            # Need input shape for first layer.
            if i == 0:
                model.add(Dense(nb_neurons, activation=activation,input_dim=5)) # input_shape=input_shape))
            else:
                model.add(Dense(nb_neurons, activation=activation))
    
            model.add(Dropout(0.2))  # hard-coded dropout
    
        # Output layer.
        model.add(Dense(nb_classes, activation= activation))
    
        model.compile(loss='mse', optimizer=optimizer,
                      metrics=['mae'])
    
        return model

    def train_and_score(self, network, dataset):
        """Train the model, return test loss.
    
        Args:
            network (dict): the parameters of the network
            dataset (str): Dataset to use for training/evaluating
    
        """
        if dataset == 'cifar10':
            nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test = self.get_cifar10()
        elif dataset == 'mnist':
            nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test = self.get_cifar10()
        
        model = self.compile_model()
    
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=10000,  # using early stopping, so no real limit
                  verbose=0,
                  validation_data=(x_test, y_test),
                  callbacks=[early_stopper])
    
        score = model.evaluate(x_test, y_test, verbose=0)
    
        return score[1]  # 1 is accuracy. 0 is loss.
    

    def train(self, dataset):
        """Train the network and record the accuracy.

        Args:
            dataset (str): Name of dataset to use.

        """
        if self.accuracy == 0.:
            self.accuracy = self.train_and_score(network, dataset)
        return self.accuracy
    
    def set_control_function(self, f):
        def g(*args, **kwargs):
            return f(*args, **kwargs)
        self.control_func = g

    def print_network(self):
        """Print out a network."""
        logging.info(self.network)
        logging.info("Network accuracy: %.2f%%" % (self.accuracy * 100))

# In[12]:

class State:

    def __init__(self, x=0., y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.w = 0.2
        self.r = 0.051
        self.pred=[0.0,0.0]

class Rover():
    def __init__(self, xf, yf, tettaf):
        #self.x_history = [x0]
        #self.y_history = [y0]
        #self.tetta_history = [tetta0]
        #self.control_history = [(0.0, 0.0)]
        self.xf = xf
        self.yf = yf
        self.tettaf = tettaf
        self.accuracy = None
        self.control_func = None
        self.net_func = None
        self.float_limit = None
        self.dt = None
        self.sim_time = None
        self.time_history = None

    def set_simulation_config(self, simulation_time, integration_step=0.01, float_limit=1e+6, accuracy=0.01):
        self.sim_time = simulation_time
        self.dt = integration_step
        self.float_limit = float_limit
        self.time_history = np.arange(0, self.sim_time, self.dt)
        self.accuracy = accuracy
        
    def set_control_function(self, f):
        def g(*args, **kwargs):
            return f(*args, **kwargs)
        self.control_func = g
    
    def set_control_function(self, f, nf):
        def g(*args, **kwargs):
            return f(*args, **kwargs)
        self.control_func = g
        self.net_func = nf
    
    def simulate(self,):   
        """t = self.dt
        su = 0
        x1, y1 = 5, 5 # obsticle with shape of circle with middle point at 5, 5
        rad = 2.5
        while t < self.sim_time: # for t in self.time_history[1:]
            new_x, new_y, new_tetta = self.euler_step()
            if new_x == self.float_limit: return self.float_limit
            dr = math.pow(rad, 2) - math.pow(x1 - new_x, 2) - math.pow(y1 - new_y, 2)
            if dr > 0: su += 1
            estimation = self.estimate()
            if estimation > self.float_limit: return self.float_limit   
            self.update_history(new_x, new_y, new_tetta)
            if estimation < self.accuracy: return t + su * self.dt#return t # maybe return t + su * dt ???
            t += self.dt
        return self.sim_time + estimation + su * self.dt"""
        
    
        a = 1  # a = 2, b = 2 krug; a = 1, b = 2 lissazu;  a = 2, b = 1 parabala;
        b = 2
        d = 0.5 * math.pi
       
        ax = [math.sin(a * t + d)*20 for t in np.arange(0., 0.98 * math.pi, 0.1)]    
        ay =  [math.sin(b * t)*20 for t in np.arange(0.,0.98 * math.pi, 0.1)] 
        
        #ax = [0.0, 0.4, 8.0, 12.0, 1.0]
        #ay = [0.0, 5.0, 0.0, 5.0, 0.0]
        goal = [ax[-1], ay[-1]]
    
        cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
            ax, ay, ds=0.1)
        target_speed = 6.6 / 3.6
        
        try:
            sp = self.calc_speed_profile(cx, cy, cyaw, target_speed)
            
            t, x, y, yaw, v, goal_flag, xr, yr, yawr, d = self.closed_loop_prediction(
            cx, cy, cyaw, ck, sp, goal)
        # Test
        ###################################################################
        #assert goal_flag, "Cannot goal"
        except AssertionError:
            #print("CANNOT GOAL")
            return 1.0e6 #CANNOT GOAL
        if show_result:
                plt.cla()
                plt.plot(cx, cy, "-r", label="course")
                plt.plot(x, y, "ob", label="trajectory")
                #plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
                plt.axis("equal")
                plt.grid(True)
                #plt.title("speed[km/h]:" + str(round(state.v * 3.6, 2)) +
                #          ",target index:" + str(target_ind))
                plt.pause(0.0001)
        if(not goal_flag):
            #print("CANNOT GOAL")
            return 1.5e6 #CANNOT GOAL
        print(mean_squared_error(x[1:], xr))
        print(mean_squared_error(y[1:], yr))
        print(mean_squared_error(yaw[1:], yawr))
        #print("Net operator:\n"+str(self.net_func.base_psi)+"\n"+str(self.net_func.psi)+"\n"+str(self.net_func.q))
        print("Net operator:\n"+str(self.net_func.psi)+"\n"+str(self.net_func.q))
        f.write("Goal\n")
        f.write(str(mean_squared_error(x[1:], xr))+"\n")
        f.write(str(mean_squared_error(y[1:], yr))+"\n")
        f.write(str(mean_squared_error(yaw[1:], yawr))+"\n")
        f.write("Net operator:\n"+str(self.net_func.psi)+"\n")
        
        input_nodes = [0, 1, 2, 3, 4]
        output_nodes = [18]
        #psi = [[(0, 20), (2, 17), (5, 0)], [(5, 0), (2, 14), (3, 12), (18, 1)]]
        #names_of_pars = ["q0","q1","q2","x0","x1"]
        names_of_inputs = ['x0','x1']
        names_of_params = ['q0','q1','q2']
        sym = symbols(names_of_inputs)
        #self.sym.append(symbols('x0'))
        #self.sym.append(symbols('x1'))
        q = self.net_func.q#{0: 0.04710989408847488, 1: 1.0, 2: 0.5}
        nop = [q[0],q[1],q[2]]
        
        s=S.Simplify(input_nodes,output_nodes,self.net_func.psi,sym,nop)
        s.simpli()
        
        return (mean_squared_error(x[1:], xr) + mean_squared_error(y[1:], yr) + mean_squared_error(yaw[1:], yawr))/3.0
    
    
    def closed_loop_prediction(self, cx, cy, cyaw, ck, speed_profile, goal):
        
        T = 500.0  # max simulation time
        goal_dis = 0.3
        stop_speed = 0.05
        
        state = State(x=cx[0], y=cy[0], yaw=1.5, v=0.0)
    
        time = 0.0
        x = [state.x]
        y = [state.y]
        yaw = [state.yaw]
        v = [state.v]
        t = [0.0]
        xr=[]
        yr=[]
        d = []
        yawr=[]
        goal_flag = False
        #target_ind = calc_nearest_index(state, cx, cy, cyaw, 0)
        
        target_ind =0
        cyaw = self.smooth_yaw(cyaw)
        while T >= time:
            #target_ind =calc_ref_trajectory(state, cx, cy, cyaw, target_ind)
            #start_time = timeit.default_timer()
            di, target_ind = self.rear_wheel_feedback_control(
                state, cx, cy, cyaw, ck, target_ind)
            #print(timeit.default_timer() - start_time)
            ai = self.PIDControl(speed_profile[target_ind], state.v)
            state = self.update(state, ai, di)
            #CNN
            
            #dl=1750-242*di
            #dr=1750+242*di
            #di,dl,dr=0.0,1746.6666666666667,1752.6666666666667
            #di,dl,dr=0.0,1500.0,1500.0
            #dl=random.gauss(dl, 50)
            #dr=random.gauss(dr, 50) 
            #di=1.0
            #Xxx= [state.x, state.y,state.yaw,state.v, xref[0][4],xref[1][4],xref[3][4]] 
            #Xxx= [di, dl,dr] 
            #xrec=xre.tolist()
            #Xxx=np.array([Xxx])
            #print(Xxx)
            #start_time = timeit.default_timer()
            #state.pred = loaded_model.predict(Xxx)               ######################
            #print(timeit.default_timer() - start_time)
            #print(state.pred)
            #print(di)
            #state = update(state, state.pred[0][0]*(1.0), state.pred[0][1]*(1.0), di)
    
            if abs(state.v) <= stop_speed:
                target_ind += 1
    
            time = time + dt
    
            # check goal
            dx = state.x - goal[0]
            dy = state.y - goal[1]
            if math.sqrt(dx ** 2 + dy ** 2) <= goal_dis:
                print("Goal")
                goal_flag = True
                break
            #print(target_ind)
            x.append(state.x)
            y.append(state.y)
            yaw.append(state.yaw)
            v.append(state.v)
            t.append(time)
            d.append(di)
            xr.append(cx[min(target_ind,len(cx)-1)])
            yr.append(cy[min(target_ind,len(cy)-1)])
            yawr.append(cyaw[min(target_ind,len(cyaw)-1)])
            if target_ind % 1 == 0 and show_animation:
                plt.cla()
                plt.plot(cx, cy, "-r", label="course")
                plt.plot(x, y, "ob", label="trajectory")
                plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
                plt.axis("equal")
                plt.grid(True)
                plt.title("speed[km/h]:" + str(round(state.v * 3.6, 2)) +
                          ",target index:" + str(target_ind))
                plt.pause(0.0001)
    
        return t, x, y, yaw, v, goal_flag, xr, yr, yawr, d
    
    def calc_speed_profile(self, cx, cy, cyaw, target_speed):
    
        speed_profile = [target_speed] * len(cx)
    
        direction = 1.0
    
        # Set stop point
        for i in range(len(cx) - 1):
            dyaw = cyaw[i + 1] - cyaw[i]
            switch = math.pi / 4.0 <= dyaw < math.pi / 2.0
    
            if switch:
                direction *= -1
    
            if direction != 1.0:
                speed_profile[i] = - target_speed
            else:
                speed_profile[i] = target_speed
    
            if switch:
                speed_profile[i] = 0.0
    
        speed_profile[-1] = 0.0
    
        return speed_profile
    
    def smooth_yaw(self, yaw):
    
        for i in range(len(yaw) - 1):
            dyaw = yaw[i + 1] - yaw[i]
    
            while dyaw >= math.pi / 2.0:
                yaw[i + 1] -= math.pi * 2.0
                dyaw = yaw[i + 1] - yaw[i]
    
            while dyaw <= -math.pi / 2.0:
                yaw[i + 1] += math.pi * 2.0
                dyaw = yaw[i + 1] - yaw[i]
    
        return yaw
    
    def rear_wheel_feedback_control(self, state, cx, cy, cyaw, ck, preind):
        ind, e = self.calc_nearest_index(state, cx, cy, cyaw, preind)
    
        k = ck[ind]
        v = state.v
        
        
        th_e = pi_2_pi(state.yaw - cyaw[ind])
        """
        omega = v * k * math.cos(th_e) / (1.0 - k * e) - \
            KTH * abs(v) * th_e - KE * v * math.sin(th_e) * e / th_e
    
        if th_e == 0.0 or omega == 0.0:
            return 0.0, ind
    
        delta = math.atan2(L * omega / v, 1.0)
        """
        delta=self.control_func([th_e,e])
        #  print(k, v, e, th_e, omega, delta)
        #print(th_e,e,delta)
        delta=delta[0]
    
        return delta, ind
    
    def update(self, state, a, delta):
        if delta >= MAX_STEER:
            delta = MAX_STEER
        elif delta <= -MAX_STEER:
            delta = -MAX_STEER
        #state.vl=delta*0.18*dT
        #state.vr=delta*0.18*dT*(-1)
        #state.r state.w 
        #vv=(state.v+state.vl+state.v+state.vr)/2;
        state.x = state.x + state.v * math.cos(state.yaw) * dt
        #state.x = random.gauss(state.x, 0.5) 
        state.y = state.y + state.v * math.sin(state.yaw) * dt
        #state.y = random.gauss(state.y, 0.5) 
        
        state.yaw = state.yaw + state.v / L * math.tan(delta) * dt
        #state.yaw = state.yaw + state.v / 0.2 * math.tan(delta) * dt
        state.v = state.v + a * dt
        #print((state.v+state.vr-state.v+state.vl))
        #print(state.vl)
        #print(state.vr)
        return state
    
    
    def PIDControl(self, target, current):
        a = Kp * (target - current)
    
        return a
    
    def calc_nearest_index(self, state, cx, cy, cyaw, pind):
    
        dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
        dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]
        
        #dx = [state.x - icx for icx in cx]
        #dy = [state.y - icy for icy in cy]
    
        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
        try:
            mind = min(d)
        except:
            #print(d)
            if show_result:
                plt.cla()
                plt.plot(cx, cy, "-r", label="course")
                plt.plot(state.x, state.y, "ob", label="trajectory")
                #plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
                plt.axis("equal")
                plt.grid(True)
                #plt.title("speed[km/h]:" + str(round(state.v * 3.6, 2)) +
                #          ",target index:" + str(target_ind))
                plt.pause(0.0001)
            raise AssertionError ("000")
            #mind = 0
            #return 0
        
        ind = d.index(mind) + pind
    
        mind = math.sqrt(mind)
    
        dxl = cx[ind] - state.x
        dyl = cy[ind] - state.y
    
        angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
        if angle < 0:
            mind *= -1
            
        if pind >= ind:
            ind = pind
    
        return ind, mind
    
    def euler_step(self,):
        x, y, tetta = self.get_current_coords()
        dx, dy, dtetta = self.__get_right_parts(tetta)
        if math.fabs(dx) > self.float_limit or math.fabs(dy) > self.float_limit or math.fabs(dtetta) > self.float_limit:
            return [self.float_limit]*3
        tilda_x = x + self.dt * dx
        tilda_y = y + self.dt * dy
        tilda_tetta = tetta + self.dt * dtetta
        
        tdx, tdy, tdtetta = self.__get_right_parts(tilda_tetta)
        x = x + (dx + tdx) * 0.5 * self.dt
        y = y + (dy + tdy) * 0.5 * self.dt
        tetta = tetta + (dtetta + tdtetta) * 0.5 * self.dt
        return x, y, tetta
    
    def __get_right_parts(self, tetta):
        current_coords = self.get_current_coords()
        terminal_coords = self.get_terminal_coords()
        state = terminal_coords - current_coords
        u1, u2 = self.control_func(state)
        self.clip_control(u1) # TODO: set control limits inside __init__
        self.clip_control(u2)
        self.update_control_history(u1, u2)
        right_x = (u1 + u2) * np.cos(tetta) * 0.5
        right_y = (u1 + u2) * np.sin(tetta) * 0.5
        right_tetta = (u1 - u2) * 0.5
        return right_x, right_y, right_tetta

    def update_history(self, x, y, tetta):
        self.x_history.append(x)
        self.y_history.append(y)
        self.tetta_history.append(tetta)

    def update_control_history(self, u1, u2):
        self.control_history.append((u1, u2))

    def clip_control(self, u):
        if u < -10: return -10
        elif u > 10: return 10
        else: return u
    
    def get_current_coords(self,):
        return np.array([self.x_history[-1], self.y_history[-1], self.tetta_history[-1]])
    
    def get_terminal_coords(self,):
        return np.array([self.xf, self.yf, self.tettaf])
    
    def estimate(self,):
        v0 = self.get_current_coords()
        vf = self.get_terminal_coords()
        return np.linalg.norm(vf - v0)
    
    def reset(self,):
        self.x_history = [self.x_history[0]]
        self.y_history = [self.y_history[0]]
        self.tetta_history = [self.tetta_history[0]]
        self.control_history = [(0.0, 0.0)]
    
    def get_x_history(self,):
        return self.x_history

    def get_y_history(self,):
        return self.y_history

    def get_tetta_history(self,):
        return self.tetta_history

    def get_control_history(self,):
        return self.control_history
        




# In[5]:
        
# # Network Operator

class NetworkOperator(object):
    """
    
    """
    def __init__(self, unaries, binaries, input_nodes, output_nodes):
        """
        Instantiates Network Operator object
        xq - list of variables and parameters
        unaries - list of unary functions
        binaries - list of binary functions
        """
        self.psi = None
        self.base_psi = None
        self.un_dict = {ind: func for ind, func in enumerate(unaries)}
        self.bin_dict = {ind: func for ind, func in enumerate(binaries)}
        self.q = []
        self.base_q = []
        self.__input_node_free_index = None
        self.output_nodes = output_nodes # list of indexes that output nodes posess
        self.input_nodes = input_nodes # list of indexes that input nodes posess
  
    def get_input_nodes(self,):
        return self.input_nodes
    
    def set_q(self, q):
        """
        q - list
        return dict
        """
        self.q = {ind: val for ind, val in enumerate(q)}
        self.__input_node_free_index = len(q)

    def get_q(self,):
        return self.q
    
    def set_base_q(self, q):
        self.base_q = {ind: val for ind, val in enumerate(q)}
        self.__input_node_free_index = len(q)
        
    def get_base_q(self,):
        return self.base_q
    
    def update_base_q(self,):
        new_q = copy(self.get_q()).values()
        self.set_base_q(new_q)
        
    def roll_back_to_base_q(self,):
        old_q = copy(self.get_base_q()).values()
        self.set_q(old_q)
        
    def get_free_input_node(self,):
        return self.__input_node_free_index
    
    def variate_parameters(self, index, value):
        q = self.get_q()
        q[index] = value
        
    def get_psi(self,):
        return self.psi

    def set_psi(self, psi):
        self.psi = psi 
    
    def get_base_psi(self,):
        return self.base_psi
        
    def set_base_psi(self, base_psi):
        self.base_psi = base_psi
        
    def update_base_psi(self,):
        new_psi = deepcopy(self.get_psi())
        self.set_base_psi(new_psi)
        
    def roll_back_to_base_psi(self,):
        old_psi = deepcopy(self.get_base_psi())
        self.set_psi(old_psi)
    
    def get_unary_dict_keys(self,):
        return list(self.un_dict.keys())
    
    def get_binary_dict_keys(self,):
        return list(self.bin_dict.keys())
    
    def eval_psi(self, x):
        """
        out_nodes - indexes of nodes which are outputs of nop. [list]
        x - list of state components
        """
        x = {self.get_free_input_node() + ind: val for ind, val in enumerate(x)}
        xq = {**self.q, **x} # merge two dicts without altering the originals
        d = xq.copy()#{} # node: value
        psi = self.get_psi()
        def apply_unary(unary_func, unary_index):
            if unary_index < len(xq): return unary_func(xq[unary_index]) # try to apply unary to q or x
            else:
                #print(unary_index,d)
                return unary_func(d[unary_index]) # apply unary to a dictinoary with that node otherwise
        for cell in psi:
            binary_index = cell[-1][1] # binary operation index
            binary_func = self.bin_dict[binary_index] # binary function object
            d[cell[-1][0]] = binary_func([apply_unary(self.un_dict[i[1]], i[0]) for i in cell[:-1]])
        #print(len(d),self.output_nodes)
        nop_outputs = [d[node] for node in self.output_nodes]
        #print(nop_outputs)
        return nop_outputs


# # Structural genetic algorithm

# In[6]:

class StructureGenetics(object):
    """
    
    """
    def __init__(self, nop, model):
        self.nop = nop
        self.model = model
        #self.model.train_and_score(model, 'cifar10')
        self.model.set_control_function(nop.eval_psi, nop)
        self.base_estimation = None
        self.qmin = None
        self.qmax = None
    
    def generate_variations_population(self, individuals, variations_per_individual):
        """
        Generate population.
        
        (int, int) -> np.array of integers (shape: [h x m x 4])
        """
        population = {}
        while len(population) < individuals:
            individual = [self.generate_variation(mutate_param=False) for i in range(variations_per_individual)]
            functional = self.estimate_variation(individual)
            if functional < 1e+3: population[functional] = individual
        return population

    def estimate_object_function(self,):
        """
        Evaluates function self.estimator with q as an incoming parameter
        
        (vector) -> real
        """
        functional = self.model.simulate()
        #functional = train.train_and_score(network, 'cifar10')
        
        #functional = self.model.train_and_score(self.model, 'cifar10')
        #functional = network.train_and_score(network, 'cifar10')
        #self.model.reset()
        return functional

    def estimate_variation(self, variation_matrix):
        for variation in variation_matrix:
            if variation[0] != 3: # first we apply all kinds of variations except of the delete
                self.apply_variation(variation)
        for variation in variation_matrix:
            if variation[0] == 3: # then we apply delete variation
                self.apply_variation(variation)
        a = self.nop.get_psi()
        b = self.nop.get_base_psi()
        c = self.nop.get_q()
        d = self.nop.get_base_q()
        if a == b and c == d: est = self.base_estimation
        else: est = self.estimate_object_function()            
        self.nop.roll_back_to_base_psi()
        self.nop.roll_back_to_base_q()
        return est

    ### function to insert into parametric optimization ###
    def estimate_parameters(self, q):
        self.nop.set_q(q)
        functional = self.estimate_object_function()
        return functional
    #######################################################

    def get_best_individual(self, population, ksearch=None):#, worst=False, ksearch=None):
        """
        Return best or worst individual:
        1) if ksearch != None and worst==False: return best individual
        from ksearch random sample without replacement.
        2) if ksearch == None and worst==True: return index of the worst
        individual from the whole population.

        (2d array of real, bool, int) -> array of real OR int
        """
        population_estimates = np.array(list(population.keys()))
        if ksearch:# and not worst:
            try:
                subpopulation_estimates = population_estimates[np.random.choice(population_estimates.shape[0], ksearch, replace=False)]
                individual_estimate = subpopulation_estimates.min()
                return (population[individual_estimate], individual_estimate)
            except ValueError as e: print('Wrong type for ksearch: {0}'.format(e))
        else:
            best_estimate = population_estimates.min()
            return (population[best_estimate], best_estimate)
       
    def generate_variation(self, mutate_param=False):
        psi = self.nop.get_base_psi()
        var_num = random.randint(0,4)
        #var_num = np.random.choice(5, p=[0.1, 0.1, 0.1, 0.1, 0.6])
        sublist_index = random.randint(0, len(psi) - 1) # operand column index 
        un_keys_list = self.nop.get_unary_dict_keys()
        bin_keys_list = self.nop.get_binary_dict_keys()
        if var_num == 4 or mutate_param: # nop must have at least one parameter
            #param_index = random.randrange(0, self.nop.get_free_input_node())
            param_index = random.randrange(0, 1)
            new_value = random.uniform(self.qmin, self.qmax)
            if not mutate_param: return [4, param_index, new_value, None]
            else: return [4, mutate_param, new_value, None]
        elif var_num == 0: # change binary operation
            bin_keys_list = self.nop.get_binary_dict_keys()
            new_bin_op = random.choice(bin_keys_list)
            c = random.randint(0, max(un_keys_list[-1], bin_keys_list[-1]))
            return [0, sublist_index, c, new_bin_op]
        elif var_num == 1: # change unary operation
            un_keys_list = self.nop.get_unary_dict_keys()
            l = len(psi[sublist_index])
            unary_cell = random.randint(0, l - 2) # except binary node
            new_un_op = random.choice(un_keys_list)
            return [1, sublist_index, unary_cell, new_un_op]
        elif var_num == 2: # add unary operation
            new_un_op = random.choice(un_keys_list)
            if sublist_index == 0:
                node = random.choice(self.nop.get_input_nodes())
            else:
                node = random.randint(0, psi[sublist_index-1][-1][0])
            return [2, sublist_index, node, new_un_op]
        elif var_num == 3: # delete unary operation
            a = random.randrange(0, len(psi))
            b = random.randrange(0, len(psi[a]))
            c = random.randint(0, max(un_keys_list[-1], bin_keys_list[-1]))
            index_to_start_from_delete = None
            exclude = []
            inputs = self.nop.get_input_nodes()
            for i in inputs:
                for ind, val in enumerate(psi):
                    for j, v in enumerate(val):
                        if v[0] == i:
                            exclude.append((ind, j))
                            break
                    break
                continue
            left_bound = max(exclude, key=itemgetter(0)) # (sublist_index, tuple_index)
            sublist_index = random.randint(left_bound[0], len(psi)-1)
            l = len(psi[sublist_index])
            if l > 3: # if that column has more than one operand
                if sublist_index == left_bound[0]:
                    sample_indices = [j for j, v in enumerate(psi[sublist_index][:-1]) if j != left_bound[1]]
                    if sample_indices:
                        cell_to_del = random.choice(sample_indices)
                    else:
                        return [3, a, b, c]
                else: cell_to_del = random.randint(0, l - 2) # choose random index of the cell, except the last(binary cell)
                node_to_del = psi[sublist_index][cell_to_del][0] # operand row index
                nodes = [list(map(itemgetter(0), sublist[:-1])) for sublist in psi] # all unary nodes (list of lists)
                if sum(x.count(node_to_del) for x in nodes) > 1: return [3, sublist_index, cell_to_del, c] # if more than one occurence 
                else: return [3, a, b, c] # lost graph connectivity
            else: return [3, a, b, c] # lost graph connectivity       

    def apply_variation(self, variation):
        loc_psi = self.nop.get_psi()
        sublist_index = variation[1]
        if variation[0] == 0: # change binary
            new_bin_op = variation[3]
            if new_bin_op > len(self.nop.get_binary_dict_keys()) - 1: return None
            node = loc_psi[sublist_index][-1][0]
            loc_psi[sublist_index][-1] = (node, new_bin_op)
        elif variation[0] == 1: # change unary
            cell = variation[2]
            new_un_op = variation[3]
            if cell >= len(loc_psi[sublist_index]) - 1: return None
            elif new_un_op > len(self.nop.get_unary_dict_keys()) - 1: return None
            node = loc_psi[sublist_index][cell][0]
            loc_psi[sublist_index][cell] = (node, new_un_op)
        elif variation[0] == 2: # add unary
            node = variation[2]
            new_un_op = variation[3]
            if new_un_op > len(self.nop.get_unary_dict_keys()) - 1: return None
            new_cell = (node, new_un_op)
            _ = loc_psi[sublist_index].pop()
            loc_psi[sublist_index].append(new_cell)
            loc_psi[sublist_index].append(_)
        elif variation[0] == 3: # delete unary
            node_to_del = variation[2]
            if len(loc_psi[sublist_index]) < 3: return None
            elif node_to_del >= len(loc_psi[sublist_index]) - 1: return None
            else:
                for ind, sublist in enumerate(loc_psi[:sublist_index]):
                    if sublist[-1][0] == node_to_del:
                        nodes = [list(map(itemgetter(0), sublist[:-1])) for sublist in loc_psi[ind + 1:]]
                        break
                else:
                    nodes = [list(map(itemgetter(0), sublist[:-1])) for sublist in loc_psi]
                if sum(x.count(node_to_del) for x in nodes) > 1:
                    del loc_psi[sublist_index][node_to_del]
                else:
                    return None
        elif variation[0] == 4: # change parameter
            param_index = variation[1]
            new_value = variation[2]
            self.nop.variate_parameters(param_index, new_value)
                
    def cross(self, population, ksearch, var_num, children_num=8):
        best_individual, best_value = self.get_best_individual(population)
        parent1, parent1_est = self.get_best_individual(population, ksearch=ksearch)
        parent2, parent2_est = self.get_best_individual(population, ksearch=ksearch)
        if np.max([best_value/parent1_est, best_value/parent2_est]) > np.random.uniform():
            param_len = len(self.nop.get_q())
            all_variations = np.vstack((parent1, parent2))
            new_vars_len = round(0.38 * all_variations.shape[0])
            _ = [self.generate_variation(mutate_param=i%param_len) for i in range(new_vars_len)]
            _ = np.reshape(_, (-1, 4))
            all_variations = np.vstack((all_variations, _))
            sex = lambda: all_variations[np.random.choice(all_variations.shape[0], var_num, replace=False), :]
            ch = [sex() for i in range(children_num)]
            children = {}
            for child in ch:
                functional = self.estimate_variation(child)
                children[functional] = child
            return children
        else: return None
            
    def insert_children(self, population, children):
        """
        Replace the worst individuals with children, if they fit better.

        (2d array of real, array of real, array of real) -> None
        """
        merge = {**children, **population}
        k = len(children)
        estimates = list(merge.keys()) # unique estimates
        bad_k = np.partition(estimates, k)[-k:]
        for e in bad_k: del merge[e]
        return merge
    
    def emigrate(self, pop_len_before, variations_per_individual, population):
        pop_len_after = len(population)
        emigration_len = pop_len_after - pop_len_before
        emigration = {}
        while len(population) < pop_len_before:
            individual = [self.generate_variation(mutate_param=False) for i in range(variations_per_individual)]
            functional = self.estimate_variation(individual)
            if functional < 1e+3: population[functional] = individual
        return population
    
    def change_base_psi(self, individual):
        """ Apply variations in individual to the structure
        :param individual: list of lists of objects
        :return: None
        """
        for variation in individual:
            self.apply_variation(variation)
        self.nop.update_base_psi()
        self.nop.update_base_q()

    def optimize(self, qmin=-10, qmax=10, individuals=100, generations=256, psi_change_epoch=2,
                       variations_per_individual=4, crossings=256, ksearch=16):
        self.qmin = qmin
        self.qmax = qmax
        parameters_len = len(self.nop.get_q())
        print('Initializing hyper-parameters...')
        print('''qmin: {0}, qmax: {1}\nPopulation size: {2}\nMax variations number for each individual: {3}\nGenerations: {4}\nCrossings per epoch: {5}\nksearch: {6}\nGeneration to change psi: each {7}th'''.format(
                 qmin, qmax, individuals, variations_per_individual, generations, crossings, ksearch, psi_change_epoch))
        print('Beginning structure synthesis with parameters: {0}'.format(list(self.nop.get_q().values())))
        print('Generating population for structure synthesis...')
        pop_gen_func = lambda: self.generate_variations_population(individuals, variations_per_individual)
        population = pop_gen_func()
        self.base_estimation = self.estimate_object_function() # estimation of base psi
        print('J: {0}'.format(self.base_estimation))
        for g in range(generations):
            f.write('Generation {0} is running...'.format(g))
            print('Generation {0} is running...'.format(g))
            if g % psi_change_epoch == 0 and g != 0 and generations > psi_change_epoch:
                candidate = self.estimate_variation(best_individual)
                if candidate < self.base_estimation:
                    self.change_base_psi(best_individual)
                    self.base_estimation = candidate
                else:
                    print('Proceeding with no changes applied to structure and parameters')
                    continue
                f.write('Refreshing parameters: \n{0}'.format(list(self.nop.get_q().values())))
                f.write('Refreshing base structure: \n{0}'.format(self.nop.get_base_psi()))
                f.write('J: {0}'.format(best_value))
                f.write("#####################################################")
                print('Refreshing parameters: \n{0}'.format(list(self.nop.get_q().values())))
                print('Refreshing base structure: \n{0}'.format(self.nop.get_base_psi()))
                print('J: {0}'.format(best_value))
                print("#####################################################")
                
            for c in range(crossings):
                children = self.cross(population, ksearch, variations_per_individual, children_num=4)
                if children is not None:
                    population = self.insert_children(population, children)
                    population = self.emigrate(individuals, variations_per_individual, population)
                else: continue
            best_individual, best_value = self.get_best_individual(population)
        candidate = self.estimate_variation(best_individual)
        if candidate < self.base_estimation:
            self.change_base_psi(best_individual) # final psi/q change
            self.base_estimation = candidate
        print('Optimizing parameters with fixed structure:\n{0}'.format(self.nop.get_base_psi()))
        gaussian_means = list(self.nop.get_base_q().values())
        pg = BaseGenetics(e=gaussian_means)
        pg.set_estimator(self.estimate_parameters)
        best_q, best_estimation = pg.optimize(qmin, qmax,
                                              individuals,
                                              generations,
                                              parameters_len,
                                              crossings,
                                              ksearch)
        self.nop.roll_back_to_base_psi()
        self.nop.set_q(best_q)
        self.nop.update_base_q()
        print('Done.')
        print('J: {0}\nQ: {1}\nStructure:\n{2}'.format(best_estimation, best_q, self.nop.get_base_psi()))
        return best_estimation, best_q, self.nop.get_base_psi()
        
# In[11]:

def get_netoperator(inp):
    unaries = [identity, negative, pow_two, sinus, logic, cosinus, atan, exp, natlog, irer, cubic, reverse, cubicroot,
               expm1, exp2, sign, log10, log2, log1p, absolute, tan, tanh, relu]#, sinc]
    binaries = [addition, multiplication, maximum, minimum, atan2, pi_2_pi]#, hypot, trapz]
    input_nodes = [0, 1, 2, 3, 4, 5]
    output_nodes = [6, 7, 10, 9, 3]
    nop = NetworkOperator(unaries, binaries, input_nodes, output_nodes)
    q = []
    nop.set_q(q) # set some parameters
    nop.update_base_q() # set those parameters as base as well
    #psi = [[(0, 9), (1, 7), (3, 14), (6, 3)], [(1, 1), (4, 0), (2, 6), (7, 1)], [(2, 0), (5, 9), (3, 2), (8, 1)], [(6, 0), (8, 0), (7, 12), (5, 5), (7, 7), (9, 0)], [(6, 0), (7, 0), (5, 9), (10, 0)]]
    #psi = [[(0, 2), (3, 0), (6, 3)], [(1, 0), (4, 0), (7, 1)], [(2, 0), (5, 0), (3, 1), (8, 1)], [(6, 0), (7, 4), (8, 0), (9, 3)], [(6, 0), (7, 0), (10, 0)]]
    #psi = [[(0, 2), (3, 0), (0, 15), (0, 15), (0, 15), (0, 15), (6, 3)], [(1, 0), (4, 0), (7, 1)], [(2, 0), (5, 0), (3, 1), (8, 1)], [(6, 0), (7, 4), (8, 0), (9, 3)], [(6, 0), (7, 0), (9, 19), (9, 19), (10, 3)]]
    psi = [[(4,0),(0,1),(6,0)],[(5,0),(1,1),(7,0)],[(7,0),(6,0),(8,4)],[(8,0),(2,1),(9,0)],[(9,0),(10,5)]]
    nop.set_psi(psi) # set some structure
    nop.update_base_psi() # set this structure as base as well
    # 3) Instantiate structure genetic and pass model with nop to it
    
    return nop.control_func(inp) #nop.eval_psi(inp)

# # Robot & genetics setup

# In[7]:
"""
if __name__=='__main__':
    # 1) Instantiate model
    time = 2.4
    x0 = 10
    y0 = 10
    tetta0 = 0
    xf = 0
    yf = 0
    tettaf = 0
    network = Network()
    robot = Robot(x0, y0, tetta0, xf, yf, tettaf)
    robot.set_simulation_config(time, integration_step=0.01, float_limit=1e+6, accuracy=0.01)
    # 2) Instantiate network operator
    unaries = [identity, negative, pow_two, sinus, logic, cosinus, atan, exp, natlog, irer, cubic, reverse, cubicroot,
               expm1, exp2, sign, log10, log2, log1p, absolute, tan, tanh, relu]#, sinc]
    binaries = [addition, multiplication, maximum, minimum]#, hypot, trapz]
    input_nodes = [0, 1, 2, 3, 4, 5]
    output_nodes = [9, 10]
    nop = NetworkOperator(unaries, binaries, input_nodes, output_nodes)
    q = [-0.5558393782649933, 7.167066632909471, 9.599063203868994]
    nop.set_q(q) # set some parameters
    nop.update_base_q() # set those parameters as base as well
    #psi = [[(0, 9), (1, 7), (3, 14), (6, 3)], [(1, 1), (4, 0), (2, 6), (7, 1)], [(2, 0), (5, 9), (3, 2), (8, 1)], [(6, 0), (8, 0), (7, 12), (5, 5), (7, 7), (9, 0)], [(6, 0), (7, 0), (5, 9), (10, 0)]]
    psi = [[(0, 2), (3, 0), (6, 3)], [(1, 0), (4, 0), (7, 1)], [(2, 0), (5, 0), (3, 1), (8, 1)], [(6, 0), (7, 4), (8, 0), (9, 3)], [(6, 0), (7, 0), (10, 0)]]
    #psi = [[(0, 2), (3, 0), (0, 15), (0, 15), (0, 15), (0, 15), (6, 3)], [(1, 0), (4, 0), (7, 1)], [(2, 0), (5, 0), (3, 1), (8, 1)], [(6, 0), (7, 4), (8, 0), (9, 3)], [(6, 0), (7, 0), (9, 19), (9, 19), (10, 3)]]
    nop.set_psi(psi) # set some structure
    nop.update_base_psi() # set this structure as base as well
    # 3) Instantiate structure genetic and pass model with nop to it
    sg = StructureGenetics(nop, robot)
    # 4) Set synthesis parameters
    #%lprun -f sg.optimize sg.optimize(qmin=-10, qmax=10, individuals=256, g=10, psi_change_epoch=2, variations_per_individual=8, r=32, ksearch=20)
    
    # psi_change_epoch <= individuals
    # ksearch <= individuals
    # variations_per_individuals >= 1
    # g > 0
    # crossings > 0
    
    sg.optimize(qmin=-10, qmax=10,
                individuals=50,
                generations=10,
                psi_change_epoch=2,
                variations_per_individual=10,
                crossings=30,
                ksearch=16)
    


# # Plot u(t), y(x) and obsticles

# ## Coordinates

# In[10]:

    robot.simulate()
    x, y = robot.get_x_history(), robot.get_y_history()
    plt.axes()
    circle = plt.Circle((5, 5), radius=2.5, lw=2.0, fc='y', edgecolor='black')
    plt.gca().add_patch(circle)
    plt.plot(x, y, 'b')
    plt.xlabel('${x}$',fontsize=20)
    plt.ylabel('${y}$',fontsize=20)
    plt.legend(['${y}({x})$'],loc='upper left')
    plt.axis('scaled')
    plt.show()


# In[9]:

    a = np.array([1,2,3])
    b = np.array([2,3,1])
    np.absolute(a-b)

"""

def main2():
    network = Network()
   
    #functional = network.train_and_score(network, 'cifar10')
    
    #print( functional*100)
    
    unaries = [identity, negative, pow_two, sinus, logic, cosinus, atan, exp, natlog, irer, cubic, reverse, cubicroot,
               expm1, exp2, sign, log10, log2, log1p, absolute, tan, tanh, relu]#, sinc]
    unariesn = ["(", "(-", "pow_two(", "sinus(", "logic(", "cosinus(", "atan(", "exp(", "natlog(", "irer(", "cubic(", "reverse(", "cubicroot(",
               "expm1(", "exp2(", "sign(", "log10(", "log2(", "log1p(", "absolute(", "tan(", "tanh(", "relu("]#, sinc]
    
    binaries = [addition, multiplication, maximum, minimum, atan2, pi_2_pi]#, hypot, trapz]
    binariesn = ["addition", "multiplication", "maximum", "minimum", "atan2", "pi_2_pi"]#, hypot, trapz]
    
    input_nodes = [0, 1, 2, 3, 4, 5]
    output_nodes = [6, 7, 10, 9, 3]
    nop = NetworkOperator(unaries, binaries, input_nodes, output_nodes)
    #q = []
    q = [1.0]
    nop.set_q(q) # set some parameters
    nop.update_base_q() # set those parameters as base as well
    #psi = [[(0, 9), (1, 7), (3, 14), (6, 3)], [(1, 1), (4, 0), (2, 6), (7, 1)], [(2, 0), (5, 9), (3, 2), (8, 1)], [(6, 0), (8, 0), (7, 12), (5, 5), (7, 7), (9, 0)], [(6, 0), (7, 0), (5, 9), (10, 0)]]
    #psi = [[(0, 2), (3, 0), (6, 3)], [(1, 0), (4, 0), (7, 1)], [(2, 0), (5, 0), (3, 1), (8, 1)], [(6, 0), (7, 4), (8, 0), (9, 3)], [(6, 0), (7, 0), (10, 0)]]
    #psi = [[(0, 2), (3, 0), (0, 15), (0, 15), (0, 15), (0, 15), (6, 3)], [(1, 0), (4, 0), (7, 1)], [(2, 0), (5, 0), (3, 1), (8, 1)], [(6, 0), (7, 4), (8, 0), (9, 3)], [(6, 0), (7, 0), (9, 19), (9, 19), (10, 3)]]
    psi = [[(4,0),(0,1),(6,0)],[(5,0),(1,1),(7,0)],[(7,0),(6,0),(8,4)],[(8,0),(2,1),(9,0)],[(9,0),(10,5)]]
    nop.set_psi(psi) # set some structure
    nop.update_base_psi() # set this structure as base as well
    
    sg = StructureGenetics(nop, network)
    
    functional = sg.estimate_object_function()
    print( functional*100)
    # 4) Set synthesis parameters
    #%lprun -f sg.optimize sg.optimize(qmin=-10, qmax=10, individuals=256, g=10, psi_change_epoch=2, variations_per_individual=8, r=32, ksearch=20)
    
    # psi_change_epoch <= individuals
    # ksearch <= individuals
    # variations_per_individuals >= 1
    # g > 0
    # crossings > 0
    
    be, bq, get_bp = sg.optimize(qmin=-1, qmax=1,
                individuals=50,
                generations=2, #10,
                psi_change_epoch=2,
                variations_per_individual=10,
                crossings=30,
                ksearch=16)
    
    names_of_pars=["x0","x1","x2","x3","x4"]
    for i in get_bp:
        #print(i)
        strr=binariesn[i[-1][1]]+'['
        for j in range(0,len(i)-1):
            #print(i[j][1])
            strr+=unariesn[int(i[j][1])]+names_of_pars[int(i[j][0])]+'), '
        strr=strr[:-2]+']'
        while(len(names_of_pars)<=i[-1][0]):
            names_of_pars.append('-')
        names_of_pars[i[-1][0]]=strr
    for k in output_nodes:
        #strr=str(int(k))+"   "
        print(k)
        print(names_of_pars[k])

def main3():
    network = Rover(1,1,1)
   
    #functional = network.train_and_score(network, 'cifar10')
    
    #print( functional*100)
    
    unaries = [identity, negative, pow_two, sinus, cosinus, atan, exp, natlog, irer, cubic, reverse, cubicroot,
               expm1, exp2, log10, log2, log1p, absolute, tan, tanh, inv, l, one]#, sinc]
    unariesn = ["(", "(-", "pow_two(", "sinus(", "logic(", "cosinus(", "atan(", "exp(", "natlog(", "irer(", "cubic(", "reverse(", "cubicroot(",
               "expm1(", "exp2(", "sign(", "log10(", "log2(", "log1p(", "absolute(", "tan(", "tanh(", "relu(", "inv(", "L(", "one("]#, sinc]
    
    binaries = [addition, multiplication, maximum, minimum, atan2, pi_2_pi]#, hypot, trapz]
    binariesn = ["addition", "multiplication", "maximum", "minimum", "atan2", "pi_2_pi"]#, hypot, trapz]
    #              v cx yaw cyaw k  e
    input_nodes = [0, 1]
    output_nodes = [18]
    nop = NetworkOperator(unaries, binaries, input_nodes, output_nodes)
    #q = []
    #         KTH   KE
    q = [1.0, 1.0, 0.5]
    nop.set_q(q) # set some parameters
    nop.update_base_q() # set those parameters as base as well
    #psi = [[(0, 9), (1, 7), (3, 14), (6, 3)], [(1, 1), (4, 0), (2, 6), (7, 1)], [(2, 0), (5, 9), (3, 2), (8, 1)], [(6, 0), (8, 0), (7, 12), (5, 5), (7, 7), (9, 0)], [(6, 0), (7, 0), (5, 9), (10, 0)]]
    #psi = [[(0, 2), (3, 0), (6, 3)], [(1, 0), (4, 0), (7, 1)], [(2, 0), (5, 0), (3, 1), (8, 1)], [(6, 0), (7, 4), (8, 0), (9, 3)], [(6, 0), (7, 0), (10, 0)]]
    #psi = [[(0, 2), (3, 0), (0, 15), (0, 15), (0, 15), (0, 15), (6, 3)], [(1, 0), (4, 0), (7, 1)], [(2, 0), (5, 0), (3, 1), (8, 1)], [(6, 0), (7, 4), (8, 0), (9, 3)], [(6, 0), (7, 0), (9, 19), (9, 19), (10, 3)]]
    """
    th_e = pi_2_pi(state.yaw - cyaw[ind])

    omega = v * k * math.cos(th_e) / (1.0 - k * e) - \
        KTH * abs(v) * th_e - KE * v * math.sin(th_e) * e / th_e
    th_e=pi_2_pi[add[state.yaw,(-cyaw[ind])] ] 
    o1=mult[v,k,cos[th_e], inv(add[q0,mult[k,e] ]) ]
    o2=mult[q1,abs(v),th_e]
    o3=mult[q2,v,sin(th_e),e,inv(th_e)]
    o=add(o1,o2,o3)
    delta = math.atan2(L * omega / v, 1.0)
    d=atan2[mult[L, o, inv(v)], 1.0 ]"""
    #psi=[[(2"""yaw""",0),(3"""cyaw""",1),(9,0)],[(9,0),(10"""th_e""",5)],[(4"""k""",0),(5"""e""",0),(11,1)],[(6,0),(11,0),(12,0)],
    #      [(0"""v""",0),(4"""k""",0),(10,5),(12,23),(13"""o1""",1)],[(7,0),(0"""v""",19),(10,0),(14"""o2""",1)],
    #      [(8,0),(0"""v""",0),(10,3),(5"""e""",0),(10,23),(15"""o3""",1)],[(13,0),(14,0),(15,0),(16"""o""",0)],
    #      [(0,24"""L"""),(16"""o""",0),(0"""v""",23),(17,1)],[(17,0),(0,25"""1.0"""),(18"""d""",4)]]
    #psi=[[(5,0),(6,1),(9,0)],[(9,0),(10,5)],[(7,0),(8,0),(11,1)],[(0,0),(11,0),(12,0)],
    #      [(3,0),(7,0),(10,5),(12,23),(13,1)],[(1,0),(3,19),(10,0),(14,1)],
    #      [(2,0),(3,0),(10,3),(8,0),(10,23),(15,1)],[(13,0),(14,0),(15,0),(16,0)],
    #      [(3,24),(16,0),(3,23),(17,1)],[(17,0),(3,25),(18,4)]]
    #psi=[[(2,0),(3,1),(9,0)],[(9,0),(10,5)],[(4,0),(5,0),(11,1)],[(6,0),(11,0),(12,0)],
    #      [(0,0),(4,0),(10,5),(12,23),(13,1)],[(7,0),(0,19),(10,0),(14,1)],
    #      [(8,0),(0,0),(10,3),(5,0),(10,23),(15,1)],[(13,0),(14,0),(15,0),(16,0)],
    #      [(0,24),(16,0),(0,23),(17,1)],[(17,0),(0,25),(18,4)]]
    
    #psi = [[(1,0),(0,1),(5,0)],[(1,0),(0,1),(6,0)],[(0,0),(1,1),(7,0)],[(7,0),(6,0),(8,4)],[(8,0),(0,1),(9,0)],[(9,0),(18,5)]]
    
    psi = [[(0,0),(2,0),(5,0)],[(5,0),(18,5)]]
    nop.set_psi(psi) # set some structure
    nop.update_base_psi() # set this structure as base as well
    
    sg = StructureGenetics(nop, network)
    
    functional = sg.estimate_object_function()
    print( functional*100)
    # 4) Set synthesis parameters
    #%lprun -f sg.optimize sg.optimize(qmin=-10, qmax=10, individuals=256, g=10, psi_change_epoch=2, variations_per_individual=8, r=32, ksearch=20)
    
    #f=open("Test_formula.txt", "a")
    #print("Start of genetic algorithm")
    f.write("\n\n\n\n\n\nNEW FORMULA\n\n\n")
    
    # psi_change_epoch <= individuals
    # ksearch <= individuals
    # variations_per_individuals >= 1
    # g > 0
    # crossings > 0
    be, bq, get_bp = sg.optimize(qmin=-1, qmax=1,
                individuals=50,
                generations=1, #10,
                psi_change_epoch=2,
                variations_per_individual=10,
                crossings=30,
                ksearch=16)
    
    f.write(be)
    f.write(bq)
    f.write(get_bp)
    names_of_pars=["x0","x1","x2","x3","x4"]
    for i in get_bp:
        #print(i)
        strr=binariesn[i[-1][1]]+'['
        for j in range(0,len(i)-1):
            #print(i[j][1])
            strr+=unariesn[int(i[j][1])]+names_of_pars[int(i[j][0])]+'), '
        strr=strr[:-2]+']'
        while(len(names_of_pars)<=i[-1][0]):
            names_of_pars.append('-')
        names_of_pars[i[-1][0]]=strr
    f.write("\nDRUGOI VID\n")
    for k in output_nodes:
        #strr=str(int(k))+"   "
        f.write(str(k))
        f.write(str(names_of_pars[k]))
        print(k)
        print(names_of_pars[k])
    f.close()

if __name__=='__main__':
    main3()
    
    
