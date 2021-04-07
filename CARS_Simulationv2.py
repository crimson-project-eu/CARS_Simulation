# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 11:31:20 2021
@author: Roopam K. Gupta
Edited by: Leone De Marco
"""

import argparse
from argparse import RawTextHelpFormatter
import argcomplete
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from multiprocessing import Pool
from functools import wraps
from time import time
from datetime import datetime as dt
import pickle


def timer(func):
    """ Prints execution time of function func """
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            print(f"Total execution time: {end_ if end_ > 0 else 0} ms")
    return _time_it

@timer
def Simul_n_CARS(n_examples:int, MaxNumPeaks: int, w_min:int, w_max:int, x: np.ndarray, MaxNRB: int, MinNRB: int, n_points:int, sigma:float, scenario:int, stokes:int):
    X = np.empty((n_examples, n_points))
    y = np.empty((n_examples, n_points))
    for i in range(n_examples):
        cars,imag,real,nrb = SimulCARS(MaxNumPeaks=MaxNumPeaks, w_min=w_min, w_max=w_max, x=x, MaxNRB=MaxNRB, MinNRB=MinNRB, n_points=n_points, sigma=sigma, scenario=scenario, stokes=stokes)
        X[i, :], y[i, :] = cars, imag
    return X, y


@timer
def Simul_n_CARS_parallel(n_threads:int, n_examples:int, MaxNumPeaks: int, w_min:int, w_max:int, x: np.ndarray, MaxNRB: int, MinNRB: int, n_points:int, sigma:float, scenario:int, stokes:int):
    pass



def SimulCARS(MaxNumPeaks: int, w_min:int, w_max:int, x: np.ndarray, MaxNRB: int, MinNRB: int, n_points:int, sigma:float, scenario:int, stokes:int):
    """ Simulates and returns CARS, imaginary part, real part, and """
    # Chi3_R simulation
    NumPeaks = np.random.randint(1,MaxNumPeaks+1) # randomly get the number of peaks in the interval [0,MaxNumPeaks]
    Amp = np.random.uniform(0.0001,0.005,NumPeaks) # amplitude variation in the interval [0.1,5)
    w = np.random.uniform(0,1,NumPeaks) # resonance position in the interval [0,1)
    g = np.random.uniform(w_min,w_max+2e-308,NumPeaks) # line width in the interval [w_min,w_max]
    Chi3_R = np.sum(Amp/(-x[:,np.newaxis]+w-1j*g),axis = 1)
    Chi3_R = Chi3_R/np.max(np.abs(Chi3_R))
    
    # NRB Simulation
    # Restricting NRB in the range (MinNRB,MaxNRB)
    st1 = np.random.uniform(-1*MaxNRB,MaxNRB)
    if st1<MinNRB:
        st2 = np.random.uniform(-1*st1 - MinNRB,MaxNRB)
    else:
        st2 = np.random.uniform(MinNRB,MaxNRB-st1)
    Chi3_NR = st1*x + st2
    if np.max(Chi3_NR)>MaxNRB or np.min(Chi3_NR)<MinNRB:
        Chi3_NR = (MaxNRB-MinNRB)*((Chi3_NR-np.min(Chi3_NR))/(np.max(Chi3_NR)-np.min(Chi3_NR))) + MinNRB
    
    # Simulating Stokes Intensity
    if stokes == 0:
        choiceList = [1,2] # randomly choose between long gaussian or sigmoidal function
    else:
        choiceList = [stokes]
    choice = random.choice(choiceList)
    if choice == 1: # Sigmoidal Stokes intensity
            bs = np.random.normal(3,5,2)
            c1 = np.random.normal(0.1,0.3)
            c2 = np.random.normal(0.7,.3)
            cs = np.r_[c1,c2]
            
            sig1 = 1/(1+np.exp(-(x-cs[0])*bs[0]))
            sig2 = 1/(1+np.exp(-(x-cs[1])*-1*bs[1]))
            I_S = sig1*sig2
    elif choice == 2: # Gaussian Stokes intensity
        mu = np.random.uniform(0,1)
        sig = np.random.randint(500,1500)
        I_S = np.abs(np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))
    
    if np.min(I_S<0):
        I_S = (I_S - np.min(I_S))
    
    # simulating CARS
    Chi3 = Chi3_R + Chi3_NR
    noise = np.random.randn(n_points) * sigma # adding noise to the spectra with standard deviation sigma
    if scenario==1:    
        cars = (np.abs(Chi3)**2)*I_S + noise
    else:
        cars = np.abs(Chi3)**2 + noise
    return cars, Chi3_R.imag, Chi3.real, Chi3_NR


def main():
    # input parameters
    params = vars(args)
    AxisVal = args.region
    if AxisVal == 1:
        n_points = 1024
        xAxis = np.linspace(300,3300,n_points)
    else:
        n_points = 512
        xAxis = np.linspace(300,1800,n_points)
    scenario = args.scenario
    sigma = args.sigma
    savefig_flag = args.savefig.lower()
    MaxNRB = 5
    MinNRB = 1
    MaxNumPeaks = 15
    NumExamples = args.num_examples
    x = np.linspace(0,1,n_points)
    stokes=args.stokes
    # Peak width variation between 4-40 cm^-1 with respect to peaks in spontaneous Raman
    w_min = (4*(x[2]-x[1]))/(xAxis[1]-xAxis[0]) #cm^-1 -> px
    w_max = (40*(x[2]-x[1]))/(xAxis[1]-xAxis[0]) #cm^-1 -> px
    X, y = Simul_n_CARS(n_examples=NumExamples, MaxNumPeaks=MaxNumPeaks, w_min=w_min, w_max=w_max, x=x, MaxNRB=MaxNRB, MinNRB=MinNRB, n_points=n_points, sigma=sigma, scenario=scenario, stokes=stokes)
    output = {'parameters': params, 'data' : (X, y), 'date': dt.now().date().strftime('%Y-%m-%d')}
    
    with open(args.outfile, 'wb') as outfile:
        pickle.dump(output, outfile)
    print('Saved {} examples in file {}'.format(X.shape[0], args.outfile))
    
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument( "-r", "--region", choices=[1,2], type=int, default=2, help='choice of spectral region:\n[1] 300 - 3300 cm^-1\n[2] 300 - 1800 cm^-1')
    parser.add_argument( "-s", "--sigma", type=float, default=1, help='sigma value for noise')
    parser.add_argument( "-t", "--stokes", type=int, default=0, help='stokes intensity: 0 --> random\n1--> sigmoidal\n2--> gaussian')
    parser.add_argument( "-i", "--scenario", choices=[0,1], type=int, default=1, help='choice of scenario:\n[0] Without Stokes instensity\n[1] With Stokes intensity')
    parser.add_argument( "-f", "--savefig", choices=['y','Y', 'n', 'N'], type=str, default='n', help='Save example figures as pdf? \n[y] Yes\n[n] No')
    parser.add_argument( "-n", "--num_examples", type=int, default=20000, help='number of examples to generate')
    parser.add_argument( "-o", "--outfile", type=str, default='./Data/cars.pkl', help='output file for pickled dataset')
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    main()