# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 11:31:20 2021
@author: Roopam K. Gupta
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

AxisVal = int(input("Enter the choice of spectral region [1] 300 - 3300 cm^-1 [2] 300 - 1800 cm^-1: "))
if AxisVal == 1:
    n_points = 1024
    xAxis = np.linspace(300,3300,n_points)
elif AxisVal == 2:
    n_points = 512
    xAxis = np.linspace(300,1800,n_points)

sigma = float(input("Enter sigma value for noise: "))
flg = int(input("Save example figures as pdf? [1] Yes [2] No: "))

scenario = int(input("Enter the scenario [0] Without Stokes instensity [1] With Stokes intensity"))

MaxNRB = 5
MinNRB = 1
MaxNumPeaks = 15
NumExamples = 20
x = np.linspace(0,1,n_points)

# Peak width variation between 4-40 cm^-1 with respect to peaks in spontaneous Raman
w_min = (4*(x[2]-x[1]))/(xAxis[1]-xAxis[0]) #cm^-1 -> px
w_max = (40*(x[2]-x[1]))/(xAxis[1]-xAxis[0]) #cm^-1 -> px

def SimulCARS():
    # Chi3_R simulation
    NumPeaks = np.random.randint(1,MaxNumPeaks+1) # randomly get the number of peaks in the interval [0,MaxNumPeaks]
    Amp = np.random.uniform(0.0001,0.005,NumPeaks) # amplitude variation in the interval [0.1,5)
    w = np.random.uniform(0,1,NumPeaks) # resonance position in the interval [0,1)
    g = np.random.uniform(w_min,w_max+2e-308,NumPeaks) # line width in the interval [w_min,w_max]
    Chi3_R = np.sum(Amp/(-x[:,np.newaxis]+w-1j*g),axis = 1)
    Chi3_R = Chi3_R/np.max(np.abs(Chi3_R))
    
    # NRB Simulation
    # Restricting NRB in the range (MinNRB,MaxNRB)
    st1 = np.random.uniform(-1*MaxNRB,MaxNRB);
    if st1<MinNRB:
        st2 = np.random.uniform(-1*st1 - MinNRB,MaxNRB)
    else:
        st2 = np.random.uniform(MinNRB,MaxNRB-st1)
    Chi3_NR = st1*x + st2
    if np.max(Chi3_NR)>MaxNRB or np.min(Chi3_NR)<MinNRB:
        Chi3_NR = (MaxNRB-MinNRB)*((Chi3_NR-np.min(Chi3_NR))/(np.max(Chi3_NR)-np.min(Chi3_NR))) + MinNRB
    
    # Simulating Stokes Intensity
    choiceList = [1,2] # randomly choose between long gaussian or sigmoidal function
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
    noise = np.random.randn(n_points) * sigma # adding noise to the spectra with standard deivation sigma
    if scenario==1:    
        cars = (np.abs(Chi3)**2)*I_S + noise
    else:
        cars = np.abs(Chi3)**2 + noise
    return cars, Chi3_R.imag, Chi3.real, Chi3_NR

if flg==1:
    print('Example figures saved in a pdf file.')
    fN = str(input('Enter the file name: '))
    fN = fN + '.pdf'
    pdf = matplotlib.backends.backend_pdf.PdfPages(fN)
    for i in range(NumExamples):
        fig = plt.figure(i)
        
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)
        
        cars,imag,real,nrb = SimulCARS()
        ax1.plot(xAxis,cars,label = 'CARS',c='g')
        ax2.plot(xAxis,imag,label = 'Imaginary',c='r')
        ax3.plot(xAxis,real,label = 'Real',c='b')
        ax4.plot(xAxis,nrb,label = 'NRB',c='y')
        
        ax1.legend(shadow=True, fancybox=True)
        ax2.legend(shadow=True, fancybox=True)
        ax3.legend(shadow=True, fancybox=True)
        ax4.legend(shadow=True, fancybox=True)
        
        ax1.set_xticks([])
        ax2.set_xticks([])
        ax3.set_xticks([])
        
        fig.text(0.5, 0.03, 'Wavenumber / cm$^{-1}$', ha='center', va='center')
        fig.text(0.03, 0.5, 'Raman intensity / arb.u.', ha='center', va='center', rotation='vertical')

        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()
else:
    for i in range(NumExamples):
        #fig = plt.figure(i)
        cars,imag,real,nrb = SimulCARS()
        fig = plt.figure()
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)
        
        ax1.plot(xAxis,cars,label = 'CARS',c='g')
        ax2.plot(xAxis,imag,label = 'Imaginary',c='r')
        ax3.plot(xAxis,real,label = 'Real',c='b')
        ax4.plot(xAxis,nrb,label = 'NRB',c='y')
        
        ax1.legend(shadow=True, fancybox=True)
        ax2.legend(shadow=True, fancybox=True)
        ax3.legend(shadow=True, fancybox=True)
        ax4.legend(shadow=True, fancybox=True)
        
        ax1.set_xticks([])
        ax2.set_xticks([])
        ax3.set_xticks([])
        
        fig.text(0.5, 0.03, 'Wavenumber / cm$^{-1}$', ha='center', va='center')
        fig.text(0.03, 0.5, 'Raman intensity / arb.u.', ha='center', va='center', rotation='vertical')