# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 12:14:46 2021

@author: Roopam K. Gupta

The script outputs a pdf file with filename "fN" containing predictions of randomly 
simulated CARS spectra for the two models "Model1_name" and "Model2_name"

Note: Before using the script, following variables need to be named:
    1. Model1_name
    2. Model2_name
    3. fN

By default, the CARS spectra are simulated with the following parameters which
can be changed as required:
    AxisVal = 1 for xAxis between 300 to 3300 cm-1
    sigma = 0.01
    MaxNRB = 5 
    MinNRB = 1
    MaxNumPeaks = 15
"""
import tensorflow as tf
import numpy as np
import random
import progressbar
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

AxisVal = 1 # int(input("Enter the choice of spectral region [1] 300 - 3300 cm^-1 [2] 300 - 1800 cm^-1: "))
if AxisVal == 1:
    n_points = 1024
    xAxis = np.linspace(300,3300,n_points)
elif AxisVal == 2:
    n_points = 512
    xAxis = np.linspace(300,1800,n_points)

sigma = 0.01 # float(input("Enter sigma value for noise: "))

MaxNRB = 5
MinNRB = 1
MaxNumPeaks = 15

Model1_name = '' #h5 file name of the model1 - output dimension = 3
Model2_name = '' #h5 file name of the model2 - output dimension = 2
fN = '' #Name of file with extension pdf; eg: Results.pdf

x = np.linspace(0,1,n_points)

# Peak width variation between 4-40 cm^-1 with respect to peaks in spontaneous Raman
w_min = (4*(x[2]-x[1]))/(xAxis[1]-xAxis[0]) #cm^-1 -> px
w_max = (40*(x[2]-x[1]))/(xAxis[1]-xAxis[0]) #cm^-1 -> px

def SimulCARS():
    # Chi3_R simulation
    NumPeaks = np.random.randint(1,MaxNumPeaks+1) # randomly get the number of peaks in the interval [0,MaxNumPeaks]
    Amp = np.random.uniform(0.0001,0.005,NumPeaks) # amplitude variation in the interval [0.1,5)
    w = np.random.uniform(0,1+2e-308,NumPeaks) # resonance position in the interval [0,1)
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
    
    # Simulating Stokes Spectrum
    choiceList = [1,2] # randomly choose between long gaussian or sigmoidal function
    choice = random.choice(choiceList)
    if choice == 1: # sigmoidal Stokes intensity
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
    
    # Simulating CARS
    Chi3 = Chi3_R + Chi3_NR
    noise = np.random.randn(n_points) * sigma # adding noise to the spectra with standard deivation sigma
    cars = (np.abs(Chi3)**2)*I_S + noise
    return cars, Chi3_R.imag, Chi3_R.real, Chi3_NR

def generateSpectra(size = 10000):
    cars = np.empty((size, n_points,1))
    chi_imag = np.empty((size,n_points))
    chi_real = np.empty((size,n_points))
    nrb = np.empty((size,n_points))
    for i in range(size):
        cars[i,:,0], chi_imag[i,:], chi_real[i,:], nrb[i,:] = SimulCARS()
    return cars,chi_imag,chi_real,nrb

numPlots = 200
CARS_Sim,Imag_CARS,Real_CARS,NRB_CARS = generateSpectra(numPlots)

print('Normalizing the simulated CARS spectra and its imaginary part')
with progressbar.ProgressBar(max_value=200) as bar:
    for i in range(numPlots):
        CARS_Sim[i,:,0] = (CARS_Sim[i,:,0] - min(CARS_Sim[i,:,0]))/(max(CARS_Sim[i,:,0]) - min(CARS_Sim[i,:,0]))
        Imag_CARS[i,:] = (Imag_CARS[i,:] - min(Imag_CARS[i,:]))/(max(Imag_CARS[i,:]) - min(Imag_CARS[i,:]))
        bar.update(i)
        

model1 = tf.keras.models.load_model(Model1_name)
model2 = tf.keras.models.load_model(Model2_name)

test_y_predictions_Model1 = model1.predict(CARS_Sim)
test_y_predictions_Model2 = model2.predict(CARS_Sim)

print('Nomalizing the model predictions')
with progressbar.ProgressBar(max_value=numPlots) as bar:
    for i in range(numPlots):
        test_y_predictions_Model1[i,:,0] = (test_y_predictions_Model1[i,:,0] - min(test_y_predictions_Model1[i,:,0]))/(max(test_y_predictions_Model1[i,:,0]) - min(test_y_predictions_Model1[i,:,0]))
        test_y_predictions_Model2[i,:] = (test_y_predictions_Model2[i,:] - min(test_y_predictions_Model2[i,:]))/(max(test_y_predictions_Model2[i,:]) - min(test_y_predictions_Model2[i,:]))
        bar.update(i)

print('Plotting the results')
with progressbar.ProgressBar(max_value=numPlots) as bar:
    pdf = matplotlib.backends.backend_pdf.PdfPages(fN)
    for i in range(numPlots):
        fig = plt.figure(i)
        
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)
    
        ax1.plot(CARS_Sim[i,:],label = 'CARS',c='g')
        ax2.plot(Imag_CARS[i,:],label = 'Imaginary CARS',c='r')
        ax3.plot(test_y_predictions_Model1[i,:,0],label = 'Imag Model1',c='b')
        ax4.plot(test_y_predictions_Model2[i,:],label = 'Imag Model2',c='y')
        
        ax1.legend(shadow=True, fancybox=True)
        ax2.legend(shadow=True, fancybox=True)
        ax3.legend(shadow=True, fancybox=True)
        ax4.legend(shadow=True, fancybox=True)
        
        ax1.set_xticks([])
        ax2.set_xticks([])
        ax3.set_xticks([])
        
        fig.text(0.5, 0.03, 'Wavenumber / arb.u.', ha='center', va='center')
        fig.text(0.03, 0.5, 'Raman intensity / arb.u.', ha='center', va='center', rotation='vertical')

        pdf.savefig(fig)
        plt.close(fig)
        bar.update(i)
    pdf.close()
