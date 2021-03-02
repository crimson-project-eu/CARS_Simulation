# CARS Simulation

Script for simulating coherent anti-stokes Raman spectra is written with respect to the fundamental formulation as follows:

<img src="https://render.githubusercontent.com/render/math?math=I_{CARS}\propto |\chi^{(3)}|^2 I_{pu}^2 I_S">
<img src="https://render.githubusercontent.com/render/math?math=\chi^{(3)} = \chi^{(3)}_{NR} %2B \chi^{(3)}_R">

Where <img src="https://render.githubusercontent.com/render/math?math=\chi^{(3)}(\omega)"> can be written as:
<img src="https://render.githubusercontent.com/render/math?math=\chi^{(3)}(\omega) = \sum_i \frac{A_i}{\omega - \Omega_i - i\Tau_i}">

In this code, the user has a choice for the following:
- The spectral regions of 300-3300 cm<sup>-1</sup> and 300-1800 cm<sup>-1</sup>.
- Sigma parameter for Gaussian Noise.
- Simulation of CARS spectra with or without Stokes intensity.

The simulation is performed such that non resonating background (NRB) is linear or constant and bounded in a given interval. The maximum number of peaks for each spectrum are
restricted to 15 and line width for each peak lies in the interval [4,40] cm<sup>-1</sup>. Stokes intensity is simulated as a long Gaussian function or multiplication of two 
sigmoidal <img src="https://render.githubusercontent.com/render/math?math=\left(\sigma(x) = \frac{1}{1 %2B e^{-(x-a)b}}\right)"> functions; whereby one of these are chosen at random for each instance.


### Basic libraries required
- numpy
- random
- matplotlib


### Running Script

* To Run
```
$ python CARS_Simulation_v1.py
```

### References
- [Deep learning as phase retrieval tool for CARS spectra](https://doi.org/10.1364/OE.390413)
- [Removing non-resonant background from CARS spectra via deep learning](https://doi.org/10.1063/5.0007821)
