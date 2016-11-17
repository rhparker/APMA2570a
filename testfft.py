#
# advection equation
#

# basic math functions from numpy
import numpy as np

# Gaussian quadrature for numerical intergration
from scipy.integrate import quad

# argparse to parse command line arguments
import argparse

# pyplot plot class plot classes from matplotlib
import matplotlib.pyplot as plt

# function to use
def f(x):
	return np.sin(x)

# compute L2 norm (nondiscrete) of f on finite interval [a, b]
def L2norm(f, a, b):
	integrand = lambda x : abs(f(x))**2
	return np.sqrt( quad( integrand, a, b)[0] )

# function does Gaussian quadrature on a complex function
def complex_quad(func, a, b, **kwargs):
    def real_func(x):
        return np.real(func(x))
    def imag_func(x):
        return np.imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return real_integral[0] + 1j*imag_integral[0]

def fourier_coeff(u, n):
	integrand = lambda x: f(x) * np.exp(-1j * n * x)
	return complex_quad(integrand, 0, 2 * np.pi) / (2 * np.pi)

def interp_coeff(u, x, n):
	N = len(x)
	return np.sum( [ u(x[i]) * np.exp(-1j * n * x[i] ) for i in xrange(N) ] ) / float(N)

def trig_poly_fft(x, coeffs, N):
	return np.real( np.sum( [ coeffs[i] * np.exp(1j * k * x) for i, k in enumerate(modes) ] ) ) / (N+1)

# min and max values on the axes
xmin = 0.
xmax = 2. * np.pi

fig = plt.figure()
scat = fig.add_subplot(111)
colors = ['red', 'blue', 'green', 'orange']
marker_size = 5

# specifics for data
gridsizes = [ 2, 4, 8, 16, 32 ]

for j, N in enumerate(gridsizes):
	# construct the grid of points, and remove the final point, 
	# since it is equated with the initial point
	plt.cla()
	x = np.linspace(xmin, xmax, N + 2)
	x = x[:-1]

	xgrid = np.linspace(xmin, xmax, 100)

	# fourier modes
	modes = np.array( range(0, N/2 + 1) + range(-N/2, 0) )

	y = f(x)
	a = np.fft.fft(y)

	y1 = [ trig_poly_fft(i, a, N) for i in xgrid ]

	plot1 = scat.scatter(xgrid, f(xgrid), color=colors[0], s=marker_size)
	plot2 = scat.scatter(xgrid, y1, color=colors[1], s=marker_size)
	plt.show()





	


