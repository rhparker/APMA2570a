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
	return np.exp(np.sin(x))

# derivative of function
def df(x):
	return np.exp(np.sin(x)) * np.cos(x)

# differentiation matrix for trigonometric polynomials
# x is grid of x points
def diff_matrix(x):
	N = len(x)
	a = np.zeros( ( N, N) )
	for i in xrange(N):
		for j in xrange(N):
			if i != j:
				a[j][i] = (-1.0)**(i+j) / ( 2 * np.tan( (x[i] - x[j]) / 2.0 ) )
	return a

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

def trig_poly(x, coeffs):
	return np.real( np.sum( [ coeffs[i] * np.exp(1j * modes[i] * x) for i in xrange(len(modes)) ] ) )

# min and max values on the axes
xmin = 0.
xmax = 2. * np.pi

# specifics for data
gridsizes = [4, 8, 16, 32 ]
L2errors = np.zeros( (len(gridsizes), 4 ) )

# column headers for table
print " grid            PnU                       PnU'                      InU                      (InU)' "
print "  pts  L2 error       order      L2 error       order      L2 error       order      L2 error       order"
print "----- -----------------------  ------------------------  ------------------------  ---------------------------"

for j, N in enumerate(gridsizes):
	# construct the grid of points, and remove the final point, 
	# since it is equated with the initial point
	xgrid = np.linspace(xmin, xmax, N + 2)
	xgrid = xgrid[:-1]

	# fourier modes, from -N/2 to N/2
	modes = range(-N/2, N/2 + 1)

	orders = np.zeros( 4 )

	P_coeffs  = [ fourier_coeff(f, n) for n in modes ]
	dP_coeffs = [ 1j * modes[i] * P_coeffs[i] for i in xrange(len(modes)) ]
	I_coeffs  = [ interp_coeff(f, xgrid, n) for n in modes ]
	dI_coeffs = [ 1j * modes[i] * I_coeffs[i] for i in xrange(len(modes)) ]


	P_error  = L2norm(lambda x: f(x) - trig_poly(x, P_coeffs), 0, 2*np.pi)
	dP_error = L2norm(lambda x: f(x) - trig_poly(x, I_coeffs), 0, 2*np.pi)
	I_error  = L2norm(lambda x: df(x) - trig_poly(x, dP_coeffs), 0, 2*np.pi)
	dI_error = L2norm(lambda x: df(x) - trig_poly(x, dI_coeffs), 0, 2*np.pi)

	L2errors[j] = [ P_error, dP_error, I_error, dI_error ]

	# compute orders
	if j > 0:
		for i in xrange(4):
			orders[i] = np.log2( L2errors[j-1][i]/L2errors[j][i] )
 
	display_data = np.hstack( zip(L2errors[j], orders) )

	print "{:4d}   {:10e}   {:8.5f}   {:10e}   {:8.5f}   {:10e}   {:8.5f}   {:10e}   {:8.5f}".format( N, *display_data )

# set up plot
fig = plt.figure()
scat = fig.add_subplot(111)
marker_size = 25
colors = ['red', 'blue', 'green', 'orange']
markers = ['+', '*', 'o', 's']
labels = [ "||U - PnU||    ", "||U' - PnU'||  ", "||U - InU||    ", "||U' - (InU)'||" ]
print

# error plot: log of L2 error vs grid size
logL2errors = np.transpose( np.log( L2errors ) )

for j in xrange(4):
	plot1 = scat.scatter( gridsizes, logL2errors[j], color=colors[j], label=labels[j], s=marker_size, marker=markers[j])
	poly = np.polyfit( gridsizes, logL2errors[j], 1)
	print labels[j], "   slope of line: ", poly[0]
	plt.plot( gridsizes, np.poly1d( poly )(gridsizes), color=colors[j] )
plt.xlabel('grid size (N)')
plt.ylabel('log L2 error')
plt.title('Log of L2 error vs grid size')
plt.legend()
plt.show()


