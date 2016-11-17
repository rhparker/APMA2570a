#
# Gaussian quadrature
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
def f(x, m):
	return m * np.exp(np.sin(m*x)) * np.cos(m*x)

# exact integral
def exact_int(m):
	xbar = (2 + 0.5/m) * np.pi
	return -1 + np.exp( np.sin(m * xbar))

def gauss_leg(n, m):
	# points and weights for Gauss-Legendre quadrature
	pts, wts = np.polynomial.legendre.leggauss(n)

	# endpoint of integration is xbar
	xbar = (2 + 0.5/m) * np.pi

	# translate x values from std interval [-1, 1] to [0, xbar]
	t = 0.5 * ( pts + 1 ) * xbar

	return sum(wts * f(t, m)) * 0.5 * xbar

def gauss_cheb(n, m):
	# points and weights for Gauss-Chebyshev quadrature
	pts, wts = np.polynomial.chebyshev.chebgauss(n)
	
	# 1/weight needed since Gauss-Chebyshev multiplies by weight
	recip_wt_fn = lambda x: np.sqrt(1 - x**2)

	# endpoint of integration is xbar
	xbar = (2 + 0.5/m) * np.pi

	# translate x values from std interval [-1, 1] to [0, xbar]
	t = 0.5 * ( pts + 1 ) * xbar

	return sum( wts * f(t, m) * recip_wt_fn(pts) ) * 0.5 * xbar

# compute L2 norm (nondiscrete) of f on finite interval [a, b]
def L2norm(f, a, b):
	integrand = lambda x : abs(f(x))**2
	return np.sqrt( quad( integrand, a, b)[0] )

gridsizes = [ 4, 8, 16, 32, 64, 128, 256 ]
mvalues = [1, 2, 3, 4, 5]
errors = np.zeros( ( 5, len(gridsizes), 2 ) )

for m in mvalues:
	print "m = ",m
	# column headers for table
	print " grid        Legendre                 Chebyshev "
	print "  pts  error           order    error            order  "
	print "----- -----------------------  ------------------------ "
	xbar = (2 + 0.5/m) * np.pi
	exact_value = exact_int(m)
	for j, N in enumerate(gridsizes):
		errors[m-1][j] = [ abs(gauss_leg(N, m) - exact_value), abs(gauss_cheb(N, m) - exact_value) ]

		orders = np.zeros(2)
		# compute orders
		if j > 0:
			for i in xrange(2):
				if abs(errors[m-1][j][i]) > 10**(-18):
					orders[i] = np.log2( errors[m-1][j-1][i]/errors[m-1][j][i] )
				else:
					orders[i] = 0
 
		display_data = np.hstack( zip(errors[m-1][j], orders) )

		print "{:4d}   {:10e}   {:8.5f}   {:10e}   {:8.5f}".format( N, *display_data )
	print

# set up plot
fig = plt.figure()
scat = fig.add_subplot(111)
marker_size = 25
colors = ['red', 'blue', 'green', 'orange', 'black']
markers = ['+', '*', 'o', 's', 'o']
labels = mvalues

logL2errors = np.transpose( np.log( L2errors ) )

# for j in xrange(4):
# 	plot1 = scat.scatter( gridsizes, logL2errors[j], color=colors[j], label=labels[j], s=marker_size, marker=markers[j])
# 	poly = np.polyfit( gridsizes, logL2errors[j], 1)
# 	print labels[j], "   slope of line: ", poly[0]
# 	plt.plot( gridsizes, np.poly1d( poly )(gridsizes), color=colors[j] )
# plt.xlabel('grid size (N)')
# plt.ylabel('log L2 error')
# plt.title('Log of L2 error vs grid size')
# plt.legend()
# plt.show()


