#
# interpolation, polynomial basis
#

# basic math functions from numpy
import numpy as np

# Gaussian quadrature for numerical intergration
from scipy.integrate import quad

# individual legendre and chebyshev polynomials
from scipy.special import legendre

# argparse to parse command line arguments
import argparse

# pyplot plot class plot classes from matplotlib
import matplotlib.pyplot as plt

# function to use
def f(x):
	return 1.0 / ( 1.0 + x**2 )

# derivative of function
def df(x):
	return -2.0 * x / ( (1.0 + x**2)**2 )

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

def disc_ip(f, g, pts, wts):
	return sum( f(pts) * g(pts) * wts )

def disc_norm(f, pts, wts):
	return np.sqrt( disc_ip(f, f, pts, wts) )

def disc_norm_legendre(n):
	return 2.0/((2.0 * n) + 1)

def legendre_coeff(upts, n, pts, wts):
	# Pn = legendre(n)
	Pn = np.polynomial.legendre.Legendre.basis(n)
	return (1.0/disc_norm_legendre(n)) * sum( upts * Pn(pts) * wts )

def cheb_coeff(upts, n, N):
	if n == 0:
		cn = 2
	else:
		cn = 1
	Tn = np.polynomial.chebyshev.Chebyshev.basis(n)
	return (2.0 / (cn * (N+1))) * sum( upts * Tn(pts) )

def legendre_diff_matrix(n, pts):
	# first derivative of P_{N+1}
	DL = np.polynomial.legendre.Legendre.basis(n+1).deriv(1)
	a = np.zeros( (n+1, n+1) )
	for i in xrange(n+1):
		for j in xrange(n+1):
			if i == j:
				a[i][j] = pts[i] / (1 - pts[i]**2)
			else:
				a[i][j] = DL(pts[i]) / ( (pts[i] - pts[j] )*( DL( pts[j] ) ) )
	return a

def cheb_diff_matrix(n, pts):
	# first derivative of P_{N+1}
	DT = np.polynomial.chebyshev.Chebyshev.basis(n+1).deriv(1)
	a = np.zeros( (n+1, n+1) )
	for i in xrange(n+1):
		for j in xrange(n+1):
			if i == j:
				a[i][j] = pts[i] / (2*(1 - pts[i]**2))
			else:
				a[i][j] = DT(pts[i]) / ( (pts[i] - pts[j] )*(DT(pts[j]) ) )
	return a

def print_header(section):
	print
	print section
	print " grid  discrete                 continuous                                         "
	print "  pts  L2 norm        order      L2 norm        order     L-inf norm      order    "
	print "----- -----------------------  ------------------------  ------------------------  "

def compute_orders(errors, j):
	num = len(errors[0])
	orders = np.zeros(num)
	if j > 0:
		for i in xrange(num):
			if abs(errors[j][i]) > 10**(-18):
				orders[i] = np.log2( errors[j-1][i]/errors[j][i] )
			else:
				orders[i] = 0
	return orders

def print_errors(N, errors, orders):
	display_data = np.hstack( zip(errors, orders) )
	print "{:4d}   {:10e}   {:8.5f}   {:10e}   {:8.5f}   {:10e}   {:8.5f}".format( N, *display_data )

# min and max values on the axes
xmin = -1.0
xmax = 1.0

# specifics for data
gridsizes = [ 4, 8, 16, 32, 64 ]
errors = np.zeros( (len(gridsizes), 3 ) )
error_grid = np.linspace(-1, 1, 1000)
L2errors = np.zeros( (4, len(gridsizes) - 1 ) )
Linferrors = np.zeros( (4, len(gridsizes) - 1) )

# column headers for table

# Legendre

# function
print_header("Legendre, ||u - Inu||")
for j, gridsize in enumerate(gridsizes):
	pts, wts = np.polynomial.legendre.leggauss(gridsize+1)
	leg_coeffs = [ legendre_coeff( f(pts), n, pts, wts) for n in xrange(gridsize + 1) ]

	diff = lambda x: np.polynomial.legendre.legval(x, leg_coeffs) - f(x)
	abs_errors = np.abs( diff(error_grid) )
	errors[j] = [ disc_norm(diff, pts, wts), L2norm(diff, -1, 1), max(abs_errors) ]
	print_errors(gridsize, errors[j], compute_orders(errors, j))
L2errors[0]   = np.transpose(errors)[1][:-1]
Linferrors[0] = np.transpose(errors)[2][:-1]

# derivative
print_header("Legendre, ||u' - (Inu)'||")
for j, gridsize in enumerate(gridsizes):
	pts, wts = np.polynomial.legendre.leggauss(gridsize+1)
	D = legendre_diff_matrix(gridsize, pts)
	d_coeffs = [ legendre_coeff( D.dot(f(pts)), n, pts, wts) for n in xrange(gridsize + 1) ]

	diff = lambda x: np.polynomial.legendre.legval(x, d_coeffs) - df(x)
	abs_errors = np.abs( diff(error_grid) )
	errors[j] = [ disc_norm(diff, pts, wts), L2norm(diff, -1, 1), max(abs_errors) ]
	print_errors(gridsize, errors[j], compute_orders(errors, j))
L2errors[1]   = np.transpose(errors)[1][:-1]
Linferrors[1] = np.transpose(errors)[2][:-1]

# Chebyshev

# function
print_header("Chebyshev, ||u - Inu||")
for j, gridsize in enumerate(gridsizes):
	pts, wts = np.polynomial.chebyshev.chebgauss(gridsize+1)
	cheb_coeffs = [ cheb_coeff( f(pts), n, gridsize) for n in xrange(gridsize + 1) ]
	diff = lambda x: np.polynomial.chebyshev.chebval(x, cheb_coeffs) - f(x)
	abs_errors = np.abs( diff(error_grid) )
	errors[j] = [ disc_norm(diff, pts, wts), L2norm(diff, -1, 1), max(abs_errors) ]
	print_errors(gridsize, errors[j], compute_orders(errors, j))
L2errors[2]   = np.transpose(errors)[1][:-1]
Linferrors[2] = np.transpose(errors)[2][:-1]


# derivative
print_header("Chebyshev, ||u' - (Inu)'||")
for j, gridsize in enumerate(gridsizes):
	pts, wts = np.polynomial.chebyshev.chebgauss(gridsize+1)
	D = cheb_diff_matrix(gridsize, pts)
	d_coeffs = [ cheb_coeff( D.dot(f(pts)), n, gridsize) for n in xrange(gridsize + 1) ]

	diff = lambda x: np.polynomial.chebyshev.chebval(x, d_coeffs) - df(x)
	abs_errors = np.abs( diff(error_grid) )
	errors[j] = [ disc_norm(diff, pts, wts), L2norm(diff, -1, 1), max(abs_errors) ]
	print_errors(gridsize, errors[j], compute_orders(errors, j))
L2errors[3]   = np.transpose(errors)[1][:-1]
Linferrors[3] = np.transpose(errors)[2][:-1]


	# fig = plt.figure()
	# scat = fig.add_subplot(111)
	# marker_size = 4
	# colors = ['red', 'blue', 'green', 'orange']
	# markers = ['+', '*', 'o', 's']

	# plot1 = scat.scatter( error_grid, f(error_grid), color=colors[0], s=marker_size )
	# plot2 = scat.scatter( error_grid, np.polynomial.chebyshev.chebval(error_grid, cheb_coeffs), color=colors[1], s=marker_size )


	# plt.show()

# set up plot
fig = plt.figure()
scat = fig.add_subplot(111)
marker_size = 25
colors = ['red', 'blue', 'green', 'orange']
markers = ['+', '*', 'o', 's']
labels = [ "Legendre, ||u - Inu||     ", "Legendre, ||u' - (Inu)'|| ", "Chebyshev, ||u - Inu||    ", "Chebyshev, ||u' - (Inu)'||" ]
print

# # error plot 1: log of L2 error vs grid size
# logL2errors = np.log( L2errors )

# for j in xrange(4):
# 	plot1 = scat.scatter( gridsizes[:-1], logL2errors[j], color=colors[j], label=labels[j], s=marker_size, marker=markers[j])
# 	poly = np.polyfit( gridsizes[:-1], logL2errors[j], 1)
# 	print labels[j], "   slope of line: ", poly[0]
# 	plt.plot( gridsizes[:-1], np.poly1d( poly )(gridsizes[:-1]), color=colors[j] )
# plt.xlabel('grid size (N)')
# plt.ylabel('log L2 error')
# plt.title('Log of continuous L2 error vs grid size')
# plt.legend()
# plt.show()

# error plot 1: log of Linfinity error vs grid size
logLinferrors = np.log( Linferrors )

for j in xrange(4):
	plot1 = scat.scatter( gridsizes[:-1], logLinferrors[j], color=colors[j], label=labels[j], s=marker_size, marker=markers[j])
	poly = np.polyfit( gridsizes[:-1], logLinferrors[j], 1)
	print labels[j], "   slope of line: ", poly[0]
	plt.plot( gridsizes[:-1], np.poly1d( poly )(gridsizes[:-1]), color=colors[j] )
plt.xlabel('grid size (N)')
plt.ylabel('log L infinity error')
plt.title('Log of L infinity error vs grid size')
plt.legend()
plt.show()




