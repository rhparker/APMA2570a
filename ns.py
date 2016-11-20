#
# navier-stokes, fourier collocation
#

# basic math functions from numpy
import numpy as np

# scipy integration and discrete convolution
from scipy.integrate import quad, dblquad
from scipy.signal import convolve2d

# pyplot plot class plot classes from matplotlib
import matplotlib.pyplot as plt

# animation class from matplotlib
import matplotlib.animation as animation

# argparse to parse command line arguments
import argparse

# Lagrange trigonometric polynomial (even number of grid points)
def lagrange_even(j, n, h, x):
	xj = h * j
	t = 0.5 * (x - xj)
	if t == 0:
		return 1.0
	else:
		return (1.0/n) * np.sin(n * t) / np.tan(t)

# Lagrange trigonometric polynomial (odd number of grid points)
def lagrange(j, n, h, x):
	xj = h * j
	t = 0.5 * (x - xj)
	if t == 0:
		return 1.0
	else:
		return (1.0/(n + 1)) * np.sin( (n+1) * t) / np.sin(t)

def lagrange_interp(n, h, u, x):
	return np.sum( [ u[j] * lagrange(j, n, h, x) for j in xrange(n+1) ] )

# operator for spectral collocation method
def spectral_colloc(n):
	a = np.zeros( (n+1, n+1) )
	for i in xrange(n+1):
		for j in xrange(n+1):
			if i != j:
				a[j][i] = (-1.0)**(j+i) / ( 2.0 * np.sin( (j - i) * np.pi / (n + 1) ) )
	return a

# time stepping schemes

# performs one iteration of fourth-order Runge-Kutta scheme
# n is size of grid, k is time step, u is current discrete solution vector
# f function for spatial operator
# returns the new discrete solution vector
def runge_kutta_4(n, k, u, f):
	k1 = k * f( u )
	k2 = k * f( u + 0.5 * k1 )
	k3 = k * f( u + 0.5 * k2 )
	k4 = k * f( u + k3 )
	return u + (1.0 / 6) * (k1 + 2*k2 + 2*k3 + k4)


# # this performs one iteration of the difference scheme
# # the vector v contains the current state; matrix m is the difference operator
# def iterate_single(m, v):
# 	return m.dot(v)

# functions to use for initial conditions
def f(x, y):
	return -2 * np.sin(x) * np.sin(y)

def f2(x, y, rho, delta):
	if y <= np.pi:
		return delta * np.cos(x) - 1./(rho * (np.cosh((y - np.pi/2)/rho )**2) ) 
	else:
		return delta * np.cos(x) + 1./(rho * (np.cosh((3 * np.pi/2 - y)/rho )**2) ) 

def exact_1(x, y, t):
	return -2 * np.sin(x) * np.sin(y) * np.exp( - 2. * t / 100. )

# u is function of x and y, N is even
def Dy(u, N, degree = 1):
	k = np.array( range(0, N/2 + 1) + range(-N/2, 0) )
	uhat = np.fft.fft2(u)
	for i in xrange(N+1):
		uhat[i] = uhat[i] * ((1.0j * k)**degree)
	return np.real( np.fft.ifft2(uhat) )

def Dx(u, N, degree = 1):
	return np.transpose( Dy(np.transpose(u), N, degree) )

# Solve Laplacian u(x, y) = g(x, y)
def poisson_solve(g, N):
	# indices for FFT
	j = np.array( range(0, N/2 + 1) + range(-N/2, 0) )
	k = np.array( range(0, N/2 + 1) + range(-N/2, 0) )
	# denominator for Poisson solution in Fourier space
	denom_fn = lambda x, y: x**2 + y**2
	denom = denom_fn(j[:,None], k[None,:])
	# don't care about the (0, 0) coefficient, since it's the constant of integration
	denom[0][0] = 1
	ghat = np.fft.fft2(g)
	uhat = -ghat/denom
	uhat[0][0] = 0
	return np.fft.ifft2(uhat)

# compute velocites from vorticity
def vel(w, N):
	# solve Poisson equation to get potential
	psi = poisson_solve(w, N)
	# velocities are derivatives of potential
	return np.real(Dy(psi, N)), -np.real(Dx(psi, N))

def spatial_method_1(w, N):
	u, v = vel(w, N)
	return -Dx(u*w, N) - Dy(v*w, N) + 1./100 * ( Dx(w, N, 2) + Dy(w, N, 2) )

def spatial_method_2(w, u, v, N):
	return -Dx(u*w, N) - Dy(v*w, N) + 1./100 * ( Dx(w, N, 2) + Dy(w, N, 2) )

def update_coeffs(b, sq, inv_sq, ij, ik):
	a = -b * inv_sq
	u = ik * a
	v = -ij * a
	return -0.01 * sq * b - convolve2d(u, ij*b, mode="same") - convolve2d(v, ik*b, mode="same") 

def helper_matrices(N):
	coeffs = range(-N/2, N/2 + 1)
	squares = np.array( [[ j**2 + k**2 for k in coeffs ] for j in coeffs ] )
	inv_squares = 1.0 / squares
	inv_squares[N/2][N/2] = 0
	ik = np.array( [[ 1.j * n for n in coeffs ] for n in coeffs ] )
	ij = np.transpose(ik)
	return squares, inv_squares, ij, ik

# compute 2D discrete L2 norm of V; h is spatial step
def discrete_L2norm(v, h):
	values = v.flatten()
	return np.sqrt( values.dot(values) * (h**2) )

def trig_poly(x, y, a, N):	
	j = np.array( range(0, N/2 + 1) + range(-N/2, 0) )
	k = np.array( range(0, N/2 + 1) + range(-N/2, 0) )
	values = np.real( np.array( [ [ a[jindex][kindex] * np.exp(1.0j * jval * x) * np.exp(1.0j * kval * y) 
			for kindex, kval in enumerate(k) ] for jindex, jval in enumerate(j) ] ) )
	return np.sum( values ) / ((N+1)**2)

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gridsize', help='number of points in spatial grid', type=int, default=4)
parser.add_argument('--endtime', help='end time for simulation', type=float, default=np.pi * 2)
parser.add_argument('--show', help='display graph or animation', action="store_true")
parser.add_argument('--second', help='use second initial condition', action="store_true")
parser.add_argument('--galerkin', help='use Fourier Galerkin method', action="store_true")
parser.add_argument('--skip', help='number of steps to skip when drawing animation (default 2)', type=int, default=8)
parser.add_argument('--delay', help='delay between animation frames in ms (default 5)', type=int, default=5)
parser.add_argument('--save', help='save figure every [SAVE] frames', type=int)


args = parser.parse_args()

# min and max values on the axes
xmin = 0.
xmax = 2. * np.pi

# end time
endtime = args.endtime

# size of grid to use
gridsize = args.gridsize

# calculate spatial step size from grid size
h = (xmax - xmin) / (gridsize + 1)

# cfl number
cfl = 0.02

# time step size
# k = 0.1 
k = cfl * h

# number of steps to take
steps = int( endtime / k )

# construct the grid of points, and remove the final point, 
# since it is equated with the initial point
x = np.linspace(xmin, xmax, gridsize + 2)
x = x[:-1]
y = np.linspace(xmin, xmax, gridsize + 2)
y = y[:-1]
X, Y = np.meshgrid(x, y)
display_grid = np.linspace(xmin, xmax, 100)

# used for plotting
colors = ['red', 'blue', 'green', 'orange']
markers = ['+', '*', 'o', 's']
labels = ['Fourier collocation', 'Fourier Galerkin', 'exact solution']

# set up plot
fig = plt.figure()
scat = fig.add_subplot(111)
marker_size = 6
skip = args.skip
delay = args.delay

# initial condition
if args.second:
	w = np.array([[f2(i, j, np.pi/15, 0.05) for j in y] for i in x])
else:
	w = f(x[:,None], y[None,:])

# FFT of w, shift zero component into center
f_w = np.fft.fftshift( np.fft.fft2(w) )

sq, inv_sq, ij, ik = helper_matrices(gridsize)

# iterates the scheme one time
def iterate():
	global w, f_w
	w = runge_kutta_4(gridsize, k, w, lambda u: spatial_method_1(u, gridsize) )
	f_w = runge_kutta_4(gridsize, k, f_w, lambda u: update_coeffs(u, sq, inv_sq, ij, ik) )

# initialization function
# for now, does nothing, but keeps update from being called an extra time
def init():
	pass

# updates the finite difference schemes as well as the plot
def update(step):
	global w, f_w

	# clear old plot data
	# scatterplot
	scat.cla()
	scat.set_xlim(xmin, xmax)
	scat.set_ylim(-2, 2)
	scat.set_title("Time:  "+str(step*skip*k))

	# approximate solution: Fourier collocation
	plot1 = scat.scatter(y, w[4], color=colors[0], s=marker_size * 2)
	a = np.fft.fft2(w)
	y1 = [ trig_poly(x[4], i, a, gridsize) for i in display_grid ]
	scat.scatter(display_grid, y1, color=colors[0], s=marker_size)

	# approximate solution: Fourier Galerkin
	y2 = [ trig_poly(x[4], i, np.fft.ifftshift(f_w), gridsize) for i in display_grid ]
	scat.scatter(display_grid, y2, color=colors[1], s=marker_size)

	# scat.plot(xgrid, u[0], color=colors[0])

	# # exact solution
	if not args.second:
		plot3 = scat.scatter(y, exact_1(x[4], y, step*skip*k), color="black", s=marker_size)

	# iterate the scheme a number of steps specified by "skip"
	for i in xrange(skip):
		iterate()

	# save plot to PNG file, if requested, every args.save frames
	if args.save:
		if step % args.save == 0:
			plt.savefig("frame"+str(step)+".png")

	return True

if args.show:
	# creates the animation routine; function "update" is called for each update
	ani = animation.FuncAnimation(fig, update, init_func=init, interval=delay, blit=False)

	# displays the plot and runs the animation
	plt.show()

else:
	if args.second:
		gridsizes = [ 32 ]
	else:
		gridsizes = [ 2, 4, 8, 16, 32 ]

	errors = np.zeros( ( len(gridsizes), 3 ) )
	orders = np.zeros( 3 )

	if not args.second:
		# column headers for table
		if args.galerkin:
			print "Fourier-Galerkin"
			print
		else:
			print "Fourier-collocation"
			print

		print " grid      discrete L2                continuous L2            L-infinity         "
		print "  pts   error          order       error         order      error          order     "
		print "----- ------------------------  ------------------------  ------------------------ "

	endtime = np.pi * 2
	# endtime = 1
	cfl = 0.1

	for n, N in enumerate(gridsizes):
		# compute step size needed from CFL number
		h = (xmax - xmin) / (N + 1)
		k = cfl * h

		# number of steps to take
		steps = int( endtime / k )

		# create grids for Fourier collocation
		x = np.linspace(xmin, xmax, N + 2)
		x = x[:-1]
		y = np.linspace(xmin, xmax, N + 2)
		y = y[:-1]

		# grids for display and L-infinity error
		lx = np.linspace(xmin, xmax, 100)
		ly = np.linspace(xmin, xmax, 100)

		X, Y = np.meshgrid(x, y)
		LX, LY = np.meshgrid(lx, ly)

		# initial condition
		if args.second:
			w = np.array([[f2(i, j, np.pi/15, 0.05) for j in y] for i in x])
		else:
			w = f(x[:,None], y[None,:])

		if args.galerkin:
			# helper matrices for Fourier Galerkin
			sq, inv_sq, ij, ik = helper_matrices(N)

			# FFT of initial condition
			f_w = np.fft.fftshift( np.fft.fft2(w) ) / ((N+1)**2)

		for i in xrange(steps):
			# Fourier Galerkin
			if args.galerkin:
				f_w = runge_kutta_4(gridsize, k, f_w, lambda u: update_coeffs(u, sq, inv_sq, ij, ik) )

			# Fourier collocation
			else:
				# # this version calculates u and v separately for each R-K substep
				# w = runge_kutta_4(gridsize, k, w, lambda z: spatial_method_1(z, N) )

				# this version calculates u and v once for all R-K steps (faster)
				u, v = vel(w, N)
				w = runge_kutta_4(gridsize, k, w, lambda z: spatial_method_2(z, u, v, N) )

		# for Galerkin method, need to convert back to physical space
		if args.galerkin:
			w = np.real( np.fft.ifft2( np.fft.ifftshift( ((N+1)**2) * f_w) ) )

		# compute error in case where exact solution is known
		if not args.second:
			# discrete L2 error
			exact_sol = exact_1(x[:,None], y[None,:], steps * k)
			disc_L2_error = discrete_L2norm(w - exact_sol, h)

			# L-infinty error
			# perform interpolation with discrete Fourier transform (FFT)
			f_coeffs = np.fft.fft2(w)
			interp = np.array( [[ trig_poly(i, j, f_coeffs, N) for j in ly] for i in lx])
			exact_sol_inf = exact_1(lx[:,None], ly[None,:], steps * k)
			Linf_error = np.max( exact_sol_inf - interp )

			# continuous L2 error
			integrand = lambda x, y: ( exact_1(x, y, steps*k) - trig_poly(x, y, f_coeffs, N) )**2
			L2_error = np.sqrt( dblquad(integrand, xmin, xmax, lambda x: xmin, lambda x: xmax) )[0]

			errors[n] = [disc_L2_error, L2_error, Linf_error]

			# compute orders
			if n > 0:
				for i in range(3):
					orders[i] = np.log2( errors[n-1][i]/errors[n][i] )
 
			display_data = np.hstack( zip( errors[n], orders) )

			print "{:4d}   {:10e}   {:8.5f}   {:10e}   {:8.5f}   {:10e}   {:8.5f}".format( N, *display_data )

			levels = np.linspace(-2, 2, 101)
			plt.contourf(LX, LY, interp, levels=levels)
			scat.set_title("First initial condition, "+str(N)+" gridpoints")

		else:
			levels = np.linspace(-5, 5, 101)
			scat.set_title("Second initial condition, "+str(N)+" gridpoints")
			plt.contourf(X, Y, np.transpose(w), levels=levels)
		
		plt.colorbar()
		plt.show()


