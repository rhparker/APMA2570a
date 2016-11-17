#
# advection equation
#

# basic math functions from numpy
import numpy as np

from scipy.integrate import quad, simps, trapz, romberg

# pyplot plot class plot classes from matplotlib
import matplotlib.pyplot as plt

# animation class from matplotlib
import matplotlib.animation as animation

# argparse to parse command line arguments
import argparse

# shift operators for finite difference methods

# eplus is right shift operator; 1s above main diagonal, zeros everywhere else
# since we have periodic BCs, we have to "wrap around".
# since the two endpoints are equated, we will remove the right endpoint once we make the grid
def eplus(n):
	a = np.zeros( (n, n) )
	i,j = np.indices(a.shape)
	a[ i == j-1 ] = 1
	a[n-1, 0] = 1
	return a

# eminus is left shift operator; 1s below main diagonal, zeros everywhere else
# since we have periodic BCs, we have to "wrap around". 
# since the two endpoints are equated, we will remove the right endpoint once we make the grid
def eminus(n):
	a = np.zeros( (n, n) )
	i,j = np.indices(a.shape)
	a[ i == j+1 ] = 1
	a[0, n-1] = 1
	return a

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

# the two different spatial operators
def spatial_1(u, cosx, m):
	return -cosx * m.dot(u)

def spatial_2(u, cosx, sinx, m):
	return -0.5 * cosx * m.dot(u) - 0.5 * m.dot( cosx * u ) - 0.5 * sinx * u 

# # this performs one iteration of the difference scheme
# # the vector v contains the current state; matrix m is the difference operator
# def iterate_single(m, v):
# 	return m.dot(v)

# functions to use for initial conditions
def f(x):
	return np.sin(x)

def g(x):
	return x - np.pi

# exact solution
def exact_sol(t, x, initial, sawtooth):
	if sawtooth:
		return 2 * np.arctan( np.tanh( 0.5 * ( (t - 2 * ( np.arctanh( np.tan( (np.pi - x)/2) + 0j ) ) ))))
	else:
		return initial( 2 * np.arctan( np.tanh( 0.5 * ( (-t + 2 * np.real( np.arctanh( np.tan(x/2) + 0j ) ) )))))

# compute discrete L2 norm of V; h is spatial step
def discrete_L2norm(v, h):
	return np.sqrt( v.dot(v) * h )

# L2 norm on [0, 2 pi]
def L2norm(f):
	integrand = lambda x : abs(f(x))**2
	return np.sqrt( quad( integrand, 0, 2 * np.pi )[0] )

# L2 norm on [0, 2 pi]
def L2norm_simps(f, points):
	integrand = [ abs(f(x))**2 for x in np.linspace(xmin, xmax, points) ]
	return np.sqrt( simps( integrand, dx = (xmax - xmin) / points) )

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gridsize', help='number of points in spatial grid', type=int, default=100)
parser.add_argument('--endtime', help='end time for simulation', type=float, default=np.pi / 2)
parser.add_argument('--noshow', help='do not display graph or animation', action="store_true")
parser.add_argument('--sawtooth', help='use sawtooth initial condition', action="store_true")
parser.add_argument('--skip', help='number of steps to skip when drawing animation (default 2)', type=int, default=2)
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
cfl = 0.01

# time step size
k = 0.005
# k = cfl * h

# number of steps to take
steps = int( endtime / k )

# construct the grid of points, and remove the final point, 
# since it is equated with the initial point
xgrid = np.linspace(xmin, xmax, gridsize + 2)
xgrid = xgrid[:-1]

# precalculate these to save time
sinxgrid = np.sin(xgrid)
cosxgrid = np.cos(xgrid)

# spatial method
colloc_matrix = spectral_colloc(gridsize)

# initialize u values using our function f
if args.sawtooth:
	initial = g
else:
	initial = f

u = [ initial(xgrid), initial(xgrid) ]

# used for plotting
colors = ['red', 'blue', 'green', 'orange']
markers = ['+', '*', 'o', 's']
labels = ['numerical solution: version 1', 'numerical solution: version 2', 'exact solution']

# set up plot
fig = plt.figure()
scat = fig.add_subplot(111)
marker_size = 6
skip = args.skip
delay = args.delay

# iterates the scheme one time
def iterate():
	global u
	u[0] = runge_kutta_4(gridsize, k, u[0], lambda x: spatial_1(x, cosxgrid, colloc_matrix) )
	u[1] = runge_kutta_4(gridsize, k, u[1], lambda x: spatial_2(x, cosxgrid, sinxgrid, colloc_matrix) )

# initialization function
# for now, does nothing, but keeps update from being called an extra time
def init():
	pass

# updates the finite difference schemes as well as the plot
def update(step):
	global u
	# clear old plot data
	scat.cla()
	scat.set_xlim(xmin, xmax)
	scat.set_ylim(-np.pi, np.pi)
	# scat.set_title("Number of time steps: "+str(step*skip)+"   Time: "+str(step*skip*k))
	scat.set_title("Time:  "+str(step*skip*k))

	# approximate solution
	plot1 = scat.scatter(xgrid, u[0], color=colors[0], s=marker_size, label=labels[0])
	y1 = [ lagrange_interp(gridsize, h, u[0], x) for x in xgrid ]
	scat.plot(xgrid, y1, color=colors[0])
	# scat.plot(xgrid, u[0], color=colors[0])

	plot2 = scat.scatter(xgrid, u[1], color=colors[1], s=marker_size, label=labels[1])
	y2 = [ lagrange_interp(gridsize, h, u[1], x) for x in xgrid ]
	scat.plot(xgrid, y2, color=colors[0])
	# scat.plot(xgrid, u[1], color=colors[1])

	# exact solution
	plot3 = scat.scatter(xgrid, exact_sol(step * skip * k, xgrid, initial, args.sawtooth), color="black", s=marker_size, label=labels[2])
	scat.plot(xgrid, exact_sol(step * skip * k, xgrid, initial, args.sawtooth), color='black')
	plt.legend()

	# iterate the finite difference scheme a number of steps specified by "skip"
	for i in xrange(skip):
		iterate()

	# save plot to PNG file, if requested, every args.save frames
	if args.save:
		if step % args.save == 0:
			plt.savefig("frame"+str(step)+".png")

	return True

# do not display the animation
# use this for error analysis
if args.noshow:
	# size of grid to use
	gridsizes = [ 8, 16, 32, 64, 128 ]
	L2errors = np.zeros( (len(gridsizes), 2 ) )
	orders = np.zeros( len(u) )

	# column headers for table
	print " grid       version 1                  version 2        "
	print "  pts  L2 error       order      L2 error       order   "
	print "----- -----------------------  ------------------------ "

	for j in xrange(len(gridsizes)):
		# calculate spatial step size from grid size
		gridsize = gridsizes[j]

		# calculate spatial step size from grid size
		h = (xmax - xmin) / (gridsize + 1)

		# time step size
		k = h * cfl
		# k = 0.002

		# number of steps to take
		steps = int( endtime / k )

		# construct the grid of points, and remove the final point,
		# since it is equated with the initial point
		xgrid = np.linspace(xmin, xmax, gridsize+2)
		xgrid = xgrid[:-1]

		# precalculate these to save time
		sinxgrid = np.sin(xgrid)
		cosxgrid = np.cos(xgrid)

		# spatial method
		colloc_matrix = spectral_colloc(gridsize)

		# initialize u values using our function f
		u = [ initial(xgrid), initial(xgrid) ]

		# iterate scheme a specific number of steps
		for i in xrange(steps):
			iterate()
		
		# compute exact solutions
		exact = [ exact_sol(k * steps, x, initial, args.sawtooth ) for x in xgrid ]

		# error = lambda x: lagrange_interp(gridsize, xgrid, u[1], x) -  exact_sol(k * steps, x, initial ) 
		# print L2norm_simps( error, 1000 ), L2norm( error ), discrete_L2norm( u[1] - exact, h)

		# compute error and L2 error
		for i in xrange(len(u)):

			# use this for discrete L2 error
			# L2errors[j][i] = discrete_L2norm( u[i] - exact, h)

			# L2 error via integragtion
			error = lambda x: lagrange_interp(gridsize, h, u[i], x) -  exact_sol(k * steps, x, initial, args.sawtooth ) 
			L2errors[j][i] = L2norm( error )

			# # plot of error, for testing purposes
			# error_grid = np.linspace(0, 2 * np.pi, 200)[:-1]
			# exact_pts = [ exact_sol(k * steps, x, initial, args.sawtooth ) for x in error_grid ]
			# approx_pts = [ lagrange_interp(gridsize, h, u[i], x) for x in error_grid ]
			# errors = np.array( [ error(x) for x in error_grid] )
			# plot2 = scat.scatter(error_grid, errors )
			# plt.show()

		# compute orders
		if j > 0:
			for i in xrange(len(u)):
				orders[i] = np.log2( L2errors[j-1][i]/L2errors[j][i] )
 
		display_data = np.hstack( zip(L2errors[j], orders) )

		print "{:4d}   {:10e}   {:8.5f}   {:10e}   {:8.5f}".format( gridsize, *display_data )

	# error plot: log of L2 error vs grid size
	logL2errors = np.transpose( np.log( L2errors ) )

	marker_size = 16
	print

	for j in xrange(len(u)):
		plot1 = scat.scatter(gridsizes, logL2errors[j], color=colors[j], label=labels[j], s=marker_size, marker=markers[j])
		poly = np.polyfit(gridsizes, logL2errors[j], 1)
		print labels[j], "   slope of line: ", poly[0]
		plt.plot( gridsizes, np.poly1d( poly )(gridsizes), color=colors[j] )
	plt.xlabel('grid size (N)')
	plt.ylabel('log L2 error')
	plt.title('Log of L2 error vs grid size')

	plt.legend()
	plt.show()



# otherwise display the animation
# use this to see graphically that scheme is working
else:
	# creates the animation routine; function "update" is called for each update
	ani = animation.FuncAnimation(fig, update, init_func=init, interval=delay, blit=False)

	# displays the plot and runs the animation
	plt.show()
