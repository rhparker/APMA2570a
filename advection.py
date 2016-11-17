#
# advection equation
#

# basic math functions from numpy
import numpy as np

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

# difference operators for finite difference methods

# forward difference operator D+ = (E+ - I)/h
def dplus(n, stepsize):
	return ( eplus(n) - np.identity(n) ) / stepsize

# backward difference operator D+ = (I - E-)/h
def dminus(n, stepsize):
	return ( np.identity(n) - eminus(n) ) / stepsize

# central difference operator (second order) is D0 = (E+ - E-) / 2h
# order is number of spacial steps to go on each side (default is 1)
def dzero(n, stepsize, order = 1):
	return ( np.linalg.matrix_power(eplus(n), order) - np.linalg.matrix_power(eminus(n), order) ) / (2 * stepsize * order)

# fourth order central difference operator
def fourth_central_diff(n, stepsize):
	return ( (4.0/3)*dzero(n, stepsize, 1) - (1.0/3)*dzero(n, stepsize, 2) )

# fourth order central difference operator
def sixth_central_diff(n, stepsize):
	return ( (3.0/2)*dzero(n, stepsize, 1) - (12.0/20)*dzero(n, stepsize, 2) + (1.0/10)*dzero(n, stepsize, 3) )

# operator for spectral collocation method
def spectral_colloc(n):
	a = np.zeros( (n+1, n+1) )
	for i in xrange(n+1):
		for j in xrange(n+1):
			if i != j:
				a[j][i] = (-1.0)**(j+i) / ( 2.0 * np.sin( (j - i) * np.pi / (n + 1) ) )
	return a

# time stepping schemes

# performs one iteration of third-order Runge-Kutta scheme
# n is size of grid, k is time step, v is advection velocity, x is current discrete solution vector
# m is matrix for discrete spatial operator
# returns the new discrete solution vector
def runge_kutta_3(n, k, v, x, m):
	k1 = v * m.dot( x )
	k2 = v * m.dot( x + 0.5 * k * k1 )
	k3 = v * m.dot( x + 0.75 * k * k2 )
	return x + (1.0/9) * k * (2 * k1 + 3 * k2 + 4 * k3)

# # performs one iteration of fourth-order Runge-Kutta scheme
# # n is size of grid, k is time step, v is advection velocity, x is current discrete solution vector
# # m is matrix for discrete spatial operator
# # returns the new discrete solution vector
# def runge_kutta_4(n, k, v, x, m):
# 	k1 = v * m.dot( x )
# 	k2 = v * m.dot( x + 1.0/3 * k * k1 )
# 	k3 = v * m.dot( x - 1.0/3 * k * k1 + k * k2 )
# 	k4 = v * m.dot( x + k * k1 - k * k2 + k * k3)
# 	return x + (1.0/8) * k * (k1 + 3 * k2 + 3 * k3 + k4)

# performs one iteration of fourth-order Runge-Kutta scheme
# n is size of grid, k is time step, v is advection velocity, u is current discrete solution vector
# f function for spatial operator
# returns the new discrete solution vector
def runge_kutta_4(n, k, v, u, m):
	k1 = v * k * m.dot( u )
	k2 = v * k * m.dot( u + 0.5 * k1 )
	k3 = v * k * m.dot( u + 0.5 * k2 )
	k4 = v * k * m.dot( u + k )
	return u + (1.0 / 6) * (k1 + 2*k2 + 2*k3 + k4)

# performs one iteration of sixth-order Runge-Kutta scheme
# n is size of grid, k is time step, v is advection velocity, x is current discrete solution vector
# m is matrix for discrete spatial operator
# returns the new discrete solution vector
def runge_kutta_6(n, k, v, x, m):
	k1 = v * m.dot( x )
	k2 = v * m.dot( x + (1.0/4) * k1 * k)
	k3 = v * m.dot( x + (3.0/32) * k * (k1 + 3.0 * k2))
	k4 = v * m.dot( x + (12.0/2197) *k*(161.0*k1-600.0*k2+608.0*k3))
	k5 = v * m.dot( x + (1.0/4104) *k*(8341.0*k1-32832.0*k2+29440.0*k3-845.0*k4))
	k6 = v * m.dot( x + k * (-(8.0/27)*k1+2*k2-(3544.0/2565)*k3+(1859.0/4104)*k4-(11.0/40)*k5))
	return x + (1.0/5)*((16.0/27)*k1+(6656.0/2565)*k3+(28561.0/11286)*k4-(9.0/10)*k5+(2.0/11)*k6)*k


# # this performs one iteration of the difference scheme
# # the vector v contains the current state; matrix m is the difference operator
# def iterate_single(m, v):
# 	return m.dot(v)

# function to use for initial conditions
def f(x):
	return np.sin(np.cos(x))

# exact solution
# v is advection velocity
def exact(t, x, v):
	return f(x + v * t)

# compute discrete L2 norm of V; h is spatial step
def L2norm(v, h):
	return np.sqrt( v.dot(v) * h )

# # parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gridsize', help='number of points in spatial grid', type=int, default=100)
parser.add_argument('--noshow', help='do not display graph or animation', action="store_true")
parser.add_argument('--skip', help='number of steps to skip when drawing animation (default 2)', type=int, default=2)
parser.add_argument('--delay', help='delay between animation frames in ms (default 5)', type=int, default=5)
parser.add_argument('--save', help='save figure every [SAVE] frames', type=int)

args = parser.parse_args()

# min and max values on the axes
xmin = 0.
xmax = 2. * np.pi

# end time is 20 pi
endtime = 20. * np.pi

# size of grid to use
gridsize = args.gridsize + 1

# calculate spatial step size from grid size
h = (xmax - xmin) / gridsize

# time step size
k = 0.01

# number of steps to take
steps = int( endtime / k )

# construct the grid of points, and remove the final point, 
# since it is equated with the initial point
xgrid = np.linspace(xmin, xmax, gridsize + 1)
xgrid = xgrid[:-1]

# use fourth central difference for spatial method
spatial_methods = [ dzero(gridsize, h), fourth_central_diff(gridsize, h), sixth_central_diff(gridsize, h), spectral_colloc(gridsize-1) ]

# initialize u values using our function f
u = [ f(xgrid), f(xgrid), f(xgrid), f(xgrid) ]

# used for plotting
colors = ['red', 'blue', 'green', 'orange']
markers = ['+', '*', 'o', 's']
labels = ['2nd order CD', '4th order CD', '6th order CD', 'Spectral Collocation']

# set up plot
fig = plt.figure()
scat = fig.add_subplot(111)
marker_size = 6
skip = args.skip
delay = args.delay

# iterates the scheme one time
def iterate():
	global u
	for i in xrange( len(u) ):
		u[i] = runge_kutta_4(gridsize, k, -1.0, u[i], spatial_methods[i] )

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
	scat.set_ylim(-2, 2)
	scat.set_title("Number of steps: "+str(step*skip))

	# approximate solution
	for i in xrange( len(u) ):
		plot1 = scat.scatter(xgrid, u[i], color=colors[i], s=marker_size)
		scat.plot(xgrid, u[i], color=colors[i])

	# exact solution 
	exact_sol = [ exact( k * (step*skip), i, -1 ) for i in xgrid ]
		
	plot2 = scat.scatter(xgrid, exact_sol, color='black', s=marker_size)
	scat.plot(xgrid, exact_sol, color='black')

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
	gridsizes = [8, 16, 32, 64, 128, 256 ]
	L2errors = np.zeros( (len(gridsizes), 4 ) )
	orders = np.zeros( len(u) )

	# column headers for table
	print " grid      2nd order CD              4th order CD              6th order CD            Spectral Collocation"
	print "  pts  L2 error       order      L2 error       order      L2 error       order      L2 error       order"
	print "----- -----------------------  ------------------------  ------------------------  ---------------------------"

	for j in xrange(len(gridsizes)):
		# calculate spatial step size from grid size
		gridsize = gridsizes[j] + 1
		h = (xmax - xmin) / gridsize

		# time step size
		k = 0.005

		# number of steps to take
		steps = int( endtime / k )

		# construct the grid of points, and remove the final point,
		# since it is equated with the initial point
		xgrid = np.linspace(xmin, xmax, gridsize+1)
		xgrid = xgrid[:-1]

		# use fourth central difference for spatial method
		spatial_methods = [ dzero(gridsize, h), fourth_central_diff(gridsize, h), sixth_central_diff(gridsize, h), spectral_colloc(gridsize-1) ]

		# initialize u values using our function f
		u = [ f(xgrid), f(xgrid), f(xgrid), f(xgrid) ]

		# iterate scheme a specific number of steps
		for i in xrange(steps):
			iterate()
		
		# compute exact solutions
		exact_sol = [ exact(k * steps, i, -1 ) for i in xgrid ]

		# compute error and L2 error
		for i in xrange(len(u)):
			L2errors[j][i] = L2norm( u[i] - exact_sol , h)

		# compute orders
		if j > 0:
			for i in xrange(len(u)):
				orders[i] = np.log2( L2errors[j-1][i]/L2errors[j][i] )
 
		display_data = np.hstack( zip(L2errors[j], orders) )

		print "{:4d}   {:10e}   {:5f}   {:10e}   {:5f}   {:10e}   {:5f}   {:10e}   {:5f}".format( gridsize-1, *display_data )

	# error plot: log of L2 error vs grid size
	logL2errors = np.transpose( np.log( L2errors ) )

	marker_size = 16
	for j in xrange(len(u)):
		plot1 = scat.scatter(gridsizes, logL2errors[j], color=colors[j], label=labels[j], s=marker_size, marker=markers[j])
		poly = np.polyfit(gridsizes[2:], logL2errors[j][2:], 1)
		plt.plot( gridsizes, np.poly1d( poly )(gridsizes), color=colors[j] )
	plt.legend()
	plt.show()



# otherwise display the animation
# use this to see graphically that scheme is working
else:
	# creates the animation routine; function "update" is called for each update
	ani = animation.FuncAnimation(fig, update, init_func=init, interval=delay, frames=steps/skip, blit=False)

	# displays the plot and runs the animation
	plt.show()
