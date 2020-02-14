import numpy as np
import random as rnd
import random
import time as tm
from math import sqrt


# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE SUBMIT.PY
# DO NOT INCLUDE PACKAGES LIKE SKLEARN, SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES FOR WHATEVER REASON WILL RESULT IN A STRAIGHT ZERO
# THIS IS BECAUSE THESE PACKAGES CONTAIN SOLVERS WHICH MAKE THIS ASSIGNMENT TRIVIAL

# DO NOT CHANGE THE NAME OF THE METHOD "solver" BELOW. THIS ACTS AS THE MAIN METHOD AND
# WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THIS NAME WILL CAUSE EVALUATION FAILURES

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

def getObjValue( X, y, wHat):
	lassoLoss = np.linalg.norm( wHat, 1 ) + pow( np.linalg.norm( X.dot( wHat ) - y, 2 ), 2 )
	return lassoLoss

def grad_desc(x,y,w):
    return np.sign(w) + 2*x.T.dot(-y + x.dot(w))

def coord_desc(x,y,w):
	idx = np.random.randint(0,w.shape[0])
	v = np.zeros(w.shape[0])
	v[idx]=1
	return v*(np.sign(w) + 2*x.T.dot(-y + x.dot(w)))

def stoch_desc(X,y,w,batch_size):
	idx = random.sample(range(0,800),batch_size)
	Xb = np.array([X[i] for i in idx])
	yb = np.array([y[i] for i in idx])
	return grad_desc(Xb,yb,w)

def prox_desc(x,y,w):
    return 2*x.T.dot(-y + x.dot(w))

def prox(u):
    d = u.shape[0]
    l = [0]*d
    for i in range(d):
        if u[i]-1 > 0:
            l[i] = u[i] - 1
        elif 1+u[i]<0:
            l[i]= u[i] + 1
    return np.array(l)   

################################
# Non Editable Region Starting #
################################
def solver( X, y, timeout, spacing, c ):
	(n, d) = X.shape
	t = 0
	totTime = 0
	
	# w is the model vector and will get returned once timeout happens
	w = np.zeros( (d,) )
	tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################
	v = np.zeros( (d,) )
	s = np.zeros( (d,) )
	# w = np.random.rand(d)

	# You may reinitialize w to your liking here
	# You may also define new variables here e.g. step_length, mini-batch size etc

################################
# Non Editable Region Starting #
################################
	while True:
		t = t + 1
		if t % spacing == 0:
			toc = tm.perf_counter()
			totTime = totTime + (toc - tic)
			if totTime > timeout:
				return (w, totTime)
			else:
				tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################

		#Polyak step size for faster convergence
		# loss_pichla = getObjValue(X,y,w)
		# step = (loss_pichla - 60) / (np.linalg.norm(grad_desc(X,y,w),2)**2)
		
		#Nesterov's Momentum

		# v = 0.99*v + 0.01*grad_desc(X,y,w-0.99*v)
		# w = w - (c/sqrt(t))*v
		# c = 2 worked well

		#Adam
		# grad = coord_desc(X,y,w)
		# for i in range(w.shape[0]):
		# 	v[i] = 0.9*v[i] - 0.1*(grad[i])
		# 	s[i] = 0.999*s[i] - 0.001*(grad[i]**2)
		# 	del_w = -(0.05 * v[i] / sqrt(abs(s[i]) + 1e-8))*grad[i]
		# 	w[i] = w[i] + del_w

		# #Uncomment for Stochastic Minibatch GD
		# w = w - c/(sqrt(t))*stoch_desc(X,y,w,256)

		# #Uncomment for Coordinate Descent
		# w = w - (c/sqrt(t))*coord_desc(X,y,w)

		# #Uncomment for Proximal GD
		# u = w - prox_desc(w)
		# w = prox(u)

		#Vanilla GD
		w = w - (c/sqrt(t))*grad_desc(X,y,w)
		# c = 0.05 worked quite good

		# #Uncomment to see the convergence trail
		# print(getObjValue(X,y,w))

		# Write all code to perform your method updates here within the infinite while loop
		# The infinite loop will terminate once timeout is reached
		# Do not try to bypass the timer check e.g. by using continue
		# It is very easy for us to detect such bypasses which will be strictly penalized
		
		# Please note that once timeout is reached, the code will simply return w
		# Thus, if you wish to return the average model (as is sometimes done for GD),
		# you need to make sure that w stores the average at all times
		# One way to do so is to define a "running" variable w_run
		# Make all GD updates to w_run e.g. w_run = w_run - step * delw
		# Then use a running average formula to update w
		# w = (w * (t-1) + w_run)/t
		# This way, w will always store the average and can be returned at any time
		# In this scheme, w plays the role of the "cumulative" variable in the course module optLib
		# w_run on the other hand, plays the role of the "theta" variable in the course module optLib
		
	return (w, totTime) # This return statement will never be reached

# Z = np.loadtxt( "train" )
# wAst = np.loadtxt( "wAstTrain" )
# k = 20

# y = Z[:,0]
# X = Z[:,1:]

# (w, totTime) = solver( X, y, 5, 10 )
# # print(sorted(abs(w)))
# w2 =  np.zeros(w.shape[0])
# ff = np.argsort(abs(w))[::-1][:20]
# for i in range(len(ff)):
# 	w2[ff[i]]=w[ff[i]]
# # c=0
# # for i in range(w.shape[0]):
# # 	if(w2[i]==0 and wAst[i]==0):
# # 		c=c+1
# # 	else:
# # 		print(w2[i],wAst[i])

# print("printing new loss")
# print(f(X,y,w2))
# print(f(X,y,wAst))

# # print(ff)
# # print(sorted(abs(wAst)))