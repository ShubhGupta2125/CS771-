import numpy as np
from submit import solver

# Find out how much loss is the learnt model incurring?
def getObjValue( X, y, wHat ):
	lassoLoss = np.linalg.norm( wHat, 1 ) + pow( np.linalg.norm( X.dot( wHat ) - y, 2 ), 2 )
	return lassoLoss

# Find out how far is the learnt model from the true one in terms of Euclidean distance
def getModelError( wHat, wAst ):
	return np.linalg.norm( wHat - wAst, 2 )

# Force the learnt model to become sparse and then see how well it approximates the true model
def getSupportError( wHat, wAst, k ):
	# Find the k coordinates where the true model has non-zero values
	idxAst = np.abs( wAst ).argsort()[::-1][:k]
	# Find the k coordinates with largest values (absolute terms) in the learnt model
	idxHat = np.abs( wHat ).argsort()[::-1][:k]
	
	# Set up indicator arrays to find the diff between the two
	# Could have used Python's set difference function here as well
	a = np.zeros_like( wAst )
	a[idxAst] = 1
	b = np.zeros_like( wAst )
	b[idxHat] = 1
	return np.linalg.norm( a - b, 1 )//2

Z = np.loadtxt( "train" )
wAst = np.loadtxt( "wAstTrain" )

k = 20

y = Z[:,0]
X = Z[:,1:]

# To avoid unlucky outcomes try running the code several times
numTrials = 5

# Try various timeouts - the timeouts are in seconds
timeouts = np.array( [0.1, 0.2, 1, 2, 5] )

# Try checking for timeout every 10 iterations
spacing = 10

result = np.zeros( (len( timeouts ), 5) )


learn_c = [0.05]

for i in range( len( timeouts )):
	to = timeouts[i]
	for C in learn_c:
		#print("The learning rate param is " + str(C))
		avgObj = 0
		avgDist = 0
		avgSupp = 0
		avgTime = 0
		for t in range( numTrials ):
			(w, totTime) = solver( X, y, to, spacing, C)
			avgObj = avgObj + getObjValue( X, y, w )
			avgDist = avgDist + getModelError( w, wAst )
			avgSupp = avgSupp + getSupportError( w, wAst, k )
			avgTime = avgTime + totTime
			#print(w[750], w[656], w[157], w[551], w[288], w[887], w[640], w[447], w[923])

			# print(w.shape)
			# print(((w>=-1)*( w<=1)))
			# print(np.argsort((w>=-1)*( w<=1))[:12])
			# for j in range(w.shape[0]):
			# 	if w[j]>=1 and w[j]<=-1:
			# 		continue
			# 	else:
			# 		w[j]=0
			# print(w[750], w[656], w[157], w[551], w[288], w[887], w[640], w[447], w[923])
			
		result[i, 0] = avgObj/numTrials
		result[i, 1] = avgDist/numTrials
		result[i, 2] = avgSupp/numTrials
		result[i, 3] = avgTime/numTrials
		result[i, 4] = C
		formatted_res = [ '%.6f' % elem for elem in result[i] ]
		print(formatted_res)

np.savetxt( "result", result, fmt = "%.6f" )