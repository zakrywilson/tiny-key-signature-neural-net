# Zach Wilson
# key signature nn in python

import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
  if(deriv==True):
    return x*(1-x)
  return 1/(1+np.exp(-x))

# input dataset
i = np.array([ 
    [1,0,1,0,1,1,0,1,0,1,0,1], # .05: C major
    [1,1,0,1,0,1,1,0,1,0,1,0], # .10: C-sharp major
    [0,1,1,0,1,0,1,1,0,1,0,1], # .15: D major
    [1,0,1,1,0,1,0,1,1,0,1,0], # .20: D-sharp major
    [0,1,0,1,1,0,1,0,1,1,0,1], # .25: E major
    [1,0,1,0,1,1,0,1,0,1,1,0], # .30: F major
    [0,1,0,1,0,1,1,0,1,0,1,1], # .35: F-sharp major
    [1,0,1,0,1,0,1,1,0,1,0,1], # .40: G major
    [1,1,0,1,0,1,0,1,1,0,1,0], # .45: G-sharp major
    [0,1,1,0,1,0,1,0,1,1,0,1], # .50: A major
    [1,0,1,1,0,1,0,1,0,1,1,0], # .55: A-sharp major
    [0,1,0,1,1,0,1,0,1,0,1,1]  # .60: B major
                               ])
# output dataset: what the expected output should be for each         
o = np.array([[.05,.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60]]).T

# seed random numbers 
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((12,1)) - 1

for iter in xrange(20000):

  # forward propagation
  layer_one = i
  layer_two = nonlin(np.dot(layer_one,syn0))

  # how much did we miss?
  layer_two_error = o - layer_two

  # multiply how much we missed by the 
  # slope of the sigmoid at the values in layer_two
  layer_two_delta = layer_two_error * nonlin(layer_two,True)

  # update weights
  syn0 += np.dot(layer_one.T,layer_two_delta)

# because scientific notation is annoying...
#np.set_printoptions(suppress=True)

# display results
print "--------        -------"
print "expected        network"
print "--------        -------"
for x, y in zip(o.tolist(), layer_two.tolist()):
  for expected, answer in zip(x, y):
    answer = round(answer, 2)
    print " ", format(expected, '.2f') ,"   -->   ", format(answer, '.2f')






