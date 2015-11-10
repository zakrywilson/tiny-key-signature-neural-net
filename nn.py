# Zach Wilson
# key signature nn in python

import numpy as np

# sigmoid function
def nonlin(x, deriv=False):
  if (deriv==True):
    return x * (1 - x)
  return 1 / (1 + np.exp(-x))

# input dataset: 12 possible key signatures
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
synapse = 2 * np.random.random((12, 1)) - 1

for iter in xrange(5000):

  # forward propagation
  layer_one = i
  layer_two = nonlin(np.dot(layer_one, synapse))

  # how much did we miss?
  layer_two_error = o - layer_two

  # multiply net error by slope of the sigmoid at values in layer_two
  layer_two_delta = layer_two_error * nonlin(layer_two, True)

  # update weights
  synapse += np.dot(layer_one.T, layer_two_delta)

# dictionary for displaying note names and future developements
notes = {'0.05': 'C ', '0.10': 'C#', '0.15': 'D ', '0.20': 'D#',
         '0.25': 'E ', '0.30': 'F ', '0.35': 'F#', '0.40': 'G ',
         '0.45': 'G#', '0.50': 'A ', '0.55': 'A#', '0.60': 'B '}

# display results
print "   --------        -------"
print "   expected        network"
print "   --------        -------"
for expected, answer in zip(o.tolist(), layer_two.tolist()):
  for exp, ans in zip(expected, answer):
    # round the net's answer to the nearest 100th
    ans = round(ans, 2)
    # was the network correct?
    right_or_wrong = "correct" if ans == exp else "incorrect"
    # format it all pretty
    print notes.get(format(exp, '.2f')), " " ,  \
          format(exp, '.2f'), "   -->   ",     \
          format(ans, '.2f'), "   ", right_or_wrong

