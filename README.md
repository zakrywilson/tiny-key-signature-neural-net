# Tiny key signature neural net in Python

Feed-forward neural network that learns to identify 
the pattern of a [major scale](https://en.wikipedia.org/wiki/Major_scale)
in fewer than 5,000 iterations.

## Requirements

Python 2.7.10

Numpy 1.9.2

## How to run

`> python nn.py`

#### Installing Numpy

`> pip install numpy`

### Note:
This neural net is not learning much because the inputs are so simple.
In later iterations, I'll add functionality for it to learn on variable input
with noise, but at the moment it only learns on this very trivial test data.
