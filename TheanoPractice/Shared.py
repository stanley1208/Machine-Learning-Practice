import numpy as np
import theano
import theano.tensor as T

state=theano.shared(np.array(0,dtype=np.float64),'state')
inc=T.scalar('inc',dtype=state.dtype)
accumulator=theano.function([inc],state,updates=[(state,state+inc)])

# to get variable value
print(state.get_value())
accumulator(1)
print(state.get_value())
accumulator(10)
print(state.get_value())


# to set variable value
state.set_value(-1)
accumulator(3)
print(state.get_value())


# temporary replace shared variable with another value in another function
tmp_func=state*2+inc
a=T.scalar(dtype=state.dtype)
skip_shared=theano.function([inc,a],tmp_func,givens=[(state,a)])
print(skip_shared(5,5))
print(state.get_value())
