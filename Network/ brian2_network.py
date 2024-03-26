from brian2 import *
from brian2.units.allunits import *
from matplotlib import pyplot as plt

start_scope()

tau = 10*ms
eqs = '''
dv/dt = (1-v)/tau : 1
'''

G = NeuronGroup(1, eqs)
run(100*ms)