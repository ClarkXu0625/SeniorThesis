import random

from xcs.scenarios import Scenario
from xcs.bitstrings import BitString

class HaystackProblem(Scenario):
    
    def __init__(self, training_cycles=1000, input_size=500):
        self.input_size = input_size
        self.possible_actions = (True, False)
        self.initial_training_cycles = training_cycles
        self.remaining_cycles = training_cycles
        self.needle_index = random.randrange(input_size)
        self.needle_value = None

    @property
    def is_dynamic(self):
        return False
        
    def get_possible_actions(self):
        return self.possible_actions
    
    def reset(self):
        self.remaining_cycles = self.initial_training_cycles
        self.needle_index = random.randrange(self.input_size)
        
    def more(self):
        return self.remaining_cycles > 0
    
    def sense(self):
        haystack = BitString.random(self.input_size)
        self.needle_value = haystack[self.needle_index]
        return haystack
    
    def execute(self, action):
        self.remaining_cycles -= 1
        return action == self.needle_value
    


##########################
### run scenario #########
##########################
    
import logging
import xcs

from xcs.scenarios import ScenarioObserver

# Setup logging so we can see the test run as it progresses.
logging.root.setLevel(logging.INFO)

# Create the scenario instance
problem = HaystackProblem()

# Wrap the scenario instance in an observer so progress gets logged,
# and pass it on to the test() function.
xcs.test(scenario=ScenarioObserver(problem))

print(len('##1#0###011#001#111000##0#11#0#01111##1#0#01###1001#111#100###101#10101011#0111#0010#0#10##00##1#11000001010111#0###111#11#1100100000##1##01100100#0##110#0#1#10001011#0#1#01#11001#0##0##000#111#11#1#1#00#11#000#11##110001001100#0#1#1#1100##1#101011#1101100#101#01011#1110101#000#01##11#0#0110##1#00#00#0#0#111#0##00111#0#00111110#101#10101##001#1100##10010#0110##10#011101#11#011##10##1#100#0##1#10##1##1000000#1##01#0#11##1001#0#1#11011#00#11##1##1##0#0#1###10#1000#010#0110011#110#11100#1##00#0###1 '))