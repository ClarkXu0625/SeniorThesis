import logging
logging.root.setLevel(logging.INFO)

import xcs
from xcs import XCSAlgorithm
from xcs.scenarios import MUXProblem, ScenarioObserver

scenario = ScenarioObserver(MUXProblem(50000))
algorithm = XCSAlgorithm

algorithm.exploration_probability = .1
algorithm.discount_factor = 0
algorithm.do_ga_subsumption = True
algorithm.do_action_set_subsumption = True

model = algorithm.new_model(scenario)
# model.run(scenario, learn=True)
# print(model)