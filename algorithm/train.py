import logging
from xcs import XCSAlgorithm
from xcs.scenarios import MUXProblem, ScenarioObserver

# Create a scenario instance, either by instantiating one of the
# predefined scenarios provided in xcs.scenarios, or by creating your
# own subclass of the xcs.scenarios.Scenario base class and
# instantiating it.
scenario = MUXProblem(training_cycles=50000)

# If you want to log the process of the run as it proceeds, set the
# logging level with the built-in logging module, and wrap the
# scenario with an OnLineObserver.
logging.root.setLevel(logging.INFO)
scenario = ScenarioObserver(scenario)

# Instantiate the algorithm and set the parameters to values that are
# appropriate for the scenario. Calling help(XCSAlgorithm) will give
# you a description of each parameter's meaning.
algorithm = XCSAlgorithm()
algorithm.exploration_probability = .1
algorithm.do_ga_subsumption = True
algorithm.do_action_set_subsumption = True

# Create a classifier set from the algorithm, tailored for the
# scenario you have selected.
model = algorithm.new_model(scenario)

# Run the classifier set in the scenario, optimizing it as the
# scenario unfolds.
model.run(scenario, learn=True)

# Use the built-in pickle module to save/reload your model for reuse.
import pickle
pickle.dump(model, open('model.bin', 'wb'))
reloaded_model = pickle.load(open('model.bin', 'rb'))

# Or just print the results out.
print(model)

# Or get a quick list of the best classifiers discovered.
for rule in model:
    if rule.fitness <= .5 or rule.experience < 10:
        continue
    print(rule.condition, '=>', rule.action, ' [%.5f]' % rule.fitness)