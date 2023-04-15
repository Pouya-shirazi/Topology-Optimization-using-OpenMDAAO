from openmdao.api import Problem, IndepVarComp, ScipyOptimizeDriver
import openmdao.api as om
import Disc0
import Disc1
import Disc2
import Disc3

class MyModel(om.Group):
    def setup(self):
        # Define the disciplines
        self.add_subsystem('Disc1', Disc1(), promotes_inputs=['xy00'], promotes_outputs=['H'])
        self.add_subsystem('Disc2', Disc2(), promotes_inputs=['H'], promotes_outputs=['dvdh'])
        self.add_subsystem('Disc3', Disc3(), promotes_inputs=['H'], promotes_outputs=['comp'])
        
        # Define the connections
        self.connect('Disc1.H', 'Disc2.H')
        self.connect('Disc1.H', 'Disc3.H')
        
        # Set up the optimizer
        self.add_design_var('Disc1.xy00')
        self.add_objective('Disc3.comp')

prob = om.Problem()
prob.model = MyModel()

# Run the optimization
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['tol'] = 1e-6

prob.setup()
prob.run_driver()