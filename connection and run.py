from openmdao.api import Problem, IndepVarComp, ScipyOptimizeDriver, Group
import openmdao.api as om
# change directory to the path containing the disciplines

# import the disciplines
import Disc0
import Disc1
import Disc2
import Disc3

class MyModel(Group):
    def setup(self):
        # Define the disciplines
        d1 = Disc1.disc1()
        d2 = Disc2.disc2()
        d3 = Disc3.disc3()

        self.add_subsystem('Disc1', d1, promotes_inputs=['xy00'], promotes_outputs=['H'])
        self.add_subsystem('Disc2', d2, promotes_inputs=['H'], promotes_outputs=['dvdh'])
        self.add_subsystem('Disc3', d3, promotes_inputs=['H'], promotes_outputs=['Comp'])


prob = om.Problem()
prob.model = MyModel()

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'

prob.model.add_design_var('xy00')
prob.model.add_objective('Comp')
prob.model.add_constraint('dvdh', lower=0.0)

prob.setup()
prob.set_solver_print(level=0)

#prob.set_val('xy00', np.array([0.0, 0.0])

prob.run_driver()

print('Minimum Compliance =', prob.get_val('Comp'))
print('Optimal Design =', prob.get_val('H'))
