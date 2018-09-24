from simtk import unit
from openmmtools import testsystems, cache
from openmmtools.mcmc import GHMCMove, MCMCSampler, MCRotationMove, BaseIntegratorMove, IntegratorMoveError
from openmmtools.states import ThermodynamicState, SamplerState, CompoundThermodynamicState
from simtk.openmm import CompoundIntegrator
from simtk import unit
from openmmtools.utils import RestorableOpenMMObject
from openmmtools.integrators import ThermostatedIntegrator, NonequilibriumLangevinIntegrator
import numpy as np
import copy, sys
import logging
from simtk import openmm
import parmed
from openmmtools.utils import SubhookedABCMeta, Timer, RestorableOpenMMObject
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
np.random.RandomState(seed=3134)
logging.basicConfig(
    format='%(asctime)s | %(levelname)s : %(message)s',
    level=logging.INFO,
    stream=sys.stdout)


class LangevinIntegrator(ThermostatedIntegrator):

    _kinetic_energy = "0.5 * m * v * v"

    def __init__(
            self,
            temperature=298.0 * unit.kelvin,
            collision_rate=1.0 / unit.picoseconds,
            timestep=1.0 * unit.femtoseconds,
            splitting="V R O R V",
            constraint_tolerance=1e-8,
            measure_shadow_work=False,
            measure_heat=False,
    ):

        # Compute constants
        gamma = collision_rate
        self._gamma = gamma

        # Check if integrator is metropolized by checking for M step:
        if splitting.find("{") > -1:
            metropolized_integrator = True
            # We need to measure shadow work if Metropolization is used
            measure_shadow_work = True
        else:
            metropolized_integrator = False

        # Record whether we are measuring heat and shadow work
        #self._measure_heat = measure_heat
        #self._measure_shadow_work = measure_shadow_work

        ORV_counts, mts, force_group_nV = self._parse_splitting_string(
            splitting)

        # Record splitting.
        self._splitting = splitting
        self._ORV_counts = ORV_counts
        self._mts = mts
        self._force_group_nV = force_group_nV

        # Create a new CustomIntegrator
        super(LangevinIntegrator, self).__init__(temperature, timestep)

        # Initialize
        self.addPerDofVariable("sigma", 0)

        # Velocity mixing parameter: current velocity component
        h = timestep / max(1, ORV_counts['O'])
        self.addGlobalVariable("a", np.exp(-gamma * h))

        # Velocity mixing parameter: random velocity component
        self.addGlobalVariable("b", np.sqrt(1 - np.exp(-2 * gamma * h)))

        # Positions before application of position constraints
        self.addPerDofVariable("x1", 0)

        # Set constraint tolerance
        self.setConstraintTolerance(constraint_tolerance)

        # Add global variables
        self._add_global_variables()

        # Add integrator steps
        self._add_integrator_steps()

    @property
    def _step_dispatch_table(self):
        """dict: The dispatch table step_name -> add_step_function."""
        # TODO use methoddispatch (see yank.utils) when dropping Python 2 support.
        dispatch_table = {
            'O': (self._add_O_step, False),
            'R': (self._add_R_step, False),
            '{': (self._add_metropolize_start, False),
            '}': (self._add_metropolize_finish, False),
            'V': (self._add_V_step, True)
        }
        return dispatch_table

    def _add_global_variables(self):
        """Add global bookkeeping variables."""
        #self.addGlobalVariable('step', 0)

        if self._measure_heat:
            self.addGlobalVariable("heat", 0)

        if self._measure_shadow_work or self._measure_heat:
            self.addGlobalVariable("old_ke", 0)
            self.addGlobalVariable("new_ke", 0)

        if self._measure_shadow_work:
            self.addGlobalVariable("old_pe", 0)
            self.addGlobalVariable("new_pe", 0)
            self.addGlobalVariable("shadow_work", 0)

        # If we metropolize, we have to keep track of the before and after (x, v)
        if self._metropolized_integrator:
            self.addGlobalVariable("accept", 0)
            self.addGlobalVariable("ntrials", 0)
            self.addGlobalVariable("nreject", 0)
            self.addGlobalVariable("naccept", 0)
            self.addPerDofVariable("vold", 0)
            self.addPerDofVariable("xold", 0)

    def reset_heat(self):
        """Reset heat."""
        if self._measure_heat:
            self.setGlobalVariableByName('heat', 0.0)

    def reset_shadow_work(self):
        """Reset shadow work."""
        if self._measure_shadow_work:
            self.setGlobalVariableByName('shadow_work', 0.0)

    def reset_ghmc_statistics(self):
        """Reset GHMC acceptance rate statistics."""
        if self._metropolized_integrator:
            self.setGlobalVariableByName('ntrials', 0)
            self.setGlobalVariableByName('naccept', 0)
            self.setGlobalVariableByName('nreject', 0)

    def reset_steps(self):
        """Reset step counter.
        """
        self.setGlobalVariableByName('step', 0)

    def reset(self):
        """Reset all statistics (heat, shadow work, acceptance rates, step).
        """
        self.reset_heat()
        self.reset_shadow_work()
        self.reset_ghmc_statistics()
        self.reset_steps()

    def _get_energy_with_units(self, variable_name, dimensionless=False):
        """Retrive an energy/work quantity and return as unit-bearing or dimensionless quantity.

        Parameters
        ----------
        variable_name : str
           Name of the global context variable to retrieve
        dimensionless : bool, optional, default=False
           If specified, the energy/work is returned in reduced (kT) unit.

        Returns
        -------
        work : unit.Quantity or float
           If dimensionless=True, the work in kT (float).
           Otherwise, the unit-bearing work in units of energy.
        """
        work = self.getGlobalVariableByName(variable_name) * _OPENMM_ENERGY_UNIT
        if dimensionless:
            return work / self.kT
        else:
            return work

    def get_shadow_work(self, dimensionless=False):
        """Get the current accumulated shadow work.

        Parameters
        ----------
        dimensionless : bool, optional, default=False
           If specified, the work is returned in reduced (kT) unit.

        Returns
        -------
        work : unit.Quantity or float
           If dimensionless=True, the protocol work in kT (float).
           Otherwise, the unit-bearing protocol work in units of energy.
        """
        try:
            return self._get_energy_with_units(
                "shadow_work", dimensionless=dimensionless)
        except:
            #if not self._measure_shadow_work:
            raise Exception(
                "This integrator must be constructed with 'measure_shadow_work=True' to measure shadow work."
            )
        #return self._get_energy_with_units("shadow_work", dimensionless=dimensionless)

    @property
    def _metropolized_integrator(self):
        try:
            self.getGlobalVariableByName('ntrials')
            self.getGlobalVariableByName('naccept')
            self.getGlobalVariableByName('nreject')
        except:
            return False
        return True

    @property
    def _measure_shadow_work(self):
        try:
            self.getGlobalVariableByName('shadow_work')
        except:
            return False
        return True

    @property
    def _measure_heat(self):
        try:
            self.getGlobalVariableByName('heat')
        except:
            return False
        return True

    @property
    def shadow_work(self):
        return self.get_shadow_work()

    @property
    def step(self):
        return self.get_step()

    def get_step(self):
        try:
            return self.getGlobalVariableByName('step')
        except Exception as e:
            raise e

    def get_heat(self, dimensionless=False):
        """Get the current accumulated heat.

        Parameters
        ----------
        dimensionless : bool, optional, default=False
           If specified, the work is returned in reduced (kT) unit.

        Returns
        -------
        work : unit.Quantity or float
           If dimensionless=True, the heat in kT (float).
           Otherwise, the unit-bearing heat in units of energy.
        """
        try:
            return self._get_energy_with_units(
                "heat", dimensionless=dimensionless)
        except:
            raise Exception(
                "This integrator must be constructed with 'measure_heat=True' in order to measure heat."
            )
        #if not self._measure_heat:
        #    raise Exception("This integrator must be constructed with 'measure_heat=True' in order to measure heat.")
        #return self._get_energy_with_units("heat", dimensionless=dimensionless)

    @property
    def heat(self):
        return self.get_heat()

    def get_acceptance_rate(self):
        """Get acceptance rate for Metropolized integrators.

        Returns
        -------
        acceptance_rate : float
           Acceptance rate.
           An Exception is thrown if the integrator is not Metropolized.
        """
        if not self._metropolized_integrator:
            raise Exception(
                "This integrator must be Metropolized to return an acceptance rate."
            )
        return self.getGlobalVariableByName(
            "naccept") / self.getGlobalVariableByName("ntrials")

    @property
    def acceptance_rate(self):
        """Get acceptance rate for Metropolized integrators."""
        return self.get_acceptance_rate()

    @property
    def is_metropolized(self):
        """Return True if this integrator is Metropolized, False otherwise."""
        return self._metropolized_integrator

    def _add_integrator_steps(self):
        """Add the steps to the integrator--this can be overridden to place steps around the integration.
        """
        # Integrate
        self.addUpdateContextState()
        self.addComputeTemperatureDependentConstants({"sigma": "sqrt(kT/m)"})

        for i, step in enumerate(self._splitting.split()):
            self._substep_function(step)

    def _sanity_check(self, splitting):
        """Perform a basic sanity check on the splitting string to ensure that it makes sense.

        Parameters
        ----------
        splitting : str
            The string specifying the integrator splitting
        mts : bool
            Whether the integrator is a multiple timestep integrator
        allowed_characters : str, optional
            The characters allowed to be present in the splitting string.
            Default RVO and the digits 0-9.
        """

        # Space is just a delimiter--remove it
        splitting_no_space = splitting.replace(" ", "")

        allowed_characters = "0123456789"
        for key in self._step_dispatch_table:
            allowed_characters += key

        # sanity check to make sure only allowed combinations are present in string:
        for step in splitting.split():
            if step[0] == "V":
                if len(step) > 1:
                    try:
                        force_group_number = int(step[1:])
                        if force_group_number > 31:
                            raise ValueError(
                                "OpenMM only allows up to 32 force groups")
                    except ValueError:
                        raise ValueError("You must use an integer force group")
            elif step == "{":
                if "}" not in splitting:
                    raise ValueError("Use of { must be followed by }")
                if not self._verify_metropolization(splitting):
                    raise ValueError(
                        "Shadow work generating steps found outside the Metropolization block"
                    )
            elif step in allowed_characters:
                continue
            else:
                raise ValueError(
                    "Invalid step name '%s' used; valid step names are %s" %
                    (step, allowed_characters))

        # Make sure we contain at least one of R, V, O steps
        assert ("R" in splitting_no_space)
        assert ("V" in splitting_no_space)
        assert ("O" in splitting_no_space)

    def _verify_metropolization(self, splitting):
        """Verify that the shadow-work generating steps are all inside the metropolis block

        Returns False if they are not.

        Parameters
        ----------
        splitting : str
            The langevin splitting string

        Returns
        -------
        valid_metropolis : bool
            Whether all shadow-work generating steps are in the {} block
        """
        # check that there is exactly one metropolized region
        #this pattern matches the { literally, then any number of any character other than }, followed by another {
        #If there's a match, then we have an attempt at a nested metropolization, which is unsupported
        regex_nested_metropolis = "{[^}]*{"
        pattern = re.compile(regex_nested_metropolis)
        if pattern.match(splitting.replace(" ", "")):
            raise ValueError("There can only be one Metropolized region.")

        # find the metropolization steps:
        M_start_index = splitting.find("{")
        M_end_index = splitting.find("}")

        # accept/reject happens before the beginning of metropolis step
        if M_start_index > M_end_index:
            return False

        #pattern to find whether any shadow work producing steps lie outside the metropolization region
        RV_outside_metropolis = "[RV](?![^{]*})"
        outside_metropolis_check = re.compile(RV_outside_metropolis)
        if outside_metropolis_check.match(splitting.replace(" ", "")):
            return False
        else:
            return True

    def _add_R_step(self):
        """Add an R step (position update) given the velocities.
        """
        if self._measure_shadow_work:
            self.addComputeGlobal("old_pe", "energy")
            self.addComputeSum("old_ke", self._kinetic_energy)

        n_R = self._ORV_counts['R']

        # update positions (and velocities, if there are constraints)
        self.addComputePerDof("x", "x + ((dt / {}) * v)".format(n_R))
        self.addComputePerDof("x1", "x")  # save pre-constraint positions in x1
        self.addConstrainPositions()  # x is now constrained
        self.addComputePerDof("v", "v + ((x - x1) / (dt / {}))".format(n_R))
        self.addConstrainVelocities()

        if self._measure_shadow_work:
            self.addComputeGlobal("new_pe", "energy")
            self.addComputeSum("new_ke", self._kinetic_energy)
            self.addComputeGlobal(
                "shadow_work",
                "shadow_work + (new_ke + new_pe) - (old_ke + old_pe)")

    def _add_V_step(self, force_group="0"):
        """Deterministic velocity update, using only forces from force-group fg.

        Parameters
        ----------
        force_group : str, optional, default="0"
           Force group to use for this step
        """
        if self._measure_shadow_work:
            self.addComputeSum("old_ke", self._kinetic_energy)

        # update velocities
        if self._mts:
            self.addComputePerDof(
                "v", "v + ((dt / {}) * f{} / m)".format(
                    self._force_group_nV[force_group], force_group))
        else:
            self.addComputePerDof(
                "v", "v + (dt / {}) * f / m".format(self._force_group_nV["0"]))

        self.addConstrainVelocities()

        if self._measure_shadow_work:
            self.addComputeSum("new_ke", self._kinetic_energy)
            self.addComputeGlobal("shadow_work",
                                  "shadow_work + (new_ke - old_ke)")

    def _add_O_step(self):
        """Add an O step (stochastic velocity update)
        """
        if self._measure_heat:
            self.addComputeSum("old_ke", self._kinetic_energy)

        # update velocities
        self.addComputePerDof("v", "(a * v) + (b * sigma * gaussian)")
        self.addConstrainVelocities()

        if self._measure_heat:
            self.addComputeSum("new_ke", self._kinetic_energy)
            self.addComputeGlobal("heat", "heat + (new_ke - old_ke)")

    def _substep_function(self, step_string):
        """Take step string, and add the appropriate R, V, O step with appropriate parameters.

        The step string input here is a single character (or character + number, for MTS)
        """
        function, can_accept_force_groups = self._step_dispatch_table[
            step_string[0]]
        if can_accept_force_groups:
            force_group = step_string[1:]
            function(force_group)
        else:
            function()

    def _parse_splitting_string(self, splitting_string):
        """Parse the splitting string to check for simple errors and extract necessary information

        Parameters
        ----------
        splitting_string : str
            The string that specifies how to do the integrator splitting

        Returns
        -------
        ORV_counts : dict
            Number of O, R, and V steps
        mts : bool
            Whether the splitting specifies an MTS integrator
        force_group_n_V : dict
            Specifies the number of V steps per force group. {"0": nV} if not MTS
        """
        # convert the string to all caps
        splitting_string = splitting_string.upper()

        # sanity check the splitting string
        self._sanity_check(splitting_string)

        ORV_counts = dict()

        # count number of R, V, O steps:
        for step_symbol in self._step_dispatch_table:
            ORV_counts[step_symbol] = splitting_string.count(step_symbol)

        # split by delimiter (space)
        step_list = splitting_string.split(" ")

        # populate a list with all the force groups in the system
        force_group_list = []
        for step in step_list:
            # if the length of the step is greater than one, it has a digit after it
            if step[0] == "V" and len(step) > 1:
                force_group_list.append(step[1:])

        # Make a set to count distinct force groups
        force_group_set = set(force_group_list)

        # check if force group list cast to set is longer than one
        # If it is, then multiple force groups are specified
        if len(force_group_set) > 1:
            mts = True
        else:
            mts = False

        # If the integrator is MTS, count how many times the V steps appear for each
        if mts:
            force_group_n_V = {
                force_group: 0
                for force_group in force_group_set
            }
            for step in step_list:
                if step[0] == "V":
                    # ensure that there are no V-all steps if it's MTS
                    assert len(step) > 1
                    # extract the index of the force group from the step
                    force_group_idx = step[1:]
                    # increment the number of V calls for that force group
                    force_group_n_V[force_group_idx] += 1
        else:
            force_group_n_V = {"0": ORV_counts["V"]}

        return ORV_counts, mts, force_group_n_V

    def _add_metropolize_start(self):
        """Save the current x and v for a metropolization step later"""
        self.addComputePerDof("xold", "x")
        self.addComputePerDof("vold", "v")

    def _add_metropolize_finish(self):
        """Add a Metropolization (based on shadow work) step to the integrator.

        When Metropolization occurs, shadow work is reset.
        """
        self.addComputeGlobal("accept",
                              "step(exp(-(shadow_work)/kT) - uniform)")
        self.addComputeGlobal("ntrials", "ntrials + 1")
        self.beginIfBlock("accept != 1")
        self.addComputePerDof("x", "xold")
        self.addComputePerDof("v", "-vold")
        self.addComputeGlobal("nreject", "nreject + 1")
        self.endBlock()
        self.addComputeGlobal("naccept", "ntrials - nreject")
        self.addComputeGlobal("shadow_work", "0")


class AlchemicalNonequilibriumLangevinIntegrator(
        NonequilibriumLangevinIntegrator):
    def __init__(self,
                 alchemical_functions=None,
                 splitting="O { V R H R V } O",
                 nsteps_neq=100,
                 *args,
                 **kwargs):
        if alchemical_functions is None:
            alchemical_functions = dict()

        if (nsteps_neq < 0) or (nsteps_neq != int(nsteps_neq)):
            raise Exception('nsteps_neq must be an integer >= 0')

        self._alchemical_functions = alchemical_functions
        self._n_steps_neq = nsteps_neq  # number of integrator steps

        # collect the system parameters.
        self._system_parameters = {
            system_parameter
            for system_parameter in alchemical_functions.keys()
        }

        # call the base class constructor
        kwargs['splitting'] = splitting
        super(AlchemicalNonequilibriumLangevinIntegrator, self).__init__(
            *args, **kwargs)

    @property
    def _step_dispatch_table(self):
        """dict: The dispatch table step_name -> add_step_function."""
        # TODO use methoddispatch (see yank.utils) when dropping Python 2 support.
        dispatch_table = super(AlchemicalNonequilibriumLangevinIntegrator,
                               self)._step_dispatch_table
        dispatch_table['H'] = (self._add_alchemical_perturbation_step, False)
        return dispatch_table

    def _add_global_variables(self):
        """Add the appropriate global parameters to the CustomIntegrator. nsteps refers to the number of
        total steps in the protocol.

        Parameters
        ----------
        nsteps : int, greater than 0
            The number of steps in the switching protocol.
        """
        super(AlchemicalNonequilibriumLangevinIntegrator,
              self)._add_global_variables()
        self.addGlobalVariable('Eold',
                               0)  #old energy value before perturbation
        self.addGlobalVariable('Enew', 0)  #new energy value after perturbation
        self.addGlobalVariable(
            'lambda', 0.0
        )  # parameter switched from 0 <--> 1 during course of integrating internal 'nsteps' of dynamics
        self.addGlobalVariable(
            'nsteps', self._n_steps_neq
        )  # total number of NCMC steps to perform; this SHOULD NOT BE CHANGED during the protocol
        self.addGlobalVariable(
            'step', 0
        )  # step counter for handling initialization and terminating integration

        # Keep track of number of Hamiltonian updates per nonequilibrium switch
        n_H = self._ORV_counts['H']  # number of H updates per integrator step
        self._n_lambda_steps = self._n_steps_neq * n_H  # number of Hamiltonian increments per switching step
        if self._n_steps_neq == 0:
            self._n_lambda_steps = 1  # instantaneous switching
        self.addGlobalVariable(
            'n_lambda_steps', self._n_lambda_steps
        )  # total number of NCMC steps to perform; this SHOULD NOT BE CHANGED during the protocol
        self.addGlobalVariable('lambda_step', 0)

    def _add_update_alchemical_parameters_step(self):
        """
        Add step to update Context parameters according to provided functions.
        """
        for context_parameter in self._alchemical_functions:
            if context_parameter in self._system_parameters:
                self.addComputeGlobal(
                    context_parameter,
                    self._alchemical_functions[context_parameter])

    def _add_alchemical_perturbation_step(self):
        """
        Add alchemical perturbation step, accumulating protocol work.

        TODO: Extend this to be able to handle force groups?

        """
        # Store initial potential energy
        self.addComputeGlobal("Eold", "energy")

        # Update lambda and increment that tracks updates.
        self.addComputeGlobal('lambda', '(lambda_step+1)/n_lambda_steps')
        self.addComputeGlobal('lambda_step', 'lambda_step + 1')

        # Update all slaved alchemical parameters
        self._add_update_alchemical_parameters_step()

        # Accumulate protocol work
        self.addComputeGlobal("Enew", "energy")
        self.addComputeGlobal("protocol_work", "protocol_work + (Enew-Eold)")

    def _add_integrator_steps(self):
        """
        Override the base class to insert reset steps around the integrator.
        """

        # First step: Constrain positions and velocities and reset work accumulators and alchemical integrators
        self.beginIfBlock('step = 0')
        self.addConstrainPositions()
        self.addConstrainVelocities()
        self._add_reset_protocol_work_step()
        self._add_alchemical_reset_step()
        self.endBlock()

        # Main body
        if self._n_steps_neq == 0:
            # If nsteps = 0, we need to force execution on the first step only.
            self.beginIfBlock('step = 0')
            super(AlchemicalNonequilibriumLangevinIntegrator,
                  self)._add_integrator_steps()
            self.addComputeGlobal("step", "step + 1")
            self.endBlock()
        else:
            #call the superclass function to insert the appropriate steps, provided the step number is less than n_steps
            self.beginIfBlock("step < nsteps")
            super(AlchemicalNonequilibriumLangevinIntegrator,
                  self)._add_integrator_steps()
            self.addComputeGlobal("step", "step + 1")
            self.endBlock()

    def _add_alchemical_reset_step(self):
        """
        Reset the alchemical lambda to its starting value
        """
        self.addComputeGlobal("lambda", "0")
        self.addComputeGlobal("protocol_work", "0")
        self.addComputeGlobal("step", "0")
        self.addComputeGlobal("lambda_step", "0")
        # Add all dependent parameters
        self._add_update_alchemical_parameters_step()


temperature = 300 * unit.kelvin
collision_rate = 1 / unit.picoseconds
timestep = 1.0 * unit.femtoseconds
n_steps = 100
alchemical_functions = {'lambda_sterics': 'lambda'}

from openmmtools import testsystems, alchemy
factory = alchemy.AbsoluteAlchemicalFactory(consistent_exceptions=False)
ala = testsystems.AlanineDipeptideVacuum()
alchemical_region = alchemy.AlchemicalRegion(alchemical_atoms=range(22))
alchemical_atoms = list(alchemical_region.alchemical_atoms)
alanine_alchemical_system = factory.create_alchemical_system(
    reference_system=ala.system, alchemical_regions=alchemical_region)
alchemical_state = alchemy.AlchemicalState.from_system(
    alanine_alchemical_system)
thermo_state = ThermodynamicState(
    system=alanine_alchemical_system, temperature=300 * unit.kelvin)
compound_state = CompoundThermodynamicState(
    thermo_state, composable_states=[alchemical_state])
sampler_state = SamplerState(positions=ala.positions)
print(compound_state.lambda_sterics)
print(compound_state.lambda_electrostatics)

ncmc_integrator = AlchemicalNonequilibriumLangevinIntegrator(
    alchemical_functions,
    splitting='H R V O V R H',
    #splitting='O { V R H R V } O',
    temperature=temperature,
    collision_rate=collision_rate,
    timestep=timestep,
    nsteps_neq=n_steps,
    measure_heat=True)
integrator = LangevinIntegrator(
    temperature=temperature,
    timestep=timestep,
    collision_rate=collision_rate,
    measure_heat=True)
#print(integrator)
#print(dir(integrator))

#print(integrator.getGlobalVariableByName('heat'))

compound_integrator = CompoundIntegrator()
compound_integrator.addIntegrator(ncmc_integrator)
compound_integrator.addIntegrator(integrator)
compound_integrator.setCurrentIntegrator(1)

dir(compound_integrator)

context_cache = cache.global_context_cache
context, compound_integrator = context_cache.get_context(
    compound_state, compound_integrator)

dir(compound_integrator)

# If we reassign velocities, we can ignore the ones in sampler_state.
sampler_state.apply_to_context(context)
context.setVelocitiesToTemperature(compound_state.temperature)

#langevin_integrator = compound_integrator.getIntegrator(1)
#RestorableOpenMMObject.restore_interface(langevin_integrator)
#global_variables = {
#    langevin_integrator.getGlobalVariableName(index): index
#    for index in range(langevin_integrator.getNumGlobalVariables())
#}
#print(global_variables)

dir(compound_integrator)
compound_integrator.step(5)

# global_variables = {
#     langevin_integrator.getGlobalVariableName(index): index
#     for index in range(langevin_integrator.getNumGlobalVariables())
# }
# print(global_variables)

langevin_integrator = compound_integrator.getIntegrator(1)
dir(langevin_integrator)
#RestorableOpenMMObject.restore_interface(langevin_integrator)

print(langevin_integrator)
#print(dir(langevin_integrator))
langevin_integrator.reset()

alch_integrator = compound_integrator.getIntegrator(0)
RestorableOpenMMObject.restore_interface(alch_integrator)

print(alch_integrator)
#print(dir(alch_integrator))
alch_integrator.reset()


class MCMCSampler(object):
    """
    Markov chain Monte Carlo (MCMC) sampler.
    This is a minimal functional implementation placeholder until we can replace this with MCMCSampler from `openmmmcmc`.
    Properties
    ----------
    positions : simtk.unit.Quantity of size [nparticles,3] with units compatible with nanometers
        The current positions.
    iteration : int
        Iterations completed.
    verbose : bool
        If True, verbose output is printed
    Examples
    --------
    >>> # Create a test system
    >>> test = testsystems.AlanineDipeptideVacuum()
    >>> # Create a sampler state.
    >>> sampler_state = SamplerState(positions=test.positions)
    >>> # Create a thermodynamic state.
    >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298.0*unit.kelvin)
    >>> # Create an MCMC sampler
    >>> sampler = MCMCSampler(thermodynamic_state, sampler_state)
    >>> # Run the sampler
    >>> sampler.verbose = False
    >>> sampler.run()
    """

    def __init__(self,
                 thermodynamic_state=None,
                 sampler_state=None,
                 platform=None,
                 ncfile=None):
        """
        Create an MCMC sampler.
        Parameters
        ----------
        thermodynamic_state : ThermodynamicState
            The thermodynamic state to simulate
        sampler_state : SamplerState
            The initial sampler state to simulate from.
        platform : simtk.openmm.Platform, optional, default=None
            If specified, this platform will be used
        ncfile : netCDF4.Dataset, optional, default=None
            NetCDF storage file.
        """
        if thermodynamic_state is None:
            raise Exception("'thermodynamic_state' must be specified")
        if sampler_state is None:
            raise Exception("'sampler_state' must be specified")

        self.thermodynamic_state = thermodynamic_state
        self.sampler_state = sampler_state
        # Initialize
        self.iteration = 0
        # For GHMC / Langevin integrator
        self.collision_rate = 1.0 / unit.picoseconds
        self.timestep = 2.0 * unit.femtoseconds
        self.nsteps = 500  # number of steps per update
        self.verbose = True
        self.platform = platform

        # For writing PDB files
        self.pdbfile = None
        self.topology = None

        self._timing = dict()
        self._initializeNetCDF(ncfile)
        self._initialized = False

    def _initialize(self):
        # Create an integrator
        integrator_name = 'Langevin'
        if integrator_name == 'GHMC':
            from openmmtools.integrators import GHMCIntegrator
            self.integrator = GHMCIntegrator(
                temperature=self.thermodynamic_state.temperature,
                collision_rate=self.collision_rate,
                timestep=self.timestep)
        elif integrator_name == 'Langevin':
            from simtk.openmm import LangevinIntegrator
            self.integrator = LangevinIntegrator(
                self.thermodynamic_state.temperature, self.collision_rate,
                self.timestep)
        else:
            raise Exception(
                "integrator_name '%s' not valid." % (integrator_name))

        # Create a Context
        if self.platform is not None:
            self.context = openmm.Context(self.thermodynamic_state.system,
                                          self.integrator, self.platform)
        else:
            self.context = openmm.Context(self.thermodynamic_state.system,
                                          self.integrator)
        self.thermodynamic_state.update_context(self.context, self.integrator)
        self.sampler_state.update_context(self.context)
        self.context.setVelocitiesToTemperature(
            self.thermodynamic_state.temperature)

        self._initialized = True

    def _initializeNetCDF(self, ncfile):
        self.ncfile = ncfile
        if self.ncfile == None:
            return

        natoms = self.thermodynamic_state.system.getNumParticles()
        self.ncfile.createDimension('iterations', None)
        self.ncfile.createDimension(
            'atoms', natoms)  # TODO: What do we do if dimension can change?
        self.ncfile.createDimension('spatial', 3)

        self.ncfile.createVariable(
            'positions',
            'f4',
            dimensions=('iterations', 'atoms', 'spatial'),
            zlib=True,
            chunksizes=(1, natoms, 3))
        self.ncfile.createVariable(
            'box_vectors',
            'f4',
            dimensions=('iterations', 'spatial', 'spatial'),
            zlib=True,
            chunksizes=(1, 3, 3))
        self.ncfile.createVariable(
            'potential', 'f8', dimensions=('iterations', ), chunksizes=(1, ))
        self.ncfile.createVariable(
            'sample_positions_time',
            'f4',
            dimensions=('iterations', ),
            chunksizes=(1, ))

        # Weight adaptation information
        self.ncfile.createVariable(
            'stage', 'i2', dimensions=('iterations', ), chunksizes=(1, ))
        self.ncfile.createVariable(
            'gamma', 'f8', dimensions=('iterations', ), chunksizes=(1, ))

    def update(self):
        """
        Update the sampler with one step of sampling.
        """
        if not self._initialized:
            self._initialize()

        if self.verbose:
            print("." * 80)
            print("MCMC sampler iteration %d" % self.iteration)

        initial_time = time.time()

        # Reset statistics
        if hasattr(self.integrator, 'setGlobalVariableByName'):
            self.integrator.setGlobalVariableByName('naccept', 0)

        # Take some steps
        self.integrator.step(self.nsteps)

        # Get new sampler state.
        self.sampler_state = SamplerState.createFromContext(self.context)

        # Report statistics
        if hasattr(self.integrator, 'getGlobalVariableByName'):
            naccept = self.integrator.getGlobalVariableByName('naccept')
            fraction_accepted = float(naccept) / float(self.nsteps)
            if self.verbose:
                print("Accepted %d / %d GHMC steps (%.2f%%)." %
                      (naccept, self.nsteps, fraction_accepted * 100))

        final_time = time.time()
        elapsed_time = final_time - initial_time
        self._timing['sample positions'] = elapsed_time

        if self.verbose:
            final_energy = self.context.getState(
                getEnergy=True).getPotentialEnergy(
                ) * self.thermodynamic_state.beta
            print('Final energy is %12.3f kT' % (final_energy))
            print('elapsed time %8.3f s' % elapsed_time)

        if self.ncfile:
            self.ncfile.variables['positions'][
                self.
                iteration, :, :] = self.sampler_state.positions[:, :] / unit.nanometers
            for k in range(3):
                self.ncfile.variables['box_vectors'][
                    self.iteration,
                    k, :] = self.sampler_state.box_vectors[k, :] / unit.nanometers
            self.ncfile.variables['potential'][
                self.
                iteration] = self.thermodynamic_state.beta * self.context.getState(
                    getEnergy=True).getPotentialEnergy()
            self.ncfile.variables['sample_positions_time'][
                self.iteration] = elapsed_time

        # Increment iteration count
        self.iteration += 1

        if self.verbose:
            print("." * 80)
        if self.pdbfile is not None:
            print("Writing frame...")
            from simtk.openmm.app import PDBFile
            PDBFile.writeModel(self.topology, self.sampler_state.positions,
                               self.pdbfile, self.iteration)
            self.pdbfile.flush()

    def run(self, niterations=1):
        """
        Run the sampler for the specified number of iterations
        Parameters
        ----------
        niterations : int, optional, default=1
            Number of iterations to run the sampler for.
        """
        for iteration in range(niterations):
            self.update()


class NCMCSampler(object):
    def __init__(self, thermodynamic_state, sampler_state, move):
        # Make a deep copy of the state so that initial state is unchanged.
        self.thermodynamic_state = copy.deepcopy(thermodynamic_state)
        self.sampler_state = copy.deepcopy(sampler_state)
        self.move = move

    def run(self, n_iterations=1, integrator_idx=0):
        """
        Run the sampler for a specified number of iterations.

        Parameters
        ----------
        n_iterations : int
            Number of iterations of the sampler to run.

        """
        # Apply move for n_iterations.
        for iteration in range(n_iterations):
            self.move.apply(self.thermodynamic_state, self.sampler_state,
                            integrator_idx)

    def minimize(self,
                 tolerance=1.0 * unit.kilocalories_per_mole / unit.angstroms,
                 max_iterations=100,
                 context_cache=None):
        """Minimize the current configuration.

        Parameters
        ----------
        tolerance : simtk.unit.Quantity, optional
            Tolerance to use for minimization termination criterion (units of
            energy/(mole*distance), default is 1*kilocalories_per_mole/angstroms).
        max_iterations : int, optional
            Maximum number of iterations to use for minimization. If 0, the minimization
            will continue until convergence (default is 100).
        context_cache : openmmtools.cache.ContextCache, optional
            The ContextCache to use for Context creation. If None, the global cache
            openmmtools.cache.global_context_cache is used (default is None).

        """
        if context_cache is None:
            context_cache = cache.global_context_cache

        timer = Timer()

        # Use LocalEnergyMinimizer
        timer.start("Context request")
        integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
        context, integrator = context_cache.get_context(
            self.thermodynamic_state, integrator)
        self.sampler_state.apply_to_context(context)
        logger.debug("LocalEnergyMinimizer: platform is %s" %
                     context.getPlatform().getName())
        logger.debug("Minimizing with tolerance %s and %d max. iterations." %
                     (tolerance, max_iterations))
        timer.stop("Context request")

        timer.start("LocalEnergyMinimizer minimize")
        openmm.LocalEnergyMinimizer.minimize(context, tolerance,
                                             max_iterations)
        timer.stop("LocalEnergyMinimizer minimize")
        context_state = context.getState(
            getPositions=True,
            getVelocities=True,
            getEnergy=True,
            enforcePeriodicBox=self.thermodynamic_state.is_periodic)

        potential_energy = context_state.getPotentialEnergy()
        print(potential_energy)
        # Retrieve data.
        self.sampler_state.update_from_context(context)

        #timer.report_timing()


class NCMCMove(BaseIntegratorMove):
    def __init__(self,
                 timestep=1.0 * unit.femtosecond,
                 collision_rate=10.0 / unit.picoseconds,
                 n_steps=1000,
                 temperature=300 * unit.kelvin,
                 reassign_velocities=True,
                 **kwargs):
        super(NCMCMove, self).__init__(n_steps=n_steps, **kwargs)
        self.timestep = timestep
        self.temperature = temperature
        self.collision_rate = collision_rate
        self.n_accepted = 0  # Number of accepted steps.
        self.n_proposed = 0  # Number of attempted steps.
        #self.reporters = [NetCDF4Reporter('test.nc', 1)]
    @property
    def fraction_accepted(self):
        """Ratio between accepted over attempted moves (read-only).

        If the number of attempted steps is 0, this is numpy.NaN.

        """
        if self.n_proposed == 0:
            return np.NaN
        # TODO drop the casting when stop Python2 support
        return float(self.n_accepted) / self.n_proposed

    @property
    def statistics(self):
        """The acceptance statistics as a dictionary."""
        return dict(n_accepted=self.n_accepted, n_proposed=self.n_proposed)

    @statistics.setter
    def statistics(self, value):
        self.n_accepted = value['n_accepted']
        self.n_proposed = value['n_proposed']

    def reset_statistics(self):
        """Reset the internal statistics of number of accepted and attempted moves."""
        self.n_accepted = 0
        self.n_proposed = 0

    def apply(self, thermodynamic_state, sampler_state, integrator_idx):
        """Propagate the state through the integrator.

            This updates the SamplerState after the integration. It also logs
            benchmarking information through the utils.Timer class.

            Parameters
            ----------
            thermodynamic_state : openmmtools.states.ThermodynamicState
               The thermodynamic state to use to propagate dynamics.
            sampler_state : openmmtools.states.SamplerState
               The sampler state to apply the move to. This is modified.

            See Also
            --------
            openmmtools.utils.Timer

            """
        move_name = self.__class__.__name__  # shortcut
        timer = Timer()

        # Check if we have to use the global cache.
        if self.context_cache is None:
            context_cache = cache.global_context_cache
        else:
            print('Found context cache')
            context_cache = self.context_cache

        # Create integrator.
        integrator = self._get_integrator(thermodynamic_state)
        self.integrator_idx = integrator_idx

        # Create context.
        timer.start("{}: Context request".format(move_name))
        context, integrator = context_cache.get_context(
            thermodynamic_state, integrator)

        #print('SamplerContext compat:', sampler_state.is_context_compatible(context))
        integrator.setCurrentIntegrator(self.integrator_idx)
        #integrator = integrator.getIntegrator(integrator_idx)
        #RestorableOpenMMObject.restore_interface(integrator)
        #integrator.pretty_print()
        #print('Current Integrator:', integrator)

        timer.stop("{}: Context request".format(move_name))
        logger.debug("{}: Context obtained, platform is {}".format(
            move_name,
            context.getPlatform().getName()))

        # Perform the integration.
        for attempt_counter in range(self.n_restart_attempts + 1):

            #If we reassign velocities, we can ignore the ones in sampler_state.
            sampler_state.apply_to_context(
                context, ignore_velocities=self.reassign_velocities)
            if self.reassign_velocities:
                context.setVelocitiesToTemperature(
                    thermodynamic_state.temperature)

            # Subclasses may implement _before_integration().
            self._before_integration(context, thermodynamic_state)

            #specify nc integrator variables to report in verbose output
            self._integrator_keys_ = [
                'lambda', 'shadow_work', 'protocol_work', 'Eold', 'Enew'
            ]
            try:
                # Run dynamics.
                timer.start("{}: step({})".format(move_name, self.n_steps))
                test_int = integrator.getIntegrator(self.integrator_idx)
                RestorableOpenMMObject.restore_interface(test_int)
                print(test_int)
                try:
                    for key in self._integrator_keys_:
                        print(key, test_int.getGlobalVariableByName(key))
                except Exception:
                    pass

                integrator.step(self.n_steps)
                try:
                    for key in self._integrator_keys_:
                        print(key, test_int.getGlobalVariableByName(key))
                except Exception:
                    pass

            except Exception as e:
                print(e)
                # Catches particle positions becoming nan during integration.
                restart = True
            else:
                timer.stop("{}: step({})".format(move_name, self.n_steps))

                # We get also velocities here even if we don't need them because we
                # will recycle this State to update the sampler state object. This
                # way we won't need a second call to Context.getState().
                context_state = context.getState(
                    getPositions=True,
                    getVelocities=True,
                    getEnergy=True,
                    enforcePeriodicBox=thermodynamic_state.is_periodic)

                # Check for NaNs in energies.
                potential_energy = context_state.getPotentialEnergy()
                print('potential_energy', potential_energy)
                restart = np.isnan(
                    potential_energy.value_in_unit(potential_energy.unit))

            # Restart the move if we found NaNs.
            if restart:
                err_msg = (
                    'Potential energy is NaN after {} attempts of integration '
                    'with move {}'.format(attempt_counter,
                                          self.__class__.__name__))

                # If we are on our last chance before crash, try to re-initialize context
                if attempt_counter == self.n_restart_attempts - 1:
                    logger.error(
                        err_msg +
                        ' Trying to reinitialize Context as a last-resort restart attempt...'
                    )
                    context.reinitialize()
                    sampler_state.apply_to_context(context)
                    thermodynamic_state.apply_to_context(context)
                # If we have hit the number of restart attempts, raise an exception.
                elif attempt_counter == self.n_restart_attempts:
                    # Restore the context to the state right before the integration.
                    sampler_state.apply_to_context(context)
                    logger.error(err_msg)
                    raise IntegratorMoveError(err_msg, self, context)
                else:
                    logger.warning(err_msg + ' Attempting a restart...')
            else:
                break

        # Subclasses can read here info from the context to update internal statistics.
        self._after_integration(context, thermodynamic_state)

        # Updated sampler state.
        timer.start("{}: update sampler state".format(move_name))
        # This is an optimization around the fact that Collective Variables are not a part of the State,
        # but are a part of the Context. We do this call twice to minimize duplicating information fetched from
        # the State.
        # Update everything but the collective variables from the State object
        sampler_state.update_from_context(
            context_state, ignore_collective_variables=True)
        # Update only the collective variables from the Context
        sampler_state.update_from_context(
            context,
            ignore_positions=True,
            ignore_velocities=True,
            ignore_collective_variables=False)
        timer.stop("{}: update sampler state".format(move_name))

        #timer.report_timing()

    def __getstate__(self):
        serialization = super(NCMCMove, self).__getstate__()
        serialization['timestep'] = self.timestep
        #serialization['temperature'] = self.temperature
        serialization['collision_rate'] = self.collision_rate
        serialization.update(self.statistics)
        return serialization

    def __setstate__(self, serialization):
        super(NCMCMove, self).__setstate__(serialization)
        self.timestep = serialization['timestep']
        #self.temperature = serialization['temperature']
        self.collision_rate = serialization['collision_rate']
        self.statistics = serialization

    def _get_integrator(self, thermodynamic_state):
        """Implement BaseIntegratorMove._get_integrator()."""
        # Store lastly generated integrator to collect statistics.
        alchemical_functions = {'lambda_sterics': 'lambda'}
        #alchemical_functions={ 'lambda_sterics' : 'min(1, (1/0.3)*abs(lambda-0.5))',
        #                       'lambda_electrostatics' : 'step(0.2-lambda) - 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)' }
        ncmc_integrator = AlchemicalNonequilibriumLangevinIntegrator(
            alchemical_functions,
            splitting='H V R O R V H',
            #splitting='O { V R H R V } O',
            temperature=self.temperature,
            collision_rate=self.collision_rate,
            timestep=self.timestep,
            nsteps_neq=self.n_steps,
            measure_heat=True)
        #         ncmc_integrator.setRandomNumberSeed(0)
        #integrator = VelocityVerletIntegrator(timestep=self.timestep)
        integrator = LangevinIntegrator(
            temperature=self.temperature,
            timestep=self.timestep,
            collision_rate=self.collision_rate,
            measure_heat=True)
        # print(dir(integrator))

        integrator.setRandomNumberSeed(0)
        #integrator.pretty_print()

        compound_integrator = openmm.CompoundIntegrator()
        compound_integrator.addIntegrator(ncmc_integrator)
        compound_integrator.addIntegrator(integrator)
        return compound_integrator

    def _after_integration(self, context, thermodynamic_state):
        """Implement BaseIntegratorMove._after_integration()."""
        integrator = context.getIntegrator()
        integrator.setCurrentIntegrator(self.integrator_idx)
        #print(integrator)
        #print(dir(integrator))
        integrator = integrator.getIntegrator(self.integrator_idx)
        RestorableOpenMMObject.restore_interface(integrator)
        print(integrator)

        # Accumulate acceptance statistics.
        ncmc_global_variables = {
            integrator.getGlobalVariableName(index): index
            for index in range(integrator.getNumGlobalVariables())
        }
        print(ncmc_global_variables)
        #n_accepted = integrator.getGlobalVariable(ncmc_global_variables['naccept'])
        #n_proposed = integrator.getGlobalVariable(ncmc_global_variables['ntrials'])
        #self.n_accepted += n_accepted
        #self.n_proposed += n_proposed
        #integrator._measure_heat = True
        #print(integrator._measure_heat)
        #print(dir(integrator))
        integrator.reset()

        ncmc_global_variables = {
            integrator.getGlobalVariableName(index): index
            for index in range(integrator.getNumGlobalVariables())
        }
        print(ncmc_global_variables)


ncmc_move = NCMCMove(timestep=1.0 * unit.femtosecond, n_steps=500)
sampler = NCMCSampler(compound_state, sampler_state, move=ncmc_move)
sampler.minimize(max_iterations=0)
sampler.run(n_iterations=1, integrator_idx=0)
np.allclose(sampler.sampler_state.positions, ala.positions)

# In[44]:

sampler.run(n_iterations=2, integrator_idx=1)
