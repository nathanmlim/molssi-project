"""
Provides classes for setting up and running the BLUES simulation.

- `SystemFactory` : setup and modifying the OpenMM System prior to the simulation.
- `SimulationFactory` : generates the OpenMM Simulations from the System.
- `BLUESSimulation` : runs the NCMC+MD hybrid simulation.
- `MonteCarloSimulation` : runs a pure Monte Carlo simulation.

Authors: Samuel C. Gill
Contributors: Nathan M. Lim, Meghan Osato, David L. Mobley
"""

import logging
import math
import sys
import copy
import numpy as np
import parmed
from openmmtools import alchemy, mcmc, states
from simtk import openmm, unit
from simtk.openmm import app

from blues import utils
from blues.integrators import AlchemicalExternalLangevinIntegrator

finfo = np.finfo(np.float32)
rtol = finfo.precision
logger = logging.getLogger(__name__)


class GHMCMove(BaseIntegratorMove):
    """Generalized hybrid Monte Carlo (GHMC) Markov chain Monte Carlo move.

    This move uses generalized Hybrid Monte Carlo (GHMC), a form of Metropolized
    Langevin dynamics, to propagate the system.

    Parameters
    ----------
    timestep : simtk.unit.Quantity, optional
        The timestep to use for Langevin integration (time units, default
        is 1*simtk.unit.femtoseconds).
    collision_rate : simtk.unit.Quantity, optional
        The collision rate with fictitious bath particles (1/time units,
        default is 20/simtk.unit.picoseconds).
    n_steps : int, optional
        The number of integration timesteps to take each time the move
        is applied (default is 1000).
    context_cache : openmmtools.cache.ContextCache, optional
        The ContextCache to use for Context creation. If None, the global cache
        openmmtools.cache.global_context_cache is used (default is None).

    Attributes
    ----------
    timestep : simtk.unit.Quantity
        The timestep to use for Langevin integration (time units).
    collision_rate : simtk.unit.Quantity
        The collision rate with fictitious bath particles (1/time units).
    n_steps : int
        The number of integration timesteps to take each time the move
        is applied.
    context_cache : openmmtools.cache.ContextCache
        The ContextCache to use for Context creation. If None, the global
        cache openmmtools.cache.global_context_cache is used.
    n_accepted : int
        The number of accepted steps.
    n_proposed : int
        The number of attempted steps.
    fraction_accepted

    References
    ----------
    Lelievre T, Stoltz G, Rousset M. Free energy computations: A mathematical
    perspective. World Scientific, 2010.

    Examples
    --------
    First we need to create the thermodynamic state and the sampler
    state to propagate. Here we create an alanine dipeptide system
    in vacuum.

    >>> from simtk import unit
    >>> from openmmtools import testsystems
    >>> from openmmtools.states import ThermodynamicState, SamplerState
    >>> test = testsystems.AlanineDipeptideVacuum()
    >>> sampler_state = SamplerState(positions=test.positions)
    >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*unit.kelvin)

    Create a GHMC move with default parameters.

    >>> move = GHMCMove()

    or create a GHMC move with specified parameters.

    >>> move = GHMCMove(timestep=0.5*unit.femtoseconds,
    ...                 collision_rate=20.0/unit.picoseconds, n_steps=10)

    Perform one update of the sampler state. The sampler state is updated
    with the new state.

    >>> move.apply(thermodynamic_state, sampler_state)
    >>> np.allclose(sampler_state.positions, test.positions)
    False

    The same move can be applied to a different state, here an ideal gas.

    >>> test = testsystems.IdealGas()
    >>> sampler_state = SamplerState(positions=test.positions)
    >>> thermodynamic_state = ThermodynamicState(system=test.system,
    ...                                          temperature=298*unit.kelvin)
    >>> move.apply(thermodynamic_state, sampler_state)
    >>> np.allclose(sampler_state.positions, test.positions)
    False

    """

    def __init__(self, timestep=1.0 * unit.femtosecond, collision_rate=20.0 / unit.picoseconds, n_steps=1000,
                 **kwargs):
        super(GHMCMove, self).__init__(n_steps=n_steps, **kwargs)
        self.timestep = timestep
        self.collision_rate = collision_rate
        self.n_accepted = 0  # Number of accepted steps.
        self.n_proposed = 0  # Number of attempted steps.

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

    def apply(self, thermodynamic_state, sampler_state):
        """Apply the GHMC MCMC move.

        This modifies the given sampler_state. The temperature of the
        thermodynamic state is used.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
           The thermodynamic state to use when applying the MCMC move.
        sampler_state : openmmtools.states.SamplerState
           The sampler state to apply the move to. This is modified.

        """
        # Explicitly implemented just to have more specific docstring.
        super(GHMCMove, self).apply(thermodynamic_state, sampler_state)

    def __getstate__(self):
        serialization = super(GHMCMove, self).__getstate__()
        serialization['timestep'] = self.timestep
        serialization['collision_rate'] = self.collision_rate
        serialization.update(self.statistics)
        return serialization

    def __setstate__(self, serialization):
        super(GHMCMove, self).__setstate__(serialization)
        self.timestep = serialization['timestep']
        self.collision_rate = serialization['collision_rate']
        self.statistics = serialization

    def _get_integrator(self, thermodynamic_state):
        """Implement BaseIntegratorMove._get_integrator()."""
        # Store lastly generated integrator to collect statistics.
        return integrators.GHMCIntegrator(
            temperature=thermodynamic_state.temperature, collision_rate=self.collision_rate, timestep=self.timestep)

    def _after_integration(self, context, thermodynamic_state):
        """Implement BaseIntegratorMove._after_integration()."""
        integrator = context.getIntegrator()

        # Accumulate acceptance statistics.
        ghmc_global_variables = {
            integrator.getGlobalVariableName(index): index for index in range(integrator.getNumGlobalVariables())
        }
        n_accepted = integrator.getGlobalVariable(ghmc_global_variables['naccept'])
        n_proposed = integrator.getGlobalVariable(ghmc_global_variables['ntrials'])
        self.n_accepted += n_accepted
        self.n_proposed += n_proposed


class NCMCSampler(object):

    def __init__(self, thermodynamic_state, sampler_state, move):
        # Make a deep copy of the state so that initial state is unchanged.
        self.thermodynamic_state = copy.deepcopy(thermodynamic_state)
        self.sampler_state = copy.deepcopy(sampler_state)
        self.move = move

    def run(self, n_iterations=1):
        """
        Run the sampler for a specified number of iterations.

        Parameters
        ----------
        n_iterations : int
            Number of iterations of the sampler to run.

        """
        # Apply move for n_iterations.
        for iteration in range(n_iterations):
            self.move.apply(self.thermodynamic_state, self.sampler_state)

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
        context, integrator = context_cache.get_context(self.thermodynamic_state, integrator)
        self.sampler_state.apply_to_context(context)
        logger.debug("LocalEnergyMinimizer: platform is %s" % context.getPlatform().getName())
        logger.debug("Minimizing with tolerance %s and %d max. iterations." % (tolerance, max_iterations))
        timer.stop("Context request")

        timer.start("LocalEnergyMinimizer minimize")
        openmm.LocalEnergyMinimizer.minimize(context, tolerance, max_iterations)
        timer.stop("LocalEnergyMinimizer minimize")

        # Retrieve data.
        self.sampler_state.update_from_context(context)

        #timer.report_timing()


class BLUESSimulation(object):
    """BLUESSimulation class provides methods to execute the NCMC+MD
    simulation.

    Parameters
    ----------
    simulations : blues.simulation.SimulationFactory object
        SimulationFactory Object which carries the 3 required
        OpenMM Simulation objects (MD, NCMC, ALCH) required to run BLUES.
    config : dict
        Dictionary of parameters for configuring the OpenMM Simulations
        If None, will search for configuration parameters on the `simulations`
        object.

    Examples
    --------
    Create our SimulationFactory object and run `BLUESSimulation`

    >>> sim_cfg = { 'platform': 'OpenCL',
                    'properties' : { 'OpenCLPrecision': 'single',
                                     'OpenCLDeviceIndex' : 2},
                    'nprop' : 1,
                    'propLambda' : 0.3,
                    'dt' : 0.001 * unit.picoseconds,
                    'friction' : 1 * 1/unit.picoseconds,
                    'temperature' : 100 * unit.kelvin,
                    'nIter': 1,
                    'nstepsMD': 10,
                    'nstepsNC': 10,}
    >>> simulations = SimulationFactory(systems, ligand_mover, sim_cfg)
    >>> blues = BLUESSimulation(simulations)
    >>> blues.run()

    """

    def __init__(self, simulations, config=None):
        self._move_engine = simulations._move_engine
        self._md_sim = simulations.md
        self._alch_sim = simulations.alch
        self._ncmc_sim = simulations.ncmc

        # Check if configuration has been specified in `SimulationFactory` object
        if not config:
            if hasattr(simulations, 'config'):
                self._config = simulations.config
        else:
            #Otherwise take specified config
            self._config = config
        if self._config:
            self._printSimulationTiming()

        self.accept = 0
        self.reject = 0
        self.acceptRatio = 0
        self.currentIter = 0

        #Dict to keep track of each simulation state before/after each iteration
        self.stateTable = {'md': {'state0': {}, 'state1': {}}, 'ncmc': {'state0': {}, 'state1': {}}}

        #specify nc integrator variables to report in verbose output
        self._integrator_keys_ = ['lambda', 'shadow_work', 'protocol_work', 'Eold', 'Enew']

        self._state_keys = {
            'getPositions': True,
            'getVelocities': True,
            'getForces': False,
            'getEnergy': True,
            'getParameters': True,
            'enforcePeriodicBox': True
        }

    @classmethod
    def getStateFromContext(cls, context, state_keys):
        """Gets the State information from the given context and
        list of state_keys to query it with.

        Returns the state data as a dict.

        Parameters
        ----------
        context : openmm.Context
            Context of the OpenMM Simulation to query.
        state_keys : list
            Default: [ positions, velocities, potential_energy, kinetic_energy ]
            A list that defines what information to get from the context State.

        Returns
        -------
        stateinfo : dict
            Current positions, velocities, energies and box vectors of the context.
        """

        stateinfo = {}
        state = context.getState(**state_keys)
        stateinfo['positions'] = state.getPositions(asNumpy=True)
        stateinfo['velocities'] = state.getVelocities(asNumpy=True)
        stateinfo['potential_energy'] = state.getPotentialEnergy()
        stateinfo['kinetic_energy'] = state.getKineticEnergy()
        stateinfo['box_vectors'] = state.getPeriodicBoxVectors()
        return stateinfo

    @classmethod
    def getIntegratorInfo(cls,
                          ncmc_integrator,
                          integrator_keys=['lambda', 'shadow_work', 'protocol_work', 'Eold', 'Enew']):
        """Returns a dict of alchemical/ncmc-swtiching data from querying the the NCMC
        integrator.

        Parameters
        ----------
        ncmc_integrator : openmm.Context.Integrator
            The integrator from the NCMC Context
        integrator_keys : list
            list containing strings of the values to get from the integrator.
            Default = ['lambda', 'shadow_work', 'protocol_work', 'Eold', 'Enew','Epert']

        Returns
        -------
        integrator_info : dict
            Work values and energies from the NCMC integrator.
        """
        integrator_info = {}
        for key in integrator_keys:
            integrator_info[key] = ncmc_integrator.getGlobalVariableByName(key)
        return integrator_info

    @classmethod
    def setContextFromState(cls, context, state, box=True, positions=True, velocities=True):
        """Update a given Context from the given State.

        Parameters
        ----------
        context : openmm.Context
            The Context to be updated from the given State.
        state : openmm.State
            The current state (box_vectors, positions, velocities) of the
            Simulation to update the given context.

        Returns
        -------
        context : openmm.Context
            The updated Context whose box_vectors, positions, and velocities
            have been updated.
        """
        # Replace ncmc data from the md context
        if box:
            context.setPeriodicBoxVectors(*state['box_vectors'])
        if positions:
            context.setPositions(state['positions'])
        if velocities:
            context.setVelocities(state['velocities'])
        return context

    def _printSimulationTiming(self):
        """Prints the simulation timing and related information."""

        dt = self._config['dt'].value_in_unit(unit.picoseconds)
        nIter = self._config['nIter']
        nprop = self._config['nprop']
        propLambda = self._config['propLambda']
        propSteps = self._config['propSteps']
        nstepsNC = self._config['nstepsNC']
        nstepsMD = self._config['nstepsMD']

        force_eval = nIter * (propSteps + nstepsMD)
        time_ncmc_iter = propSteps * dt
        time_ncmc_total = time_ncmc_iter * nIter
        time_md_iter = nstepsMD * dt
        time_md_total = time_md_iter * nIter
        time_iter = time_ncmc_iter + time_md_iter
        time_total = time_iter * nIter

        msg = 'Total BLUES Simulation Time = %s ps (%s ps/Iter)\n' % (time_total, time_iter)
        msg += 'Total Force Evaluations = %s \n' % force_eval
        msg += 'Total NCMC time = %s ps (%s ps/iter)\n' % (time_ncmc_total, time_ncmc_iter)

        # Calculate number of lambda steps inside/outside region with extra propgation steps
        steps_in_prop = int(nprop * (2 * math.floor(propLambda * nstepsNC)))
        steps_out_prop = int((2 * math.ceil((0.5 - propLambda) * nstepsNC)))

        prop_lambda_window = self._ncmc_sim.context._integrator._prop_lambda
        # prop_range = round(prop_lambda_window[1] - prop_lambda_window[0], 4)
        if propSteps != nstepsNC:
            msg += '\t%s lambda switching steps within %s total propagation steps.\n' % (nstepsNC, propSteps)
            msg += '\tExtra propgation steps between lambda [%s, %s]\n' % (prop_lambda_window[0],
                                                                           prop_lambda_window[1])
            msg += '\tLambda: 0.0 -> %s = %s propagation steps\n' % (prop_lambda_window[0], int(steps_out_prop / 2))
            msg += '\tLambda: %s -> %s = %s propagation steps\n' % (prop_lambda_window[0], prop_lambda_window[1],
                                                                    steps_in_prop)
            msg += '\tLambda: %s -> 1.0 = %s propagation steps\n' % (prop_lambda_window[1], int(steps_out_prop / 2))

        msg += 'Total MD time = %s ps (%s ps/iter)\n' % (time_md_total, time_md_iter)

        #Get trajectory frame interval timing for BLUES simulation
        if 'md_trajectory_interval' in self._config.keys():
            frame_iter = nstepsMD / self._config['md_trajectory_interval']
            timetraj_frame = (time_ncmc_iter + time_md_iter) / frame_iter
            msg += 'Trajectory Interval = %s ps/frame (%s frames/iter)' % (timetraj_frame, frame_iter)

        logger.info(msg)

    def _setStateTable(self, simkey, stateidx, stateinfo):
        """Updates `stateTable` (dict) containing:  Positions, Velocities, Potential/Kinetic energies
        of the state before and after a NCMC step or iteration.

        Parameters
        ----------
        simkey : str (key: 'md', 'ncmc', 'alch')
            Key corresponding to the simulation.
        stateidx : str (key: 'state0' or 'state1')
            Key corresponding to the state information being stored.
        stateinfo : dict
            Dictionary containing the State information.
        """
        self.stateTable[simkey][stateidx] = stateinfo

    def _syncStatesMDtoNCMC(self):
        """Retrieves data on the current State of the MD context to
        replace the box vectors, positions, and velocties in the NCMC context.
        """
        # Retrieve MD state from previous iteration
        md_state0 = self.getStateFromContext(self._md_sim.context, self._state_keys)
        self._setStateTable('md', 'state0', md_state0)

        # Sync MD state to the NCMC context
        self._ncmc_sim.context = self.setContextFromState(self._ncmc_sim.context, md_state0)

    def _stepNCMC(self, nstepsNC, moveStep, move_engine=None):
        """Advance the NCMC simulation.

        Parameters
        ----------
        nstepsNC : int
            The number of NCMC switching steps to advance by.
        moveStep : int
            The step number to perform the chosen move, which should be half
            the number of nstepsNC.
        move_engine : blues.moves.MoveEngine
            The object that executes the chosen move.

        """

        logger.info('Advancing %i NCMC switching steps...' % (nstepsNC))
        # Retrieve NCMC state before proposed move
        ncmc_state0 = self.getStateFromContext(self._ncmc_sim.context, self._state_keys)
        self._setStateTable('ncmc', 'state0', ncmc_state0)

        #choose a move to be performed according to move probabilities
        #TODO: will have to change to work with multiple alch region
        if not move_engine: move_engine = self._move_engine
        self._ncmc_sim.currentIter = self.currentIter
        move_engine.selectMove()

        lastStep = nstepsNC - 1
        for step in range(int(nstepsNC)):
            try:
                #Attempt anything related to the move before protocol is performed
                if not step:
                    self._ncmc_sim.context = move_engine.selected_move.beforeMove(self._ncmc_sim.context)

                # Attempt selected MoveEngine Move at the halfway point
                #to ensure protocol is symmetric
                if step == moveStep:
                    if hasattr(logger, 'report'):
                        logger.info = logger.report
                    #Do move
                    logger.info('Performing %s...' % move_engine.move_name)
                    self._ncmc_sim.context = move_engine.runEngine(self._ncmc_sim.context)

                # Do 1 NCMC step with the integrator
                self._ncmc_sim.step(1)

                #Attempt anything related to the move after protocol is performed
                if step == lastStep:
                    self._ncmc_sim.context = move_engine.selected_move.afterMove(self._ncmc_sim.context)

            except Exception as e:
                logger.error(e)
                move_engine.selected_move._error(self._ncmc_sim.context)
                break

        # ncmc_state1 stores the state AFTER a proposed move.
        ncmc_state1 = self.getStateFromContext(self._ncmc_sim.context, self._state_keys)
        self._setStateTable('ncmc', 'state1', ncmc_state1)

    def _computeAlchemicalCorrection(self):
        """Computes the alchemical correction term from switching between the NCMC
        and MD potentials."""
        # Retrieve the MD/NCMC state before the proposed move.
        md_state0_PE = self.stateTable['md']['state0']['potential_energy']
        ncmc_state0_PE = self.stateTable['ncmc']['state0']['potential_energy']

        # Retreive the NCMC state after the proposed move.
        ncmc_state1 = self.stateTable['ncmc']['state1']
        ncmc_state1_PE = ncmc_state1['potential_energy']

        # Set the box_vectors and positions in the alchemical simulation to after the proposed move.
        self._alch_sim.context = self.setContextFromState(self._alch_sim.context, ncmc_state1, velocities=False)

        # Retrieve potential_energy for alch correction
        alch_PE = self._alch_sim.context.getState(getEnergy=True).getPotentialEnergy()
        correction_factor = (ncmc_state0_PE - md_state0_PE + alch_PE - ncmc_state1_PE) * (
            -1.0 / self._ncmc_sim.context._integrator.kT)

        return correction_factor

    def _acceptRejectMove(self, write_move=False):
        """Choose to accept or reject the proposed move based
        on the acceptance criterion.

        Parameters
        ----------
        write_move : bool, default=False
            If True, writes the proposed NCMC move to a PDB file.
        """
        work_ncmc = self._ncmc_sim.context._integrator.getLogAcceptanceProbability(self._ncmc_sim.context)
        randnum = math.log(np.random.random())

        # Compute correction if work_ncmc is not NaN
        if not np.isnan(work_ncmc):
            correction_factor = self._computeAlchemicalCorrection()
            logger.debug(
                'NCMCLogAcceptanceProbability = %.6f + Alchemical Correction = %.6f' % (work_ncmc, correction_factor))
            work_ncmc = work_ncmc + correction_factor

        if work_ncmc > randnum:
            self.accept += 1
            logger.info('NCMC MOVE ACCEPTED: work_ncmc {} > randnum {}'.format(work_ncmc, randnum))

            # If accept move, sync NCMC state to MD context
            ncmc_state1 = self.stateTable['ncmc']['state1']
            self._md_sim.context = self.setContextFromState(self._md_sim.context, ncmc_state1, velocities=False)

            if write_move:
                utils.saveSimulationFrame(self._md_sim, '{}acc-it{}.pdb'.format(self._config['outfname'],
                                                                                self.currentIter))

        else:
            self.reject += 1
            logger.info('NCMC MOVE REJECTED: work_ncmc {} < {}'.format(work_ncmc, randnum))

            # If reject move, do nothing,
            # NCMC simulation be updated from MD Simulation next iteration.

            # Potential energy should be from last MD step in the previous iteration
            md_state0 = self.stateTable['md']['state0']
            md_PE = self._md_sim.context.getState(getEnergy=True).getPotentialEnergy()
            if not math.isclose(md_state0['potential_energy']._value, md_PE._value, rel_tol=float('1e-%s' % rtol)):
                logger.error(
                    'Last MD potential energy %s != Current MD potential energy %s. Potential energy should match the prior state.'
                    % (md_state0['potential_energy'], md_PE))
                sys.exit(1)

    def _resetSimulations(self, temperature=None):
        """At the end of each iteration:

        1. Reset the step number in the NCMC context/integrator
        2. Set the velocities to random values chosen from a Boltzmann distribution at a given `temperature`.

        Parameters
        ----------
        temperature : float
            The target temperature for the simulation.

        """
        if not temperature:
            temperature = self._md_sim.context._integrator.getTemperature()

        self._ncmc_sim.currentStep = 0
        self._ncmc_sim.context._integrator.reset()

        #Reinitialize velocities, preserving detailed balance?
        self._md_sim.context.setVelocitiesToTemperature(temperature)

    def _stepMD(self, nstepsMD):
        """Advance the MD simulation.

        Parameters
        ----------
        nstepsMD : int
            The number of steps to advance the MD simulation.
        """
        logger.info('Advancing %i MD steps...' % (nstepsMD))
        self._md_sim.currentIter = self.currentIter
        # Retrieve MD state before proposed move
        # Helps determine if previous iteration placed ligand poorly
        md_state0 = self.stateTable['md']['state0']

        for md_step in range(int(nstepsMD)):
            try:
                self._md_sim.step(1)
            except Exception as e:
                logger.error(e, exc_info=True)
                logger.error('potential energy before NCMC: %s' % md_state0['potential_energy'])
                logger.error('kinetic energy before NCMC: %s' % md_state0['kinetic_energy'])
                #Write out broken frame
                utils.saveSimulationFrame(self._md_sim,
                                          'MD-fail-it%s-md%i.pdb' % (self.currentIter, self._md_sim.currentStep))
                sys.exit(1)

    def run(self, nIter=0, nstepsNC=0, moveStep=0, nstepsMD=0, temperature=300, write_move=False, **config):
        """Executes the BLUES engine to iterate over the actions:
        Perform NCMC simulation, perform proposed move, accepts/rejects move,
        then performs the MD simulation from the NCMC state, niter number of times.
        **Note:** If the parameters are not given explicitly, will look for the parameters
        in the provided configuration on the `SimulationFactory` object.

        Parameters
        ----------
        nIter : int, default = None
            Number of iterations of NCMC+MD to perform.
        nstepsNC : int
            The number of NCMC switching steps to advance by.
        moveStep : int
            The step number to perform the chosen move, which should be half
            the number of nstepsNC.
        nstepsMD : int
            The number of steps to advance the MD simulation.
        temperature : float
            The target temperature for the simulation.
        write_move : bool, default=False
            If True, writes the proposed NCMC move to a PDB file.

        """
        if not nIter: nIter = self._config['nIter']
        if not nstepsNC: nstepsNC = self._config['nstepsNC']
        if not nstepsMD: nstepsMD = self._config['nstepsMD']
        if not moveStep: moveStep = self._config['moveStep']

        logger.info('Running %i BLUES iterations...' % (nIter))
        for N in range(int(nIter)):
            self.currentIter = N
            logger.info('BLUES Iteration: %s' % N)
            self._syncStatesMDtoNCMC()
            self._stepNCMC(nstepsNC, moveStep)
            self._acceptRejectMove(write_move)
            self._resetSimulations(temperature)
            self._stepMD(nstepsMD)

        # END OF NITER
        self.acceptRatio = self.accept / float(nIter)
        logger.info('Acceptance Ratio: %s' % self.acceptRatio)
        logger.info('nIter: %s ' % nIter)


class MonteCarloSimulation(BLUESSimulation):
    """Simulation class provides the functions that perform the MonteCarlo run.

    Parameters
    ----------
        simulations : SimulationFactory
            Contains 3 required OpenMM Simulationobjects
        config : dict, default = None
            Dict with configuration info.
    """

    def __init__(self, simulations, config=None):
        super(MonteCarloSimulation, self).__init__(simulations, config)

    def _stepMC_(self):
        """Function that performs the MC simulation.
        """

        #choose a move to be performed according to move probabilities
        self._move_engine.selectMove()
        #change coordinates according to Moves in MoveEngine
        new_context = self._move_engine.runEngine(self._md_sim.context)
        md_state1 = self.getStateFromContext(new_context, self._state_keys)
        self._setStateTable('md', 'state1', md_state1)

    def _acceptRejectMove(self, temperature=None):
        """Function that chooses to accept or reject the proposed move.
        """
        md_state0 = self.stateTable['md']['state0']
        md_state1 = self.stateTable['md']['state1']
        work_mc = (md_state1['potential_energy'] - md_state0['potential_energy']) * (
            -1.0 / self._ncmc_sim.context._integrator.kT)
        randnum = math.log(np.random.random())

        if work_mc > randnum:
            self.accept += 1
            logger.info('MC MOVE ACCEPTED: work_mc {} > randnum {}'.format(work_mc, randnum))
            self._md_sim.context.setPositions(md_state1['positions'])
        else:
            self.reject += 1
            logger.info('MC MOVE REJECTED: work_mc {} < {}'.format(work_mc, randnum))
            self._md_sim.context.setPositions(md_state0['positions'])
        self._md_sim.context.setVelocitiesToTemperature(temperature)

    def run(self, nIter=0, mc_per_iter=0, nstepsMD=0, temperature=300, write_move=False):
        """Function that runs the BLUES engine to iterate over the actions:
        perform proposed move, accepts/rejects move,
        then performs the MD simulation from the accepted or rejected state.

        Parameters
        ----------
        nIter : None or int, optional default = None
            The number of iterations to perform. If None, then
            uses the nIter specified in the opt dictionary when
            the Simulation class was created.
        mc_per_iter : int, default = 1
            Number of Monte Carlo iterations.
        nstepsMD : int, default = None
            Number of steps the MD simulation will advance
        write_move : bool, default = False
            Writes the move if True
        """
        if not nIter: nIter = self._config['nIter']
        if not nstepsMD: nstepsMD = self._config['nstepsMD']
        #controls how many mc moves are performed during each iteration
        if not mc_per_iter: mc_per_iter = self._config['mc_per_iter']

        self._syncStatesMDtoNCMC()
        for N in range(nIter):
            self.currentIter = N
            logger.info('MonteCarlo Iteration: %s' % N)
            for i in range(mc_per_iter):
                self._syncStatesMDtoNCMC()
                self._stepMC_()
                self._acceptRejectMove(temperature)
            self._stepMD(nstepsMD)
