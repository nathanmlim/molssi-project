from simtk import unit
from openmmtools import testsystems, cache
from openmmtools.mcmc import GHMCMove, MCMCSampler, MCRotationMove, BaseIntegratorMove, IntegratorMoveError
from openmmtools.states import ThermodynamicState, SamplerState, CompoundThermodynamicState
from simtk.openmm import CompoundIntegrator
from openmmtools import alchemy
from simtk import unit
from openmmtools.utils import RestorableOpenMMObject
from openmmtools.integrators import *
import numpy as np
import copy, sys, time, os, math
import logging
from simtk import openmm
import parmed
from openmmtools.utils import SubhookedABCMeta, Timer, RestorableOpenMMObject
from openmmtools import testsystems, alchemy
import netCDF4 as nc
from blues.integrators import AlchemicalExternalLangevinIntegrator
import mdtraj
import argparse

finfo = np.finfo(np.float32)
rtol = finfo.precision
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger("openmmtools.alchemy").setLevel(logging.ERROR)
np.random.RandomState(seed=3134)
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.DEBUG, stream=sys.stdout)


class DummySimulation(object):

    def __init__(self, integrator, system, topology):
        self.integrator = integrator
        self.topology = topology
        self.system = system


class LangevinDynamicsMove(BaseIntegratorMove):
    """Langevin dynamics segment as a (pseudo) Monte Carlo move.

    Parameters
    ----------
    timestep : simtk.unit.Quantity, optional
        The timestep to use for Langevin integration
        (time units, default is 1*simtk.unit.femtosecond).
    collision_rate : simtk.unit.Quantity, optional
        The collision rate with fictitious bath particles
        (1/time units, default is 10/simtk.unit.picoseconds).
    n_steps : int, optional
        The number of integration timesteps to take each time the
        move is applied (default is 1000).
    reassign_velocities : bool, optional
        If True, the velocities will be reassigned from the Maxwell-Boltzmann
        distribution at the beginning of the move (default is False).
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
    reassign_velocities : bool
        If True, the velocities will be reassigned from the Maxwell-Boltzmann
        distribution at the beginning of the move.
    context_cache : openmmtools.cache.ContextCache
        The ContextCache to use for Context creation. If None, the global
        cache openmmtools.cache.global_context_cache is used.

    """

    def __init__(self,
                 timestep=1.0 * unit.femtosecond,
                 collision_rate=10.0 / unit.picoseconds,
                 n_steps=1000,
                 reassign_velocities=False,
                 **kwargs):
        super(LangevinDynamicsMove, self).__init__(n_steps=n_steps, reassign_velocities=reassign_velocities, **kwargs)
        self.timestep = timestep
        self.collision_rate = collision_rate

    def _create_dummy_simulation(self, integrator, system, topology):
        """
        Generate a dummy Simulation object because the Reporter
        expects an `openmm.Simulation` object in order to report information
        from it's respective attached integrator/
        """
        return DummySimulation(integrator, system, topology)

    def _before_integration(self, context, integrator, thermodynamic_state):
        """Execute code after Context creation and before integration."""

        return self._create_dummy_simulation(integrator, thermodynamic_state.get_system(),
                                             thermodynamic_state.topology)

    def apply(self, thermodynamic_state, sampler_state, integrator, reporter):
        """Apply the Langevin dynamics MCMC move.

        This modifies the given sampler_state. The temperature of the
        thermodynamic state is used in Langevin dynamics.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
           The thermodynamic state to use to propagate dynamics.
        sampler_state : openmmtools.states.SamplerState
           The sampler state to apply the move to. This is modified.

        """
        move_name = self.__class__.__name__  # shortcut
        timer = Timer()

        # Check if we have to use the global cache.
        if self.context_cache is None:
            context_cache = cache.global_context_cache
        else:
            context_cache = self.context_cache

        # Create context.
        timer.start("{}: Context request".format(move_name))
        context, integrator = context_cache.get_context(thermodynamic_state, integrator)
        print(integrator)

        timer.stop("{}: Context request".format(move_name))
        logger.debug("{}: {} \t Context obtained, platform is {}".format(move_name, integrator,
                                                                         context.getPlatform().getName()))

        # Perform the integration.
        for attempt_counter in range(self.n_restart_attempts + 1):

            # If we reassign velocities, we can ignore the ones in sampler_state.
            sampler_state.apply_to_context(context, ignore_velocities=False)

            # Subclasses may implement _before_integration().
            simulation = self._before_integration(context, integrator, thermodynamic_state)
            init_context_state = context.getState(
                getPositions=True,
                getVelocities=True,
                getEnergy=True,
                enforcePeriodicBox=thermodynamic_state.is_periodic)

            try:
                # Run dynamics.
                timer.start("{}: step({})".format(move_name, self.n_steps))

                #NML: Do in 1 steps for debugging
                for n in range(self.n_steps + 1):
                    integrator.step(1)

                    # Get the context state for reporting
                    context_state = context.getState(
                        getPositions=True,
                        getVelocities=True,
                        getEnergy=True,
                        enforcePeriodicBox=thermodynamic_state.is_periodic)
                    #print('IntgrParameters:', 'Step:', step, 'PE:', context_state.getPotentialEnergy(), 'Work:', protocol_work, 'Lambda: ', alch_lambda, 'Lambda Step: ', lambda_step)
                    #print('\t'+'ContextParameters:', lambda_step, 'Sterics:', lambda_sterics, 'Elec:', lambda_electrostatics)

                    if step == n_steps:
                        break

                    else:
                        # Get the context state for reporting
                        context_state = context.getState(
                            getPositions=True,
                            getVelocities=True,
                            getEnergy=True,
                            enforcePeriodicBox=thermodynamic_state.is_periodic)
                        #print('Pos:', context_state.getPositions()[0])
                        #print('Step:', n, 'PE:', context_state.getPotentialEnergy())

                    if n % reporter.reportInterval == 0:
                        print('NCMC---IntgrParameters:', 'Step:', step, 'PE:', context_state.getPotentialEnergy(),
                              'Work:', protocol_work, 'Lambda: ', alch_lambda, 'Lambda Step: ', lambda_step)

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
                restart = np.isnan(potential_energy.value_in_unit(potential_energy.unit))

            # Restart the move if we found NaNs.
            if restart:
                err_msg = ('Potential energy is NaN after {} attempts of integration '
                           'with move {}'.format(attempt_counter, self.__class__.__name__))

                # If we are on our last chance before crash, try to re-initialize context
                if attempt_counter == self.n_restart_attempts - 1:
                    logger.error(err_msg + ' Trying to reinitialize Context as a last-resort restart attempt...')
                    context.reinitialize()
                    sampler_state.apply_to_context(context, ignore_velocities=self.reassign_velocities)
                    thermodynamic_state.apply_to_context(context)
                # If we have hit the number of restart attempts, raise an exception.
                elif attempt_counter == self.n_restart_attempts:
                    # Restore the context to the state right before the integration.
                    sampler_state.apply_to_context(context, ignore_velocities=False)

                    logger.error(err_msg)
                    print(IntegratorMoveError(err_msg, self, context))
                    break
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
        sampler_state.update_from_context(context_state, ignore_collective_variables=True, ignore_velocities=True)
        # Update only the collective variables from the Context
        sampler_state.update_from_context(
            context, ignore_positions=True, ignore_velocities=True, ignore_collective_variables=False)

        timer.stop("{}: update sampler state".format(move_name))
        final_context_state = context.getState(
            getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=thermodynamic_state.is_periodic)

    def __getstate__(self):
        serialization = super(LangevinDynamicsMove, self).__getstate__()
        serialization['timestep'] = self.timestep
        serialization['collision_rate'] = self.collision_rate
        return serialization

    def __setstate__(self, serialization):
        super(LangevinDynamicsMove, self).__setstate__(serialization)
        self.timestep = serialization['timestep']
        self.collision_rate = serialization['collision_rate']

    def _get_integrator(self, thermodynamic_state):
        """Implement BaseIntegratorMove._get_integrator()."""
        return openmm.LangevinIntegrator(thermodynamic_state.temperature, self.collision_rate, self.timestep)


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
        self.random_state = np.random.RandomState(1)

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

    def _create_dummy_simulation(self, integrator, system, topology):
        """
        Generate a dummy Simulation object because the Reporter
        expects an `openmm.Simulation` object in order to report information
        from it's respective attached integrator/
        """
        return DummySimulation(integrator, system, topology)

    def _before_integration(self, context, integrator, thermodynamic_state):
        """Execute code after Context creation and before integration."""

        return self._create_dummy_simulation(integrator, thermodynamic_state.get_system(),
                                             thermodynamic_state.topology)

    def getMasses(self, atom_subset, topology):
        """
        Returns a list of masses of the specified ligand atoms.
        Parameters
        ----------
        topology: parmed.Topology
            ParmEd topology object containing atoms of the system.
        Returns
        -------
        masses: 1xn numpy.array * simtk.unit.dalton
            array of masses of len(self.atom_indices), denoting
            the masses of the atoms in self.atom_indices
        totalmass: float * simtk.unit.dalton
            The sum of the mass found in masses
        """
        atoms = [list(topology.atoms())[i] for i in atom_subset]
        masses = unit.Quantity(np.zeros([int(len(atoms)), 1], np.float32), unit.dalton)
        for idx, atom in enumerate(atoms):
            masses[idx] = atom.element._mass
        totalmass = masses.sum()
        return masses, totalmass

    def getCenterOfMass(self, positions, masses):
        """Returns the calculated center of mass of the ligand as a numpy.array
        Parameters
        ----------
        positions: nx3 numpy array * simtk.unit compatible with simtk.unit.nanometers
            ParmEd positions of the atoms to be moved.
        masses : numpy.array
            numpy.array of particle masses
        Returns
        -------
        center_of_mass: numpy array * simtk.unit compatible with simtk.unit.nanometers
            1x3 numpy.array of the center of mass of the given positions
        """
        coordinates = np.asarray(positions._value, np.float32)
        center_of_mass = parmed.geometry.center_of_mass(coordinates, masses) * positions.unit
        return center_of_mass

    def rotate(self, context, thermodynamic_state, sampler_state):
        print('Performing rotation...')
        atom_subset = thermodynamic_state.alchemical_atoms
        context_state = context.getState(
            getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=thermodynamic_state.is_periodic)
        all_positions = context_state.getPositions(asNumpy=True)
        sub_positions = all_positions[atom_subset]

        masses, totalmass = self.getMasses(atom_subset, thermodynamic_state.topology)
        center_of_mass = self.getCenterOfMass(sub_positions, masses)
        reduced_pos = sub_positions - center_of_mass

        # Define random rotational move on the ligand
        rand_quat = mdtraj.utils.uniform_quaternion(size=None, random_state=self.random_state)
        rand_rotation_matrix = mdtraj.utils.rotation_matrix_from_quaternion(rand_quat)
        #multiply lig coordinates by rot matrix and add back COM translation from origin
        rot_move = np.dot(reduced_pos, rand_rotation_matrix) * sub_positions.unit + center_of_mass

        # Update ligand positions in sampler_state and context
        for index, atomidx in enumerate(atom_subset):
            all_positions[atomidx] = rot_move[index]
        sampler_state.positions = all_positions
        sampler_state.apply_to_context(context, ignore_velocities=True)

        return context

    def apply(self, thermodynamic_state, sampler_state, integrator, reporter, alch=True, ignore_velocities=True):
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
            context_cache = self.context_cache

        # Create context.
        timer.start("{}: Context request".format(move_name))
        context, integrator = context_cache.get_context(thermodynamic_state, integrator)
        print(integrator)
        timer.stop("{}: Context request".format(move_name))
        logger.debug("{}: {} \t Context obtained, platform is {}".format(move_name, integrator,
                                                                         context.getPlatform().getName()))

        rotation_step = int(n_steps / 2)
        # Perform the integration.
        for attempt_counter in range(self.n_restart_attempts + 1):

            # If we reassign velocities, we can ignore the ones in sampler_state.
            sampler_state.apply_to_context(context, ignore_velocities=False)

            # Subclasses may implement _before_integration().
            simulation = self._before_integration(context, integrator, thermodynamic_state)
            init_context_state = context.getState(
                getPositions=True,
                getVelocities=True,
                getEnergy=True,
                enforcePeriodicBox=thermodynamic_state.is_periodic)
            try:
                # Run dynamics.
                timer.start("{}: step({})".format(move_name, self.n_steps))

                #NML: Do in 1 steps for debugging
                for n in range(self.n_steps + 1):
                    integrator.step(1)

                    step = integrator.getGlobalVariableByName('step')
                    alch_lambda = integrator.getGlobalVariableByName('lambda')
                    lambda_step = integrator.getGlobalVariableByName('lambda_step')
                    protocol_work = integrator.getGlobalVariableByName('protocol_work')

                    thermodynamic_state.set_alchemical_variable('lambda', alch_lambda)
                    thermodynamic_state.apply_to_context(context)

                    #lambda_sterics = context.getParameter('lambda_sterics')
                    #lambda_electrostatics = context.getParameter('lambda_electrostatics')

                    if n == rotation_step:
                        context = self.rotate(context, thermodynamic_state, sampler_state)

                    # Get the context state for reporting
                    context_state = context.getState(
                        getPositions=True,
                        getVelocities=True,
                        getEnergy=True,
                        enforcePeriodicBox=thermodynamic_state.is_periodic)
                    #print('IntgrParameters:', 'Step:', step, 'PE:', context_state.getPotentialEnergy(), 'Work:', protocol_work, 'Lambda: ', alch_lambda, 'Lambda Step: ', lambda_step)
                    #print('\t'+'ContextParameters:', lambda_step, 'Sterics:', lambda_sterics, 'Elec:', lambda_electrostatics)

                    if step == n_steps:
                        break

                    else:
                        # Get the context state for reporting
                        context_state = context.getState(
                            getPositions=True,
                            getVelocities=True,
                            getEnergy=True,
                            enforcePeriodicBox=thermodynamic_state.is_periodic)
                        #print('Pos:', context_state.getPositions()[0])
                        #print('Step:', n, 'PE:', context_state.getPotentialEnergy())

                    if n % reporter.reportInterval == 0:
                        print('NCMC---IntgrParameters:', 'Step:', step, 'PE:', context_state.getPotentialEnergy(),
                              'Work:', protocol_work, 'Lambda: ', alch_lambda, 'Lambda Step: ', lambda_step)

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
                restart = np.isnan(potential_energy.value_in_unit(potential_energy.unit))

            # Restart the move if we found NaNs.
            if restart:
                err_msg = ('Potential energy is NaN after {} attempts of integration '
                           'with move {}'.format(attempt_counter, self.__class__.__name__))

                # If we are on our last chance before crash, try to re-initialize context
                if attempt_counter == self.n_restart_attempts - 1:
                    logger.error(err_msg + ' Trying to reinitialize Context as a last-resort restart attempt...')
                    context.reinitialize()
                    sampler_state.apply_to_context(context, ignore_velocities=self.reassign_velocities)
                    thermodynamic_state.apply_to_context(context)
                # If we have hit the number of restart attempts, raise an exception.
                elif attempt_counter == self.n_restart_attempts:
                    # Restore the context to the state right before the integration.
                    sampler_state.apply_to_context(context, ignore_velocities=False)

                    logger.error(err_msg)
                    print(IntegratorMoveError(err_msg, self, context))
                    break
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
        sampler_state.update_from_context(context_state, ignore_collective_variables=True, ignore_velocities=True)
        # Update only the collective variables from the Context
        sampler_state.update_from_context(
            context, ignore_positions=True, ignore_velocities=True, ignore_collective_variables=False)

        timer.stop("{}: update sampler state".format(move_name))
        final_context_state = context.getState(
            getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=thermodynamic_state.is_periodic)

        #timer.report_timing()
        return context, init_context_state, final_context_state

    def _get_integrator(self, thermodynamic_state):
        """Implement BaseIntegratorMove._get_integrator()."""
        return LangevinIntegrator(
            temperature=thermodynamic_state.temperature, collision_rate=self.collision_rate, timestep=self.timestep)

    def __getstate__(self):
        if self.context_cache is None:
            context_cache_serialized = None
        else:
            context_cache_serialized = utils.serialize(self.context_cache)
        return dict(
            n_steps=self.n_steps,
            context_cache=context_cache_serialized,
            reassign_velocities=self.reassign_velocities,
            n_restart_attempts=self.n_restart_attempts)

    def __setstate__(self, serialization):
        self.n_steps = serialization['n_steps']
        self.reassign_velocities = serialization['reassign_velocities']
        self.n_restart_attempts = serialization['n_restart_attempts']
        if serialization['context_cache'] is None:
            self.context_cache = None
        else:
            self.context_cache = utils.deserialize(serialization['context_cache'])

    def _after_integration(self, context, thermodynamic_state):
        """Implement BaseIntegratorMove._after_integration()."""
        integrator = context.getIntegrator()
        try:
            # Accumulate acceptance statistics.
            ghmc_global_variables = {
                integrator.getGlobalVariableName(index): index for index in range(integrator.getNumGlobalVariables())
            }
            n_accepted = integrator.getGlobalVariable(ghmc_global_variables['naccept'])
            n_proposed = integrator.getGlobalVariable(ghmc_global_variables['ntrials'])
            self.n_accepted += n_accepted
            self.n_proposed += n_proposed
        except Exception as e:
            print(e)
            pass


class NCMCSampler(object):

    def __init__(self,
                 thermodynamic_state=None,
                 alch_thermodynamic_state=None,
                 sampler_state=None,
                 md_move=None,
                 ncmc_move=None,
                 platform=None,
                 reporter=None,
                 pdbfile=None,
                 topology=None):
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

        if alch_thermodynamic_state is None:
            raise Exception("'alch_thermodynamic_state' must be specified")
        if thermodynamic_state is None:
            raise Exception("'thermodynamic_state' must be specified")
        if sampler_state is None:
            raise Exception("'sampler_state' must be specified")

        self.alch_thermodynamic_state = alch_thermodynamic_state
        self.thermodynamic_state = thermodynamic_state

        #NML: Attach topology to thermodynamic_states
        self.alch_thermodynamic_state.topology = topology
        self.thermodynamic_state.topology = topology

        self.sampler_state = sampler_state
        self.md_move = md_move
        self.ncmc_move = ncmc_move

        # Initialize
        self.accept = 0
        self.reject = 0
        self.iteration = 0
        # For GHMC / Langevin integrator
        self.collision_rate_MD = md_move.collision_rate
        self.timestepMD = md_move.timestep
        self.nstepsMD = md_move.n_steps

        self.collision_rate_NC = ncmc_move.collision_rate
        self.timestepNC = ncmc_move.timestep
        self.nstepsNC = ncmc_move.n_steps  # number of steps per update

        self.verbose = True
        self.platform = platform

        # For writing trajectory files
        self.reporter = reporter

        self._timing = dict()

    def _get_alch_integrator(self, alch_thermodynamic_state):
        return AlchemicalExternalLangevinIntegrator(
            alchemical_functions={
                'lambda_sterics':
                'min(1, (1/0.3)*abs(lambda-0.5))',
                'lambda_electrostatics':
                'step(0.2-lambda) - 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)'
            },
            splitting="H V R O R V H",
            temperature=alch_thermodynamic_state.temperature,
            nsteps_neq=self.nstepsNC,
            timestep=self.timestepNC,
            nprop=1,
            prop_lambda=0.3)

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
            context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
        if positions:
            context.setPositions(state.getPositions(asNumpy=True))
        if velocities:
            context.setVelocities(state.getVelocities(asNumpy=True))
        return context

    def _computeAlchemicalCorrection(self, md_context, md_init_context_state, ncmc_init_context_state,
                                     ncmc_final_context_state, kT):
        """Computes the alchemical correction term from switching between the NCMC
        and MD potentials."""
        # Retrieve the MD/NCMC PE before the proposed move.
        md_init_PE = md_init_context_state.getPotentialEnergy()
        ncmc_init_PE = ncmc_init_context_state.getPotentialEnergy()

        # Retreive the NCMC state after the proposed move.
        ncmc_final_PE = ncmc_final_context_state.getPotentialEnergy()
        md_final_context_state = md_context.getState(
            getPositions=True,
            getVelocities=True,
            getEnergy=True,
            enforcePeriodicBox=self.thermodynamic_state.is_periodic)

        # Set the box_vectors and positions in the alchemical simulation to after the proposed move.
        md_context = self.setContextFromState(md_context, ncmc_final_context_state, velocities=False)

        #self._ncmc_sim.context._integrator.kT
        # Retrieve potential_energy for alch correction
        alch_PE = md_context.getState(getEnergy=True).getPotentialEnergy()
        correction_factor = (ncmc_init_PE - md_init_PE + alch_PE - ncmc_final_PE) * (-1.0 / kT)

        md_context = self.setContextFromState(md_context, md_final_context_state)

        return correction_factor

    def _acceptRejectMove(self):
        work_ncmc = ncmc_context._integrator.getLogAcceptanceProbability(ncmc_context)
        randnum = math.log(np.random.random())
        # Compute correction if work_ncmc is not NaN
        if not np.isnan(work_ncmc):
            correction_factor = self._computeAlchemicalCorrection(md_context, md_init_context_state,
                                                                  ncmc_init_context_state, ncmc_final_context_state,
                                                                  ncmc_context._integrator.kT)
            logger.debug(
                'NCMCLogAcceptanceProbability = %.6f + Alchemical Correction = %.6f' % (work_ncmc, correction_factor))
            work_ncmc = work_ncmc + correction_factor

        if work_ncmc > randnum:
            self.accept += 1
            logger.info('NCMC MOVE ACCEPTED: work_ncmc {} > randnum {}'.format(work_ncmc, randnum))

            # If accept move, sync NCMC state to MD context
            md_context = self.setContextFromState(md_context, ncmc_final_context_state, velocities=False)
            #self.sampler_state.update_from_context(md_context)
        else:
            self.reject += 1
            logger.info('NCMC MOVE REJECTED: work_ncmc {} < {}'.format(work_ncmc, randnum))

            # If reject move, do nothing,
            # NCMC simulation be updated from MD Simulation next iteration.
            md_context = self.setContextFromState(md_context, md_init_context_state)
            #self.sampler_state.update_from_context(md_context)
            md_final_context_state = md_context.getState(
                getPositions=True,
                getVelocities=True,
                getEnergy=True,
                enforcePeriodicBox=self.thermodynamic_state.is_periodic)
            # Potential energy should be from last MD step in the previous iteration
            md_PE = md_final_context_state.getPotentialEnergy()
            if not math.isclose(
                    md_init_context_state.getPotentialEnergy()._value, md_PE._value, rel_tol=float('1e-%s' % rtol)):
                logger.error(
                    'Last MD potential energy %s != Current MD potential energy %s. Potential energy should match the prior state.'
                    % (md_init_context_state.getPotentialEnergy(), md_PE))
                sys.exit(1)

    def simulateNCMC(self):
        """
        Run the NCMC simulation.
        """
        alch_integrator = self._get_alch_integrator(self.alch_thermodynamic_state)
        ncmc_context, ncmc_init_context_state, ncmc_final_context_state = self.ncmc_move.apply(
            self.alch_thermodynamic_state,
            self.sampler_state,
            alch_integrator,
            self.reporter,
            alch=True,
            ignore_velocities=True)
        return ncmc_context, ncmc_init_context_state, ncmc_final_context_state

    def simulateMD(self):
        """
        Run the MD Simulation
        """
        integrator = self.md_move._get_integrator(self.thermodynamic_state)
        md_context, md_init_context_state, md_final_context_state = self.md_move.apply(
            self.thermodynamic_state, self.sampler_state, integrator, self.reporter)

        return md_context, md_init_context_state, md_final_context_state

    def update(self, contexts):
        """
        Update the sampler with one step of sampling.
        """
        if self.verbose:
            print("." * 80)
            print("MCMC sampler iteration %d" % self.iteration)

        # Update sampler state.
        self.sampler_state.update_from_context(contexts[0])

        # Increment iteration count
        self.iteration += 1

        if self.verbose:
            print("." * 80)

    def run(self, n_iterations=1):
        """
        Run the sampler for the specified number of iterations
        Parameters
        ----------
        niterations : int, optional, default=1
            Number of iterations to run the sampler for.
        """
        for iteration in range(n_iterations):
            print('Running simulateNCMC')
            initial_time = time.time()

            ncmc_contexts = self.simulateNCMC()
            self.update(ncmc_contexts)
            print('Running MD Simulations')
            md_contexts = self.simulateMD()

            final_time = time.time()
            elapsed_time = final_time - initial_time
            self._timing['sample positions'] = elapsed_time

            if self.verbose:
                final_energy = contexts[0].getState(getEnergy=True).getPotentialEnergy() * self.thermodynamic_state.beta
                print('Final energy is %12.3f kT' % (final_energy))
                print('elapsed time %8.3f s' % elapsed_time)

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

        context_state = context.getState(
            getPositions=True,
            getVelocities=True,
            getEnergy=True,
            enforcePeriodicBox=self.thermodynamic_state.is_periodic)
        potential_energy = context_state.getPotentialEnergy()
        print(potential_energy)

        #timer.report_timing()


from blues import utils
parser = argparse.ArgumentParser(description='Restart file name')
parser.add_argument('-j', '--jobname', type=str, help="store jobname")
parser.add_argument('-n', '--nIter', default=1, type=int, help="number of Iterations")
parser.add_argument('-s', '--nsteps', default=100, type=int, help="number of steps")
parser.add_argument('-r', '--reportInterval', default=10, type=int, help="reportInterval")
args = parser.parse_args()

# Define parameters
temperature = 300 * unit.kelvin
collision_rate = 1 / unit.picoseconds
timestep = 4.0 * unit.femtoseconds
n_steps = args.nsteps
reportInterval = args.reportInterval
nIter = args.nIter

print('nIter: {}, nsteps: {}, timestep: {}, reportInterval: {}'.format(nIter, n_steps, timestep, reportInterval))

prmtop = utils.get_data_filename('blues', 'tests/data/eqToluene.prmtop')
inpcrd = utils.get_data_filename('blues', 'tests/data/eqToluene.inpcrd')
tol = parmed.load_file(prmtop, xyz=inpcrd)
tol.system = tol.createSystem(
    nonbondedMethod=openmm.app.PME,
    nonbondedCutoff=10 * unit.angstrom,
    constraints=openmm.app.HBonds,
    hydrogenMass=3.024 * unit.dalton,
    rigidWater=True,
    removeCMMotion=True,
    flexibleConstraints=True,
    splitDihedrals=False)

factory = alchemy.AbsoluteAlchemicalFactory(
    consistent_exceptions=False,
    disable_alchemical_dispersion_correction=True,
    alchemical_pme_treatment='direct-space')
alchemical_atom_idx = utils.atomIndexfromTop('LIG', tol.topology)
alchemical_region = alchemy.AlchemicalRegion(
    alchemical_atoms=alchemical_atom_idx,
    softcore_alpha=0.5,
    softcore_a=1,
    softcore_b=1,
    softcore_c=6,
    softcore_beta=0.0,
    softcore_d=1,
    softcore_e=1,
    softcore_f=2,
    annihilate_sterics=False,
    annihilate_electrostatics=True,
)
alchemical_atoms = list(alchemical_region.alchemical_atoms)
toluene_alchemical_system = factory.create_alchemical_system(
    reference_system=tol.system, alchemical_regions=alchemical_region)
alchemical_state = alchemy.AlchemicalState.from_system(toluene_alchemical_system)
# Create our custom State objects
# Need two different Thermodynamic State objects
# Context cache will grab correct thermodynamic state
# Keeping them in sync is in SamplerState.apply to context
# Have apply return accumulated work

alch_thermodynamic_state = ThermodynamicState(system=toluene_alchemical_system, temperature=temperature)
alch_thermodynamic_state = CompoundThermodynamicState(alch_thermodynamic_state, composable_states=[alchemical_state])
alch_thermodynamic_state.alchemical_atoms = alchemical_atoms
thermodynamic_state = ThermodynamicState(system=tol.system, temperature=temperature)
sampler_state = SamplerState(positions=tol.positions)

from blues.reporters import NetCDF4Reporter
with open('%s.pdb' % args.jobname, 'w') as pdb:
    openmm.app.pdbfile.PDBFile.writeFile(tol.topology, tol.positions, pdb)
filename = '%s.nc' % args.jobname
if os.path.exists(filename):
    os.remove(filename)
else:
    print("Sorry, I can not remove %s file." % filename)
nc_reporter = NetCDF4Reporter(filename, 1)
nc_reporter.reportInterval = reportInterval

langevin = LangevinDynamicsMove(
    timestep=timestep, collision_rate=collision_rate, n_steps=n_steps, n_restart_attempts=5, reassign_velocities=True)
ncmc_move = NCMCMove(
    timestep=timestep,
    collision_rate=collision_rate,
    n_steps=n_steps,
    temperature=temperature,
    n_restart_attempts=5,
    reassign_velocities=True)
sampler = NCMCSampler(
    thermodynamic_state,
    alch_thermodynamic_state,
    sampler_state,
    md_move=langevin,
    ncmc_move=ncmc_move,
    platform=None,
    reporter=nc_reporter,
    topology=tol.topology)
sampler.minimize(max_iterations=0)
sampler.run(n_iterations=nIter)
