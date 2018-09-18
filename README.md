# MolSSI Project
A newly developed toolkit from the Mobley group called BLUES (Binding modes of Ligands Using Enhanced Sampling) takes a novel approach that combines non-equilibrium candidate Monte Carlo (NCMC) move proposals with molecular dynamics (MD) simulations through OpenMM, a GPU accelerated molecular simulation package. Our work on BLUES has shown us that mixing specific types of well-selected Monte Carlo (MC) moves with MD can dramatically improve efficiency in sampling. We have been able to successfully implement several of such types of moves, such as random rotations and protein side chain rotations, as has the Chodera lab with SaltSwap and constant pH moves. However, to allow the field to benefit from all of these we need to provide a toolkit which essentially provides access to a library for mixing such moves (both via standard MC and via NCMC) with MD simulations.
Here, my goal is to **develop a generalized framework in BLUES which enables the use of a variety of move types in MD simulations**, thus allowing users to select or mix moves suitable for their particular sampling problem.

## OpenMMTools Notes
- Entry point: `openmm.System`
- **States module**
    - `ThermodynamicState` generated from: `openmm.System`
    - `AlchemicalState` generated from: `AbsoluteAlchemicalFactory` -> `AlchemicalRegion` -> `AlchemicalState`.
    - Combine `ThermodynamicState` with `AlchemicalState` to create the `CompoundThermodynamicState`
    - `SamplerState` generated from: `openmm.System`

- **MCMC (moves) module**
    - Moves must implement 4 methods:
        - `get_integrator()`
        - `_before_integration()`
        - `_after_integration()`
        - `apply(self, thermodynamic_state, sampler_state):`
            - Call the BaseIntegratorMove().apply() from new mcmc moves via:
                - `super(GHMCMove, self).apply(thermodynamic_state, sampler_state)`
    - Moves generate the integrator, accumulate the statistics, handle updating the `States` and the `Contexts`, and handle acceptance/rejection.

# Issues
