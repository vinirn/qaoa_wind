# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a quantum computing project implementing QAOA (Quantum Approximate Optimization Algorithm) for wind turbine placement optimization, specifically addressing wake effects between turbines.

## Running the Code

Execute with default configuration:
```bash
python qaoa_turbinas.py
```

Use the automated script:
```bash
./run_qaoa.sh
```

## Configuration

The system is fully configurable via JSON files:

- `config.json` - Default 2x3 grid
- `config_3x3.json` - Larger 3x3 grid with constraints
- `config_vertical.json` - Vertical 4x2 grid with north→south wind

### Configuration Parameters:
- **Grid size**: rows, cols
- **Wind direction**: [0,1] = west→east, [1,0] = north→south  
- **Energy production**: fixed values, random, or uniform
- **Penalties**: max penalty and decay factor
- **QAOA settings**: layers, iterations, optimizer
- **Constraints**: min/max number of turbines

## Dependencies

Install dependencies using the requirements file:
```bash
pip install -r requirements.txt
```

Or create a virtual environment (recommended):
```bash
python3 -m venv qiskit_env
source qiskit_env/bin/activate
pip install -r requirements.txt
```

Main dependencies:
- qiskit >= 2.1.0 (quantum computing framework)
- qiskit-aer >= 0.17.0 (quantum simulator)
- numpy >= 2.0.0 (numerical computing)
- scipy >= 1.15.0 (optimization algorithms)

## Architecture

The codebase consists of a single file `qaoa_turbinas.py` that:

1. **Problem Setup**: Defines a 2x3 grid for turbine placement with energy production values and wake penalty matrix
2. **QUBO Formulation**: Converts the optimization problem to Quadratic Unconstrained Binary Optimization format
3. **QAOA Execution**: Uses quantum approximate optimization with COBYLA classical optimizer
4. **Results**: Outputs optimal turbine positions and objective function value

### Key Components

- **Grid Layout**: 2x3 positions (x0-x5) representing potential turbine locations
- **Energy Array**: Expected energy production for each position
- **Wake Penalties**: Dictionary defining interference penalties between turbine pairs aligned with wind direction
- **QAOA Configuration**: Single repetition (reps=1) with AerEstimator for quantum simulation

The problem maximizes total energy production while minimizing wake interference penalties through binary decision variables (1 = turbine installed, 0 = empty position).

## QAOA Algorithm Details

### What is an Ansatz?

**Ansatz** is a German word meaning "approach" or "attempt". In quantum computing, it refers to a **parameterized quantum circuit** that represents a family of possible quantum states.

#### QAOA Ansatz Structure:
```
|ψ(γ,β)⟩ = e^{-iβH_mix} e^{-iγH_cost} |+⟩^⊗n
```

Where:
- `|+⟩^⊗n` = Initial state (uniform superposition of all configurations)
- `H_cost` = Cost Hamiltonian (encodes our objective function)
- `H_mix` = Mixing Hamiltonian (X rotations for exploration)
- `γ, β` = **Variational parameters** (optimized classically)

#### How it Works:

1. **Initial State**: All turbine configurations are equally probable
2. **Cost Evolution**: Parameter γ controls how much the problem influences the state
3. **Mixing**: Parameter β controls how much we "mix" between solutions
4. **Result**: Configurations with better energy yield have higher measurement probability

#### Circuit Structure:
```
     ┌───┐ ┌──────────┐ ┌─────────┐
q_0: ┤ H ├─┤ e^(-iγH) ├─┤ RX(2β) ├─
     ├───┤ │          │ ├─────────┤
q_1: ┤ H ├─┤  (cost   ├─┤ RX(2β) ├─
     └───┘ │   part)  │ └─────────┘
      ...  └──────────┘    ...
```

#### Parameter Optimization:
- **γ = 0, β = 0**: Completely random state
- **γ very large**: Gets "stuck" in one solution  
- **β very large**: Always mixing, never converges
- **Optimized values**: Balance exploration vs exploitation

The ansatz is the **structural form** of the quantum circuit, while the parameters are the **"knobs"** we tune to find the best solution through classical optimization.