#!/usr/bin/env python3
"""
Optimization script for finding optimal BESS and private wire sizing.

Uses scipy optimization to find the combination of:
- BESS energy capacity (MWh)
- BESS power capacity (MW)
- Private wire capacity (MW)

That minimizes total system cost while meeting performance constraints.
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from typing import Dict, Tuple, Callable
import yaml
from pathlib import Path
import argparse
import sys

from src.simulator import SystemSimulator


class SizingOptimizer:
    """
    Optimizes BESS and wire sizing to minimize total cost.
    """

    def __init__(
        self,
        simulator: SystemSimulator,
        optimization_config: Dict = None,
        max_curtailment_rate: float = None,
        max_grid_import_rate: float = None,
        first_year_only: bool = False,
        assume_all_sellable: bool = False
    ):
        """
        Initialize sizing optimizer.

        Parameters
        ----------
        simulator : SystemSimulator
            Configured system simulator
        optimization_config : dict, optional
            Optimization configuration dictionary (from YAML)
        max_curtailment_rate : float, optional
            Maximum allowable curtailment rate (fraction of generation)
            If None, uses value from optimization_config
        max_grid_import_rate : float, optional
            Maximum allowable grid import rate (fraction of load)
            If None, uses value from optimization_config
        first_year_only : bool
            If True, run simulations for first year only (faster optimization)
        assume_all_sellable : bool
            If True, assume all energy can be sold (sets load to wire capacity)
        """
        self.simulator = simulator
        self.optimization_config = optimization_config or {}
        self.first_year_only = first_year_only
        self.assume_all_sellable = assume_all_sellable

        # Set constraints (command-line args override config)
        # Constraints can be None to disable them
        constraints = self.optimization_config.get('constraints', {})
        self.max_curtailment_rate = (
            max_curtailment_rate
            if max_curtailment_rate is not None
            else constraints.get('max_curtailment_rate')
        )
        self.max_grid_import_rate = (
            max_grid_import_rate
            if max_grid_import_rate is not None
            else constraints.get('max_grid_import_rate')
        )

        # Get penalty values from config
        penalties = self.optimization_config.get('penalties', {})
        self.curtailment_penalty = penalties.get('curtailment_violation', 1e9)
        self.grid_import_penalty = penalties.get('grid_import_violation', 1e9)
        self.simulation_failure_cost = penalties.get('simulation_failure', 1e12)

        # Get economic parameters
        econ_config = simulator.config['economics']
        self.bess_capex_per_mwh = econ_config['bess_capex_eur_per_mwh']
        self.wire_capex_per_mw = econ_config['wire_capex_eur_per_mw']
        self.ppa_price = econ_config['ppa_price_eur_per_mwh']
        self.project_lifetime = econ_config['project_lifetime_years']
        self.discount_rate = econ_config['discount_rate']

        # Track evaluations
        self.evaluation_count = 0
        self.best_cost = np.inf
        self.best_params = None

    def objective_function(self, x: np.ndarray) -> float:
        """
        Objective function to minimize: Net Present Cost (NPC).

        NPC = Total Capex - NPV(Revenue)

        Parameters
        ----------
        x : np.ndarray
            [bess_energy_mwh, bess_power_mw, wire_capacity_mw]

        Returns
        -------
        float
            Net present cost (to minimize)
        """
        bess_energy_mwh, bess_power_mw, wire_capacity_mw = x

        self.evaluation_count += 1

        try:
            # Run simulation
            results = self.simulator.run_simulation(
                bess_energy_mwh=bess_energy_mwh,
                bess_power_mw=bess_power_mw,
                wire_capacity_mw=wire_capacity_mw,
                verbose=False,
                first_year_only=self.first_year_only,
                assume_all_sellable=self.assume_all_sellable
            )

            metrics = results['metrics']

            # Check constraints
            curtailment_rate = metrics['utilization']['curtailment_rate']
            grid_import_rate = metrics['utilization']['grid_import_rate']

            # Penalty for constraint violations (only if constraints are set)
            penalty = 0
            if self.max_curtailment_rate is not None and curtailment_rate > self.max_curtailment_rate:
                penalty += self.curtailment_penalty * (curtailment_rate - self.max_curtailment_rate)

            if self.max_grid_import_rate is not None and grid_import_rate > self.max_grid_import_rate:
                penalty += self.grid_import_penalty * (grid_import_rate - self.max_grid_import_rate)

            # Calculate Net Present Cost
            capex = metrics['economics']['total_capex_eur']

            # Annual revenue from PPA
            total_delivery_mwh = metrics['energy']['total_wire_delivery_mwh']
            simulation_years = len(results['timeseries']) * (self.simulator.config['simulation']['timestep_minutes'] / 60) / 8760
            annual_revenue = (total_delivery_mwh / simulation_years) * self.ppa_price

            # NPV of revenue stream
            npv_revenue = self._calculate_npv(
                annual_revenue,
                self.project_lifetime,
                self.discount_rate
            )

            # Net present cost (lower is better)
            npc = capex - npv_revenue + penalty

            # Track best solution
            if npc < self.best_cost:
                self.best_cost = npc
                self.best_params = x.copy()
                print(f"\n[Eval {self.evaluation_count}] New best solution:")
                print(f"  BESS: {bess_energy_mwh:.1f} MWh / {bess_power_mw:.1f} MW")
                print(f"  Wire: {wire_capacity_mw:.1f} MW")
                print(f"  Capex: €{capex:,.0f}")
                print(f"  NPV Revenue: €{npv_revenue:,.0f}")
                print(f"  NPC: €{npc:,.0f}")
                print(f"  Curtailment: {curtailment_rate:.1%}")
                print(f"  Grid Import: {grid_import_rate:.1%}")

            return npc

        except Exception as e:
            print(f"Simulation failed: {e}")
            return self.simulation_failure_cost

    def _calculate_npv(
        self,
        annual_cashflow: float,
        years: int,
        discount_rate: float
    ) -> float:
        """Calculate net present value of annual cashflow."""
        npv = 0
        for year in range(1, years + 1):
            npv += annual_cashflow / (1 + discount_rate) ** year
        return npv

    def optimize(
        self,
        method: str = None,
        bounds: Tuple[Tuple[float, float], ...] = None
    ) -> Dict:
        """
        Run optimization to find optimal sizing.

        Parameters
        ----------
        method : str, optional
            Optimization method: 'differential_evolution' or 'nelder-mead'
            If None, uses value from optimization_config
        bounds : tuple of tuples, optional
            Bounds for (bess_energy_mwh, bess_power_mw, wire_capacity_mw)
            If None, uses values from optimization_config

        Returns
        -------
        dict
            Optimization results including optimal parameters and metrics
        """
        # Get method from config if not specified
        if method is None:
            method = self.optimization_config.get('method', 'differential_evolution')

        # Get bounds from config if not specified
        if bounds is None:
            bounds_config = self.optimization_config.get('bounds', {})
            bounds = (
                (
                    bounds_config.get('bess_energy_mwh', {}).get('min', 50),
                    bounds_config.get('bess_energy_mwh', {}).get('max', 500)
                ),
                (
                    bounds_config.get('bess_power_mw', {}).get('min', 20),
                    bounds_config.get('bess_power_mw', {}).get('max', 200)
                ),
                (
                    bounds_config.get('wire_capacity_mw', {}).get('min', 40),
                    bounds_config.get('wire_capacity_mw', {}).get('max', 200)
                )
            )

        # Get display settings
        display_config = self.optimization_config.get('display', {})
        separator = display_config.get('print_header_separator', '=' * 70)

        print(separator)
        print("STARTING SIZING OPTIMIZATION")
        print(separator)
        print(f"Method: {method}")
        print(f"Bounds:")
        print(f"  BESS Energy:  {bounds[0][0]:.0f} - {bounds[0][1]:.0f} MWh")
        print(f"  BESS Power:   {bounds[1][0]:.0f} - {bounds[1][1]:.0f} MW")
        print(f"  Wire Capacity: {bounds[2][0]:.0f} - {bounds[2][1]:.0f} MW")
        print(f"Constraints:")
        if self.max_curtailment_rate is not None:
            print(f"  Max Curtailment: {self.max_curtailment_rate:.1%}")
        else:
            print(f"  Max Curtailment: None (no constraint)")
        if self.max_grid_import_rate is not None:
            print(f"  Max Grid Import: {self.max_grid_import_rate:.1%}")
        else:
            print(f"  Max Grid Import: None (no constraint)")
        print(separator)

        self.evaluation_count = 0
        self.best_cost = np.inf

        # Get algorithm settings from config
        algo_settings = self.optimization_config.get('algorithm_settings', {})

        if method == "differential_evolution":
            # Global optimization using differential evolution
            de_settings = algo_settings.get('differential_evolution', {})
            result = differential_evolution(
                self.objective_function,
                bounds=bounds,
                maxiter=de_settings.get('maxiter', 50),
                popsize=de_settings.get('popsize', 10),
                seed=de_settings.get('seed', 42),
                disp=True,
                workers=de_settings.get('workers', 1),
                updating=de_settings.get('updating', 'deferred'),
                polish=de_settings.get('polish', True)
            )

        elif method == "nelder-mead":
            # Local optimization using Nelder-Mead
            # Start from middle of bounds
            x0 = np.array([
                (bounds[0][0] + bounds[0][1]) / 2,
                (bounds[1][0] + bounds[1][1]) / 2,
                (bounds[2][0] + bounds[2][1]) / 2
            ])

            nm_settings = algo_settings.get('nelder_mead', {})
            result = minimize(
                self.objective_function,
                x0=x0,
                method='Nelder-Mead',
                options={'maxiter': nm_settings.get('maxiter', 100), 'disp': True}
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        # Run final simulation with optimal parameters
        optimal_bess_energy, optimal_bess_power, optimal_wire = result.x

        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)

        final_results = self.simulator.run_simulation(
            bess_energy_mwh=optimal_bess_energy,
            bess_power_mw=optimal_bess_power,
            wire_capacity_mw=optimal_wire,
            verbose=True,
            first_year_only=self.first_year_only,
            assume_all_sellable=self.assume_all_sellable
        )

        return {
            'optimal_bess_energy_mwh': optimal_bess_energy,
            'optimal_bess_power_mw': optimal_bess_power,
            'optimal_wire_capacity_mw': optimal_wire,
            'optimal_npc': result.fun,
            'optimization_result': result,
            'simulation_results': final_results,
            'evaluations': self.evaluation_count
        }

    def grid_search(
        self,
        bess_energy_range: np.ndarray = None,
        bess_power_range: np.ndarray = None,
        wire_capacity_range: np.ndarray = None
    ) -> pd.DataFrame:
        """
        Perform grid search over parameter space.

        Useful for visualization and understanding sensitivity.

        Parameters
        ----------
        bess_energy_range : np.ndarray, optional
            Array of BESS energy capacities to test
            If None, uses values from optimization_config
        bess_power_range : np.ndarray, optional
            Array of BESS power capacities to test
            If None, uses values from optimization_config
        wire_capacity_range : np.ndarray, optional
            Array of wire capacities to test
            If None, uses values from optimization_config

        Returns
        -------
        pd.DataFrame
            Results for all combinations
        """
        # Get grid search ranges from config if not specified
        if bess_energy_range is None or bess_power_range is None or wire_capacity_range is None:
            algo_settings = self.optimization_config.get('algorithm_settings', {})
            grid_settings = algo_settings.get('grid_search', {})

            if bess_energy_range is None:
                bess_energy_range = np.array(grid_settings.get('bess_energy_values', [100, 200, 300]))
            if bess_power_range is None:
                bess_power_range = np.array(grid_settings.get('bess_power_values', [40, 60, 80]))
            if wire_capacity_range is None:
                wire_capacity_range = np.array(grid_settings.get('wire_capacity_values', [60, 80, 100]))
        results = []

        total_combinations = len(bess_energy_range) * len(bess_power_range) * len(wire_capacity_range)
        print(f"Running grid search over {total_combinations} combinations...")

        count = 0
        for bess_e in bess_energy_range:
            for bess_p in bess_power_range:
                for wire_c in wire_capacity_range:
                    count += 1
                    print(f"\r[{count}/{total_combinations}] Testing: BESS={bess_e:.0f}MWh/{bess_p:.0f}MW, Wire={wire_c:.0f}MW", end='')

                    npc = self.objective_function([bess_e, bess_p, wire_c])

                    # Get last simulation metrics
                    if self.simulator.results is not None:
                        metrics = self.simulator.results['metrics']
                        results.append({
                            'bess_energy_mwh': bess_e,
                            'bess_power_mw': bess_p,
                            'wire_capacity_mw': wire_c,
                            'npc': npc,
                            'curtailment_rate': metrics['utilization']['curtailment_rate'],
                            'grid_import_rate': metrics['utilization']['grid_import_rate'],
                            'renewable_fraction': metrics['utilization']['renewable_fraction'],
                            'capex': metrics['economics']['total_capex_eur']
                        })

        print("\nGrid search complete!")
        return pd.DataFrame(results)


def main():
    """Main entry point for sizing optimization."""
    parser = argparse.ArgumentParser(
        description="Optimize BESS and private wire sizing for renewable energy system"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to system configuration file'
    )
    parser.add_argument(
        '--optimization-config',
        type=str,
        default='optimization_config.yaml',
        help='Path to optimization configuration file'
    )
    parser.add_argument(
        '--method',
        type=str,
        default=None,
        choices=['differential_evolution', 'nelder-mead', 'grid'],
        help='Optimization method (overrides config file)'
    )
    parser.add_argument(
        '--max-curtailment',
        type=float,
        default=None,
        help='Maximum curtailment rate (0-1) (overrides config file)'
    )
    parser.add_argument(
        '--max-grid-import',
        type=float,
        default=None,
        help='Maximum grid import rate (0-1) (overrides config file)'
    )
    parser.add_argument(
        '--first-year-only',
        action='store_true',
        help='Run simulation for first year only (faster optimization)'
    )
    parser.add_argument(
        '--assume-all-sellable',
        action='store_true',
        help='Assume all energy can be sold (sets load to wire capacity)'
    )

    args = parser.parse_args()

    # Load optimization configuration
    optimization_config_path = Path(args.optimization_config)
    if optimization_config_path.exists():
        with open(optimization_config_path, 'r') as f:
            optimization_config = yaml.safe_load(f)
        print(f"Loaded optimization config from: {optimization_config_path}")
    else:
        print(f"Warning: Optimization config not found at {optimization_config_path}")
        print("Using default optimization parameters")
        optimization_config = {}

    # Initialize simulator
    simulator = SystemSimulator(args.config)

    # Initialize optimizer
    optimizer = SizingOptimizer(
        simulator,
        optimization_config=optimization_config,
        max_curtailment_rate=args.max_curtailment,
        max_grid_import_rate=args.max_grid_import,
        first_year_only=args.first_year_only,
        assume_all_sellable=args.assume_all_sellable
    )

    # Determine method (command-line overrides config)
    method = args.method if args.method is not None else optimization_config.get('method', 'differential_evolution')

    # Run optimization
    if method == 'grid':
        # Grid search - uses config values if not specified
        results_df = optimizer.grid_search()
        print("\nGrid search results:")
        print(results_df.sort_values('npc').head(10))

        # Save results
        output_path = Path("outputs")
        output_path.mkdir(exist_ok=True)
        results_df.to_csv(output_path / "grid_search_results.csv", index=False)

    else:
        # Run optimization
        results = optimizer.optimize(method=args.method)

        # Save optimal configuration
        output_path = Path("outputs")
        output_path.mkdir(exist_ok=True)

        optimal_config = {
            'optimal_parameters': {
                'bess_energy_mwh': float(results['optimal_bess_energy_mwh']),
                'bess_power_mw': float(results['optimal_bess_power_mw']),
                'wire_capacity_mw': float(results['optimal_wire_capacity_mw'])
            },
            'optimization': {
                'method': method,
                'evaluations': results['evaluations'],
                'optimal_npc': float(results['optimal_npc'])
            }
        }

        with open(output_path / "optimal_sizing.yaml", 'w') as f:
            yaml.dump(optimal_config, f, default_flow_style=False)

        print(f"\nOptimal configuration saved to {output_path / 'optimal_sizing.yaml'}")

        # Save detailed results
        simulator.save_results()


if __name__ == "__main__":
    main()
