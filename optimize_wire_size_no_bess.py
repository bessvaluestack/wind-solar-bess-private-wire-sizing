#!/usr/bin/env python3
"""
Wire sizing optimization without BESS (no battery storage).

This optimizer finds the optimal private wire capacity by balancing:
- Wire capex (higher capacity = higher cost)
- Curtailment (lower capacity = more curtailment)

Uses the simulator with skip_bess=True to run without battery storage,
focusing purely on the wire sizing decision for direct renewable energy delivery.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, differential_evolution
from typing import Dict, Optional
import yaml
from pathlib import Path
import argparse

from src.simulator import SystemSimulator


class WireSizingOptimizer:
    """
    Optimizes private wire capacity without BESS to maximize net profit
    (NPV Revenue - Wire Capex), finding the point where marginal wire cost
    exceeds marginal revenue from reduced curtailment.
    """

    def __init__(
        self,
        simulator: SystemSimulator,
        optimization_config: Dict = None,
        curtailment_penalty_eur_per_mwh: float = None,
        first_year_only: bool = False,
        assume_all_sellable: bool = False
    ):
        """
        Initialize wire sizing optimizer.

        Parameters
        ----------
        simulator : SystemSimulator
            Configured system simulator
        optimization_config : dict, optional
            Optimization configuration dictionary (from YAML)
        curtailment_penalty_eur_per_mwh : float, optional
            Economic penalty for curtailed energy (EUR/MWh)
            Represents opportunity cost of lost energy sales
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

        # Get economic parameters
        econ_config = simulator.config['economics']
        self.wire_capex_per_mw = econ_config['wire_capex_eur_per_mw']
        self.ppa_price = econ_config['ppa_price_eur_per_mwh']
        self.project_lifetime = econ_config['project_lifetime_years']
        self.discount_rate = econ_config['discount_rate']

        # Set curtailment penalty (command-line arg overrides config)
        wire_optimization = self.optimization_config.get('wire_optimization', {})
        self.curtailment_penalty_eur_per_mwh = (
            curtailment_penalty_eur_per_mwh
            if curtailment_penalty_eur_per_mwh is not None
            else wire_optimization.get('curtailment_penalty_eur_per_mwh', self.ppa_price)
        )

        # Penalty for simulation failures
        penalties = self.optimization_config.get('penalties', {})
        self.simulation_failure_cost = penalties.get('simulation_failure', 1e12)

        # Track evaluations
        self.evaluation_count = 0
        self.best_npc = np.inf  # Most negative (best profit) is optimal
        self.best_wire_capacity = None

        # Store results for plotting/analysis
        self.evaluation_history = []

    def objective_function(self, wire_capacity_mw: float) -> float:
        """
        Objective function to minimize: Net Present Cost (NPC).

        NPC = Wire Capex - NPV Revenue

        More negative NPC = better (higher profit). This finds the wire size
        where marginal cost of additional capacity equals marginal revenue
        from reduced curtailment.

        Parameters
        ----------
        wire_capacity_mw : float
            Private wire capacity (MW)

        Returns
        -------
        float
            Net Present Cost (to minimize; more negative = better profit)
        """
        self.evaluation_count += 1

        try:
            # Run simulation WITHOUT BESS
            results = self.simulator.run_simulation(
                bess_energy_mwh=0,  # Not used when skip_bess=True
                bess_power_mw=0,    # Not used when skip_bess=True
                wire_capacity_mw=wire_capacity_mw,
                verbose=False,
                first_year_only=self.first_year_only,
                assume_all_sellable=self.assume_all_sellable,
                skip_bess=True  # Skip BESS modeling
            )

            metrics = results['metrics']

            # Calculate wire capex
            wire_capex = wire_capacity_mw * self.wire_capex_per_mw

            # Calculate NPV revenue from wire deliveries
            total_delivery_mwh = metrics['energy']['total_wire_delivery_mwh']
            simulation_years = len(results['timeseries']) * (
                self.simulator.config['simulation']['timestep_minutes'] / 60
            ) / 8760
            annual_revenue = (total_delivery_mwh / simulation_years) * self.ppa_price
            npv_revenue = self._calculate_npv(
                annual_revenue,
                self.project_lifetime,
                self.discount_rate
            )

            # Net Present Cost (Capex - Revenue) - more negative = better profit
            npc = wire_capex - npv_revenue

            # Calculate curtailment metrics for reference
            total_curtailment_mwh = metrics['energy']['total_curtailment_mwh']
            annual_curtailment_mwh = total_curtailment_mwh / simulation_years
            npv_curtailment_cost = self._calculate_npv(
                annual_curtailment_mwh * self.curtailment_penalty_eur_per_mwh,
                self.project_lifetime,
                self.discount_rate
            )

            # Total cost (for reference only - not used in optimization)
            total_cost = wire_capex + npv_curtailment_cost

            # Store evaluation results
            eval_result = {
                'wire_capacity_mw': wire_capacity_mw,
                'npc': npc,
                'wire_capex': wire_capex,
                'npv_revenue': npv_revenue,
                'net_profit': -npc,  # Negative NPC = profit
                'total_cost': total_cost,  # For reference
                'npv_curtailment_cost': npv_curtailment_cost,
                'curtailment_rate': metrics['utilization']['curtailment_rate'],
                'curtailment_mwh': total_curtailment_mwh,
                'wire_delivery_mwh': total_delivery_mwh,
                'wire_utilization': metrics['utilization']['wire_utilization']
            }
            self.evaluation_history.append(eval_result)

            # Track best solution (most negative NPC = highest profit)
            if npc < self.best_npc:
                self.best_npc = npc
                self.best_wire_capacity = wire_capacity_mw
                print(f"\n[Eval {self.evaluation_count}] New best solution:")
                print(f"  Wire Capacity: {wire_capacity_mw:.1f} MW")
                print(f"  Wire Capex: €{wire_capex:,.0f}")
                print(f"  NPV Revenue: €{npv_revenue:,.0f}")
                print(f"  NPC (Capex-Revenue): €{npc:,.0f}")
                print(f"  Net Profit: €{-npc:,.0f}")
                print(f"  ---")
                print(f"  Curtailment: {metrics['utilization']['curtailment_rate']:.2%} "
                      f"({total_curtailment_mwh:,.0f} MWh)")
                print(f"  Wire Utilization: {metrics['utilization']['wire_utilization']:.1%}")

            return npc

        except Exception as e:
            print(f"Simulation failed for wire_capacity={wire_capacity_mw:.1f} MW: {e}")
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
        wire_capacity_bounds: tuple = None
    ) -> Dict:
        """
        Run optimization to find optimal wire sizing.

        Parameters
        ----------
        method : str, optional
            Optimization method: 'bounded' (scalar minimization) or 'differential_evolution'
            If None, uses value from optimization_config
        wire_capacity_bounds : tuple, optional
            Bounds for wire capacity (min_mw, max_mw)
            If None, uses values from optimization_config

        Returns
        -------
        dict
            Optimization results including optimal wire capacity and metrics
        """
        # Get method from config if not specified
        if method is None:
            wire_optimization = self.optimization_config.get('wire_optimization', {})
            method = wire_optimization.get('method', 'bounded')

        # Get bounds from config if not specified
        if wire_capacity_bounds is None:
            bounds_config = self.optimization_config.get('bounds', {})
            wire_bounds = bounds_config.get('wire_capacity_mw', {})
            wire_capacity_bounds = (
                wire_bounds.get('min', 40),
                wire_bounds.get('max', 200)
            )

        # Get display settings
        display_config = self.optimization_config.get('display', {})
        separator = display_config.get('print_header_separator', '=' * 70)

        print(separator)
        print("WIRE SIZING OPTIMIZATION (NO BESS)")
        print(separator)
        print(f"Method: {method}")
        print(f"Wire Capacity Bounds: {wire_capacity_bounds[0]:.0f} - {wire_capacity_bounds[1]:.0f} MW")
        print(f"Curtailment Penalty: €{self.curtailment_penalty_eur_per_mwh:.2f}/MWh")
        print(f"Wire Capex: €{self.wire_capex_per_mw:,.0f}/MW")
        print(f"PPA Price: €{self.ppa_price:.2f}/MWh")
        if self.first_year_only:
            print("Running simulations for FIRST YEAR ONLY")
        if self.assume_all_sellable:
            print("Assuming ALL ENERGY SELLABLE (load = wire capacity)")
        print(separator)

        self.evaluation_count = 0
        self.best_npc = np.inf
        self.evaluation_history = []

        if method == "bounded":
            # Bounded scalar minimization (efficient for 1D optimization)
            result = minimize_scalar(
                self.objective_function,
                bounds=wire_capacity_bounds,
                method='bounded',
                options={'disp': True}
            )
            optimal_wire = result.x

        elif method == "differential_evolution":
            # Differential evolution (global optimization)
            algo_settings = self.optimization_config.get('algorithm_settings', {})
            de_settings = algo_settings.get('differential_evolution', {})

            result = differential_evolution(
                lambda x: self.objective_function(x[0]),  # Wrap for 1D array
                bounds=[wire_capacity_bounds],
                maxiter=de_settings.get('maxiter', 50),
                popsize=de_settings.get('popsize', 10),
                seed=de_settings.get('seed', 42),
                disp=True,
                workers=de_settings.get('workers', 1),
                updating=de_settings.get('updating', 'deferred'),
                polish=de_settings.get('polish', True)
            )
            optimal_wire = result.x[0]

        else:
            raise ValueError(f"Unknown method: {method}. Choose 'bounded' or 'differential_evolution'")

        # Run final simulation with optimal wire capacity
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)

        final_results = self.simulator.run_simulation(
            bess_energy_mwh=0,
            bess_power_mw=0,
            wire_capacity_mw=optimal_wire,
            verbose=True,
            first_year_only=self.first_year_only,
            assume_all_sellable=self.assume_all_sellable,
            skip_bess=True
        )

        return {
            'optimal_wire_capacity_mw': optimal_wire,
            'optimal_npc': result.fun,
            'optimal_profit': -result.fun,
            'optimization_result': result,
            'simulation_results': final_results,
            'evaluations': self.evaluation_count,
            'evaluation_history': pd.DataFrame(self.evaluation_history)
        }

    def grid_search(
        self,
        wire_capacity_range: np.ndarray = None
    ) -> pd.DataFrame:
        """
        Perform grid search over wire capacity values.

        Useful for visualization and understanding cost vs curtailment tradeoff.

        Parameters
        ----------
        wire_capacity_range : np.ndarray, optional
            Array of wire capacities to test (MW)
            If None, uses values from optimization_config

        Returns
        -------
        pd.DataFrame
            Results for all tested wire capacities
        """
        # Get grid search range from config if not specified
        if wire_capacity_range is None:
            algo_settings = self.optimization_config.get('algorithm_settings', {})
            grid_settings = algo_settings.get('grid_search', {})
            wire_capacity_range = np.array(
                grid_settings.get('wire_capacity_values', [40, 60, 80, 100, 120, 140, 160, 180, 200])
            )

        results = []
        total = len(wire_capacity_range)

        print(f"Running grid search over {total} wire capacity values...")
        print(f"Range: {wire_capacity_range.min():.0f} - {wire_capacity_range.max():.0f} MW")

        self.evaluation_history = []

        for i, wire_c in enumerate(wire_capacity_range):
            print(f"[{i+1}/{total}] Testing wire capacity: {wire_c:.0f} MW")

            npc = self.objective_function(wire_c)

            # Last evaluation is stored in history
            if self.evaluation_history:
                results.append(self.evaluation_history[-1])

        print("\nGrid search complete!")
        return pd.DataFrame(results)


def main():
    """Main entry point for wire-only sizing optimization."""
    parser = argparse.ArgumentParser(
        description="Optimize private wire capacity without BESS (no battery storage)"
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
        choices=['bounded', 'differential_evolution', 'grid'],
        help='Optimization method (overrides config file)'
    )
    parser.add_argument(
        '--curtailment-penalty',
        type=float,
        default=None,
        help='Curtailment penalty in EUR/MWh (opportunity cost of lost sales)'
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

    # Initialize wire-only optimizer
    optimizer = WireSizingOptimizer(
        simulator,
        optimization_config=optimization_config,
        curtailment_penalty_eur_per_mwh=args.curtailment_penalty,
        first_year_only=args.first_year_only,
        assume_all_sellable=args.assume_all_sellable
    )

    # Determine method (command-line overrides config)
    wire_optimization = optimization_config.get('wire_optimization', {})
    method = args.method if args.method is not None else wire_optimization.get('method', 'bounded')

    # Create output directory
    output_path = Path("outputs")
    output_path.mkdir(exist_ok=True)

    # Run optimization
    if method == 'grid':
        # Grid search
        results_df = optimizer.grid_search()

        print("\nGrid search results (sorted by Net Profit, best to worst):")
        print(results_df.sort_values('npc').head(10))

        # Save results
        results_df.to_csv(output_path / "wire_optimization_grid_search.csv", index=False)
        print(f"\nResults saved to {output_path / 'wire_optimization_grid_search.csv'}")

        # Find and display optimal
        optimal_row = results_df.loc[results_df['npc'].idxmin()]
        print(f"\nOptimal wire capacity from grid search: {optimal_row['wire_capacity_mw']:.1f} MW")
        print(f"NPC: €{optimal_row['npc']:,.0f}")
        print(f"Net Profit: €{optimal_row['net_profit']:,.0f}")
        print(f"Curtailment rate: {optimal_row['curtailment_rate']:.2%}")

    else:
        # Run optimization
        results = optimizer.optimize(method=method)

        # Save optimal configuration
        optimal_config = {
            'optimal_parameters': {
                'wire_capacity_mw': float(results['optimal_wire_capacity_mw']),
                'bess_energy_mwh': 0.0,  # No BESS
                'bess_power_mw': 0.0     # No BESS
            },
            'optimization': {
                'method': method,
                'evaluations': results['evaluations'],
                'optimal_npc': float(results['optimal_npc']),
                'optimal_profit': float(results['optimal_profit'])
            },
            'note': 'Optimization performed without BESS (skip_bess=True) - maximizing net profit'
        }

        with open(output_path / "optimal_wire_size_no_bess.yaml", 'w') as f:
            yaml.dump(optimal_config, f, default_flow_style=False)

        print(f"\nOptimal configuration saved to {output_path / 'optimal_wire_size_no_bess.yaml'}")

        # Save evaluation history
        eval_history_df = results['evaluation_history']
        eval_history_df.to_csv(output_path / "wire_optimization_history.csv", index=False)
        print(f"Evaluation history saved to {output_path / 'wire_optimization_history.csv'}")

        # Save detailed simulation results
        simulator.save_results()


if __name__ == "__main__":
    main()
