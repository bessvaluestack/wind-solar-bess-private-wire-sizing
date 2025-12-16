#!/usr/bin/env python3
"""
Optimization script with baseline (no-BESS) comparison.

This script extends the sizing optimization by comparing each BESS configuration
against a baseline scenario without BESS (skip_bess=True). This allows evaluating
the incremental value provided by the battery storage system.

For each configuration tested:
- Run simulation WITH BESS
- Run baseline simulation WITHOUT BESS (same wire capacity)
- Calculate delta metrics showing improvement from adding BESS
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from typing import Dict, Tuple, Callable, Optional
import yaml
from pathlib import Path
import argparse
import sys

from src.simulator import SystemSimulator


class SizingOptimizerWithBaseline:
    """
    Optimizes BESS and wire sizing while comparing to no-BESS baseline.
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
        Initialize sizing optimizer with baseline comparison.

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

        # Cache for baseline simulations (keyed by wire_capacity_mw)
        self.baseline_cache = {}

    def _run_baseline_simulation(self, wire_capacity_mw: float) -> Dict:
        """
        Run baseline simulation without BESS for given wire capacity.

        Results are cached to avoid redundant simulations.

        Parameters
        ----------
        wire_capacity_mw : float
            Wire capacity for baseline scenario

        Returns
        -------
        dict
            Baseline simulation results
        """
        # Check cache first
        if wire_capacity_mw in self.baseline_cache:
            return self.baseline_cache[wire_capacity_mw]

        # Run baseline simulation (no BESS)
        baseline_results = self.simulator.run_simulation(
            bess_energy_mwh=0,  # Not used when skip_bess=True
            bess_power_mw=0,    # Not used when skip_bess=True
            wire_capacity_mw=wire_capacity_mw,
            verbose=False,
            first_year_only=self.first_year_only,
            assume_all_sellable=self.assume_all_sellable,
            skip_bess=True  # This is the key: skip BESS simulation
        )

        # Cache the results
        self.baseline_cache[wire_capacity_mw] = baseline_results

        return baseline_results

    def _calculate_comparison_metrics(
        self,
        bess_results: Dict,
        baseline_results: Dict
    ) -> Dict:
        """
        Calculate comparison metrics between BESS and baseline scenarios.

        Parameters
        ----------
        bess_results : dict
            Results from simulation with BESS
        baseline_results : dict
            Results from baseline simulation (no BESS)

        Returns
        -------
        dict
            Comparison metrics showing deltas and improvements
        """
        bess_metrics = bess_results['metrics']
        baseline_metrics = baseline_results['metrics']

        # Extract key metrics
        bess_curtailment = bess_metrics['utilization']['curtailment_rate']
        baseline_curtailment = baseline_metrics['utilization']['curtailment_rate']

        bess_grid_import = bess_metrics['utilization']['grid_import_rate']
        baseline_grid_import = baseline_metrics['utilization']['grid_import_rate']

        bess_renewable_frac = bess_metrics['utilization']['renewable_fraction']
        baseline_renewable_frac = baseline_metrics['utilization']['renewable_fraction']

        bess_wire_delivery = bess_metrics['energy']['total_wire_delivery_mwh']
        baseline_wire_delivery = baseline_metrics['energy']['total_wire_delivery_mwh']

        bess_capex = bess_metrics['economics']['total_capex_eur']
        baseline_capex = baseline_metrics['economics']['total_capex_eur']

        # Calculate deltas (positive = improvement from BESS)
        curtailment_reduction = baseline_curtailment - bess_curtailment
        grid_import_reduction = baseline_grid_import - bess_grid_import
        renewable_fraction_increase = bess_renewable_frac - baseline_renewable_frac
        wire_delivery_increase = bess_wire_delivery - baseline_wire_delivery
        additional_capex = bess_capex - baseline_capex

        # Calculate revenue impact
        simulation_years = len(bess_results['timeseries']) * (
            self.simulator.config['simulation']['timestep_minutes'] / 60
        ) / 8760

        bess_annual_revenue = (bess_wire_delivery / simulation_years) * self.ppa_price
        baseline_annual_revenue = (baseline_wire_delivery / simulation_years) * self.ppa_price
        annual_revenue_increase = bess_annual_revenue - baseline_annual_revenue

        # NPV of additional revenue
        npv_additional_revenue = self._calculate_npv(
            annual_revenue_increase,
            self.project_lifetime,
            self.discount_rate
        )

        # Net benefit of adding BESS (NPV of additional revenue - additional capex)
        net_bess_benefit = npv_additional_revenue - additional_capex

        return {
            'baseline': {
                'curtailment_rate': baseline_curtailment,
                'grid_import_rate': baseline_grid_import,
                'renewable_fraction': baseline_renewable_frac,
                'wire_delivery_mwh': baseline_wire_delivery,
                'capex_eur': baseline_capex,
                'annual_revenue_eur': baseline_annual_revenue
            },
            'with_bess': {
                'curtailment_rate': bess_curtailment,
                'grid_import_rate': bess_grid_import,
                'renewable_fraction': bess_renewable_frac,
                'wire_delivery_mwh': bess_wire_delivery,
                'capex_eur': bess_capex,
                'annual_revenue_eur': bess_annual_revenue
            },
            'deltas': {
                'curtailment_reduction': curtailment_reduction,
                'grid_import_reduction': grid_import_reduction,
                'renewable_fraction_increase': renewable_fraction_increase,
                'wire_delivery_increase_mwh': wire_delivery_increase,
                'additional_capex_eur': additional_capex,
                'annual_revenue_increase_eur': annual_revenue_increase,
                'npv_additional_revenue_eur': npv_additional_revenue,
                'net_bess_benefit_eur': net_bess_benefit
            },
            'relative_improvements': {
                'curtailment_reduction_pct': (curtailment_reduction / baseline_curtailment * 100)
                    if baseline_curtailment > 0 else 0,
                'grid_import_reduction_pct': (grid_import_reduction / baseline_grid_import * 100)
                    if baseline_grid_import > 0 else 0,
                'wire_delivery_increase_pct': (wire_delivery_increase / baseline_wire_delivery * 100)
                    if baseline_wire_delivery > 0 else 0
            }
        }

    def objective_function(self, x: np.ndarray) -> float:
        """
        Objective function to minimize: Net Present Cost (NPC).

        Also compares to baseline (no-BESS) scenario.

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
            # Run simulation WITH BESS
            bess_results = self.simulator.run_simulation(
                bess_energy_mwh=bess_energy_mwh,
                bess_power_mw=bess_power_mw,
                wire_capacity_mw=wire_capacity_mw,
                verbose=False,
                first_year_only=self.first_year_only,
                assume_all_sellable=self.assume_all_sellable
            )

            # Run baseline simulation WITHOUT BESS
            baseline_results = self._run_baseline_simulation(wire_capacity_mw)

            # Calculate comparison metrics
            comparison = self._calculate_comparison_metrics(bess_results, baseline_results)

            metrics = bess_results['metrics']

            # Check constraints
            curtailment_rate = metrics['utilization']['curtailment_rate']
            grid_import_rate = metrics['utilization']['grid_import_rate']

            # Penalty for constraint violations
            penalty = 0
            if self.max_curtailment_rate is not None and curtailment_rate > self.max_curtailment_rate:
                penalty += self.curtailment_penalty * (curtailment_rate - self.max_curtailment_rate)

            if self.max_grid_import_rate is not None and grid_import_rate > self.max_grid_import_rate:
                penalty += self.grid_import_penalty * (grid_import_rate - self.max_grid_import_rate)

            # Calculate Net Present Cost
            capex = metrics['economics']['total_capex_eur']

            # Annual revenue from PPA
            total_delivery_mwh = metrics['energy']['total_wire_delivery_mwh']
            simulation_years = len(bess_results['timeseries']) * (
                self.simulator.config['simulation']['timestep_minutes'] / 60
            ) / 8760
            annual_revenue = (total_delivery_mwh / simulation_years) * self.ppa_price

            # NPV of revenue stream
            npv_revenue = self._calculate_npv(
                annual_revenue,
                self.project_lifetime,
                self.discount_rate
            )

            # Net present cost
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
                print(f"\n  Comparison to NO-BESS baseline:")
                print(f"    Curtailment: {curtailment_rate:.1%} (baseline: {comparison['baseline']['curtailment_rate']:.1%}, "
                      f"Δ: {comparison['deltas']['curtailment_reduction']:.1%})")
                print(f"    Grid Import: {grid_import_rate:.1%} (baseline: {comparison['baseline']['grid_import_rate']:.1%}, "
                      f"Δ: {comparison['deltas']['grid_import_reduction']:.1%})")
                print(f"    Wire Delivery: {total_delivery_mwh:,.0f} MWh (baseline: {comparison['baseline']['wire_delivery_mwh']:,.0f} MWh, "
                      f"Δ: {comparison['deltas']['wire_delivery_increase_mwh']:,.0f} MWh)")
                print(f"    Net BESS Benefit: €{comparison['deltas']['net_bess_benefit_eur']:,.0f}")

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
        Run optimization to find optimal sizing with baseline comparison.

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
            Optimization results including optimal parameters, metrics, and baseline comparison
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
        print("SIZING OPTIMIZATION WITH BASELINE COMPARISON")
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
        print(f"\nNote: Each configuration will be compared to no-BESS baseline")
        print(separator)

        self.evaluation_count = 0
        self.best_cost = np.inf
        self.baseline_cache = {}  # Clear cache

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
        print("OPTIMIZATION COMPLETE - FINAL COMPARISON")
        print("=" * 70)

        # Run optimal BESS configuration
        final_bess_results = self.simulator.run_simulation(
            bess_energy_mwh=optimal_bess_energy,
            bess_power_mw=optimal_bess_power,
            wire_capacity_mw=optimal_wire,
            verbose=True,
            first_year_only=self.first_year_only,
            assume_all_sellable=self.assume_all_sellable
        )

        # Run baseline for optimal wire capacity
        final_baseline_results = self._run_baseline_simulation(optimal_wire)

        # Calculate final comparison
        final_comparison = self._calculate_comparison_metrics(
            final_bess_results,
            final_baseline_results
        )

        # Print detailed comparison
        print("\n" + "=" * 70)
        print("BASELINE VS BESS COMPARISON")
        print("=" * 70)
        print("\nBaseline (No BESS):")
        print(f"  Curtailment Rate:    {final_comparison['baseline']['curtailment_rate']:.2%}")
        print(f"  Grid Import Rate:    {final_comparison['baseline']['grid_import_rate']:.2%}")
        print(f"  Renewable Fraction:  {final_comparison['baseline']['renewable_fraction']:.2%}")
        print(f"  Wire Delivery:       {final_comparison['baseline']['wire_delivery_mwh']:,.0f} MWh")
        print(f"  Capex:               €{final_comparison['baseline']['capex_eur']:,.0f}")
        print(f"  Annual Revenue:      €{final_comparison['baseline']['annual_revenue_eur']:,.0f}")

        print("\nWith BESS:")
        print(f"  Curtailment Rate:    {final_comparison['with_bess']['curtailment_rate']:.2%}")
        print(f"  Grid Import Rate:    {final_comparison['with_bess']['grid_import_rate']:.2%}")
        print(f"  Renewable Fraction:  {final_comparison['with_bess']['renewable_fraction']:.2%}")
        print(f"  Wire Delivery:       {final_comparison['with_bess']['wire_delivery_mwh']:,.0f} MWh")
        print(f"  Capex:               €{final_comparison['with_bess']['capex_eur']:,.0f}")
        print(f"  Annual Revenue:      €{final_comparison['with_bess']['annual_revenue_eur']:,.0f}")

        print("\nImprovements from BESS:")
        print(f"  Curtailment Reduction:       {final_comparison['deltas']['curtailment_reduction']:.2%} "
              f"({final_comparison['relative_improvements']['curtailment_reduction_pct']:.1f}% reduction)")
        print(f"  Grid Import Reduction:       {final_comparison['deltas']['grid_import_reduction']:.2%} "
              f"({final_comparison['relative_improvements']['grid_import_reduction_pct']:.1f}% reduction)")
        print(f"  Renewable Fraction Increase: {final_comparison['deltas']['renewable_fraction_increase']:.2%}")
        print(f"  Wire Delivery Increase:      {final_comparison['deltas']['wire_delivery_increase_mwh']:,.0f} MWh "
              f"({final_comparison['relative_improvements']['wire_delivery_increase_pct']:.1f}% increase)")
        print(f"  Additional Capex:            €{final_comparison['deltas']['additional_capex_eur']:,.0f}")
        print(f"  Annual Revenue Increase:     €{final_comparison['deltas']['annual_revenue_increase_eur']:,.0f}")
        print(f"  NPV Additional Revenue:      €{final_comparison['deltas']['npv_additional_revenue_eur']:,.0f}")
        print(f"  Net BESS Benefit:            €{final_comparison['deltas']['net_bess_benefit_eur']:,.0f}")
        print("=" * 70)

        return {
            'optimal_bess_energy_mwh': optimal_bess_energy,
            'optimal_bess_power_mw': optimal_bess_power,
            'optimal_wire_capacity_mw': optimal_wire,
            'optimal_npc': result.fun,
            'optimization_result': result,
            'simulation_results': final_bess_results,
            'baseline_results': final_baseline_results,
            'comparison_metrics': final_comparison,
            'evaluations': self.evaluation_count
        }

    def grid_search(
        self,
        bess_energy_range: np.ndarray = None,
        bess_power_range: np.ndarray = None,
        wire_capacity_range: np.ndarray = None
    ) -> pd.DataFrame:
        """
        Perform grid search with baseline comparison.

        Parameters
        ----------
        bess_energy_range : np.ndarray, optional
            Array of BESS energy capacities to test
        bess_power_range : np.ndarray, optional
            Array of BESS power capacities to test
        wire_capacity_range : np.ndarray, optional
            Array of wire capacities to test

        Returns
        -------
        pd.DataFrame
            Results for all combinations including baseline comparisons
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
        self.baseline_cache = {}  # Clear cache

        total_combinations = len(bess_energy_range) * len(bess_power_range) * len(wire_capacity_range)
        print(f"Running grid search with baseline comparison over {total_combinations} combinations...")

        count = 0
        for wire_c in wire_capacity_range:
            # Run baseline once per wire capacity
            print(f"\nRunning baseline for wire capacity: {wire_c:.0f} MW")
            baseline_results = self._run_baseline_simulation(wire_c)
            baseline_metrics = baseline_results['metrics']

            for bess_e in bess_energy_range:
                for bess_p in bess_power_range:
                    count += 1
                    print(f"\r[{count}/{total_combinations}] Testing: BESS={bess_e:.0f}MWh/{bess_p:.0f}MW, Wire={wire_c:.0f}MW", end='')

                    npc = self.objective_function([bess_e, bess_p, wire_c])

                    # Get last simulation metrics
                    if self.simulator.results is not None:
                        bess_results = self.simulator.results
                        comparison = self._calculate_comparison_metrics(bess_results, baseline_results)

                        metrics = bess_results['metrics']
                        results.append({
                            # Configuration
                            'bess_energy_mwh': bess_e,
                            'bess_power_mw': bess_p,
                            'wire_capacity_mw': wire_c,
                            # BESS scenario
                            'bess_npc': npc,
                            'bess_curtailment_rate': metrics['utilization']['curtailment_rate'],
                            'bess_grid_import_rate': metrics['utilization']['grid_import_rate'],
                            'bess_renewable_fraction': metrics['utilization']['renewable_fraction'],
                            'bess_wire_delivery_mwh': metrics['energy']['total_wire_delivery_mwh'],
                            'bess_capex': metrics['economics']['total_capex_eur'],
                            # Baseline scenario
                            'baseline_curtailment_rate': baseline_metrics['utilization']['curtailment_rate'],
                            'baseline_grid_import_rate': baseline_metrics['utilization']['grid_import_rate'],
                            'baseline_renewable_fraction': baseline_metrics['utilization']['renewable_fraction'],
                            'baseline_wire_delivery_mwh': baseline_metrics['energy']['total_wire_delivery_mwh'],
                            'baseline_capex': baseline_metrics['economics']['total_capex_eur'],
                            # Deltas
                            'delta_curtailment_reduction': comparison['deltas']['curtailment_reduction'],
                            'delta_grid_import_reduction': comparison['deltas']['grid_import_reduction'],
                            'delta_renewable_fraction_increase': comparison['deltas']['renewable_fraction_increase'],
                            'delta_wire_delivery_increase_mwh': comparison['deltas']['wire_delivery_increase_mwh'],
                            'delta_additional_capex_eur': comparison['deltas']['additional_capex_eur'],
                            'net_bess_benefit_eur': comparison['deltas']['net_bess_benefit_eur'],
                            # Relative improvements
                            'curtailment_reduction_pct': comparison['relative_improvements']['curtailment_reduction_pct'],
                            'grid_import_reduction_pct': comparison['relative_improvements']['grid_import_reduction_pct'],
                            'wire_delivery_increase_pct': comparison['relative_improvements']['wire_delivery_increase_pct']
                        })

        print("\nGrid search complete!")
        return pd.DataFrame(results)


def main():
    """Main entry point for sizing optimization with baseline comparison."""
    parser = argparse.ArgumentParser(
        description="Optimize BESS and private wire sizing with no-BESS baseline comparison"
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

    # Initialize optimizer with baseline comparison
    optimizer = SizingOptimizerWithBaseline(
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
        # Grid search with baseline comparison
        results_df = optimizer.grid_search()
        print("\nGrid search results (sorted by NPC):")
        print(results_df.sort_values('bess_npc').head(10))

        # Save results
        output_path = Path("outputs")
        output_path.mkdir(exist_ok=True)
        results_df.to_csv(output_path / "grid_search_with_baseline.csv", index=False)
        print(f"\nResults saved to {output_path / 'grid_search_with_baseline.csv'}")

    else:
        # Run optimization with baseline comparison
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
            },
            'baseline_comparison': {
                'baseline_metrics': {
                    k: float(v) if isinstance(v, (int, float, np.number)) else v
                    for k, v in results['comparison_metrics']['baseline'].items()
                },
                'with_bess_metrics': {
                    k: float(v) if isinstance(v, (int, float, np.number)) else v
                    for k, v in results['comparison_metrics']['with_bess'].items()
                },
                'improvements': {
                    k: float(v) if isinstance(v, (int, float, np.number)) else v
                    for k, v in results['comparison_metrics']['deltas'].items()
                }
            }
        }

        with open(output_path / "optimal_sizing_with_baseline.yaml", 'w') as f:
            yaml.dump(optimal_config, f, default_flow_style=False)

        print(f"\nOptimal configuration saved to {output_path / 'optimal_sizing_with_baseline.yaml'}")

        # Save detailed results
        simulator.save_results()


if __name__ == "__main__":
    main()
