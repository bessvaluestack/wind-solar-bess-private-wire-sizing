"""
Battery Energy Storage System (BESS) dispatch optimization module.

Implements optimal battery dispatch using convex optimization (CVXPY)
with constraints on:
- SoC limits
- Power limits
- Daily cycling limits
- Throughput-based degradation
- Private wire capacity
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Tuple, Dict, Optional


class BESSDispatcher:
    """
    Optimizes battery dispatch to minimize curtailment and maximize value.

    The dispatcher solves a convex optimization problem to find optimal
    charge/discharge schedules subject to physical and operational constraints.
    """

    def __init__(
        self,
        energy_capacity_mwh: float,
        power_capacity_mw: float,
        min_soc: float = 0.1,
        max_soc: float = 0.9,
        initial_soc: float = 0.5,
        charge_efficiency: float = 0.95,
        discharge_efficiency: float = 0.95,
        max_daily_cycles: float = 1.5,
        degradation_per_cycle: float = 0.0001
    ):
        """
        Initialize BESS dispatcher.

        Parameters
        ----------
        energy_capacity_mwh : float
            Battery energy capacity (MWh)
        power_capacity_mw : float
            Battery power capacity (MW)
        min_soc : float
            Minimum state of charge (0-1)
        max_soc : float
            Maximum state of charge (0-1)
        initial_soc : float
            Initial state of charge (0-1)
        charge_efficiency : float
            One-way charging efficiency (0-1)
        discharge_efficiency : float
            One-way discharging efficiency (0-1)
        max_daily_cycles : float
            Maximum equivalent full cycles per day
        degradation_per_cycle : float
            Capacity degradation per full cycle
        """
        self.energy_capacity_mwh = energy_capacity_mwh
        self.power_capacity_mw = power_capacity_mw
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.initial_soc = initial_soc
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.max_daily_cycles = max_daily_cycles
        self.degradation_per_cycle = degradation_per_cycle

        # Track cumulative degradation
        self.cumulative_degradation = 0.0
        self.current_capacity_mwh = energy_capacity_mwh

    def optimize_dispatch(
        self,
        generation_mw: np.ndarray,
        load_mw: np.ndarray,
        wire_capacity_mw: float,
        timestep_hours: float = 0.25,
        verbose: bool = True,
        aux_load_mw: np.ndarray | float = 0.0
    ) -> Dict[str, np.ndarray]:
        """
        Optimize battery dispatch for given generation and load profiles.

        Strategy:
        - Charge battery when generation exceeds wire capacity + load
        - Discharge battery when generation is below load (up to wire capacity)
        - Minimize curtailment (wasted energy)
        - Respect all battery constraints

        Parameters
        ----------
        generation_mw : np.ndarray
            Total generation (wind + solar) in MW
        load_mw : np.ndarray
            Datacenter load in MW
        wire_capacity_mw : float
            Private wire transmission capacity in MW
        timestep_hours : float
            Timestep duration in hours (default: 0.25 for 15-min)
        verbose : bool
            Print optimization details
        aux_load_mw : np.ndarray or float
            Auxiliary load in MW (deducted from generation or served by BESS).
            Can be a scalar (constant load) or array (time-varying load profile)

        Returns
        -------
        dict
            Dictionary containing:
            - 'charge_mw': Charging power (MW, positive)
            - 'discharge_mw': Discharging power (MW, positive)
            - 'soc': State of charge (0-1)
            - 'curtailment_mw': Curtailed generation (MW)
            - 'grid_import_mw': Power imported from grid (MW)
            - 'wire_flow_mw': Net flow through private wire (MW, positive = to datacenter)
            - 'unserved_aux_load_mw': Auxiliary load that could not be served (MW)
            - 'cycles': Cumulative full cycles
        """
        n = len(generation_mw)

        # Convert aux_load_mw to array if it's a scalar
        if np.isscalar(aux_load_mw):
            aux_load_array = np.full(n, aux_load_mw)
        else:
            aux_load_array = np.asarray(aux_load_mw)
            if len(aux_load_array) != n:
                raise ValueError(f"aux_load_mw array length ({len(aux_load_array)}) must match generation length ({n})")

        # Decision variables
        charge = cp.Variable(n, nonneg=True)  # Charging power (MW)
        discharge = cp.Variable(n, nonneg=True)  # Discharging power (MW)
        soc = cp.Variable(n + 1)  # State of charge (fraction)
        curtailment = cp.Variable(n, nonneg=True)  # Curtailed generation (MW)
        wire_flow = cp.Variable(n)  # Net flow through wire (positive = to datacenter)
        unserved_aux_load = cp.Variable(n, nonneg=True)  # Unserved auxiliary load (MW)

        # Constraints list
        constraints = []

        # Initial SoC
        constraints.append(soc[0] == self.initial_soc)

        # SoC dynamics (energy balance)
        for t in range(n):
            constraints.append(
                soc[t + 1] == soc[t]
                + (charge[t] * self.charge_efficiency * timestep_hours / self.current_capacity_mwh)
                - (discharge[t] / self.discharge_efficiency * timestep_hours / self.current_capacity_mwh)
            )

        # SoC limits
        constraints.append(soc >= self.min_soc)
        constraints.append(soc <= self.max_soc)

        # Power limits
        constraints.append(charge <= self.power_capacity_mw)
        constraints.append(discharge <= self.power_capacity_mw)

        # Wire flow constraints with auxiliary load
        # Wire flow = generation - curtailment - charge + discharge - aux_load + unserved_aux_load
        # Auxiliary load is served first (from generation or BESS), remaining goes to wire
        for t in range(n):
            constraints.append(
                wire_flow[t] == generation_mw[t] - curtailment[t] - charge[t] + discharge[t] - aux_load_array[t] + unserved_aux_load[t]
            )
            # Unserved aux load occurs when available power < aux load
            constraints.append(
                unserved_aux_load[t] >= aux_load_array[t] - (generation_mw[t] - curtailment[t] - charge[t] + discharge[t])
            )

        # Wire capacity constraints
        constraints.append(wire_flow <= wire_capacity_mw)
        constraints.append(wire_flow >= 0)  # No reverse flow

        # Energy balance: wire flow must meet load
        # If wire_flow < load, we need grid import (but this is implicitly handled)
        # We want to maximize wire flow to minimize grid import

        # Daily cycling constraint
        # Total throughput per day should not exceed max_daily_cycles * capacity
        timesteps_per_day = int(24 / timestep_hours)

        for day_start in range(0, n, timesteps_per_day):
            day_end = min(day_start + timesteps_per_day, n)
            daily_throughput = cp.sum(charge[day_start:day_end]) * timestep_hours
            constraints.append(
                daily_throughput <= self.max_daily_cycles * self.current_capacity_mwh
            )

        # Objective: Minimize curtailment + penalize grid import
        # Grid import = max(0, load - wire_flow)
        grid_import = cp.Variable(n, nonneg=True)
        for t in range(n):
            constraints.append(grid_import[t] >= load_mw[t] - wire_flow[t])

        # Weighted objective
        # Primary: minimize unserved aux load (highest priority)
        # Secondary: minimize curtailment
        # Tertiary: minimize grid import (encourage battery to help during low generation)
        objective = cp.Minimize(
            10000 * cp.sum(unserved_aux_load) +  # Very high penalty for unserved aux load
            1000 * cp.sum(curtailment) +  # High penalty for curtailment
            1 * cp.sum(grid_import)  # Lower penalty for grid import
        )

        # Solve optimization problem
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.ECOS, verbose=verbose)

            if problem.status not in ['optimal', 'optimal_inaccurate']:
                print(f"Warning: Optimization status: {problem.status}")

        except Exception as e:
            print(f"Optimization failed: {e}")
            # Return zero dispatch if optimization fails
            return self._zero_dispatch(n)

        # Calculate grid import
        grid_import_mw = np.maximum(0, load_mw - wire_flow.value)

        # Calculate cumulative cycles
        total_throughput_mwh = np.sum(charge.value) * timestep_hours
        cycles = total_throughput_mwh / self.current_capacity_mwh

        # Update degradation
        self.cumulative_degradation += cycles * self.degradation_per_cycle
        self.current_capacity_mwh = self.energy_capacity_mwh * (1 - self.cumulative_degradation)

        # Return results
        return {
            'charge_mw': charge.value,
            'discharge_mw': discharge.value,
            'soc': soc.value[:-1],  # Exclude final SoC
            'curtailment_mw': curtailment.value,
            'grid_import_mw': grid_import_mw,
            'wire_flow_mw': wire_flow.value,
            'unserved_aux_load_mw': unserved_aux_load.value,
            'cycles': cycles,
            'objective_value': problem.value
        }

    def _zero_dispatch(self, n: int) -> Dict[str, np.ndarray]:
        """Return zero dispatch in case of optimization failure."""
        return {
            'charge_mw': np.zeros(n),
            'discharge_mw': np.zeros(n),
            'soc': np.full(n, self.initial_soc),
            'curtailment_mw': np.zeros(n),
            'grid_import_mw': np.zeros(n),
            'wire_flow_mw': np.zeros(n),
            'unserved_aux_load_mw': np.zeros(n),
            'cycles': 0.0,
            'objective_value': np.inf
        }

    def reset(self):
        """Reset battery state for new simulation."""
        self.cumulative_degradation = 0.0
        self.current_capacity_mwh = self.energy_capacity_mwh

    def simulate_multiyear(
        self,
        generation_mw: np.ndarray,
        load_mw: np.ndarray,
        wire_capacity_mw: float,
        timestep_hours: float = 0.25,
        timesteps_per_year: int = 35040,
        verbose: bool = False,
        timestamps: Optional[pd.DatetimeIndex] = None,
        aux_load_mw: np.ndarray | float = 0.0
    ) -> pd.DataFrame:
        """
        Simulate battery dispatch over multiple years.

        Parameters
        ----------
        generation_mw : np.ndarray
            Multi-year generation profile
        load_mw : np.ndarray
            Multi-year load profile
        wire_capacity_mw : float
            Wire capacity
        timestep_hours : float
            Timestep in hours
        timesteps_per_year : int
            Number of timesteps per year (35040 for 15-min timesteps)
        verbose : bool
            Print progress
        timestamps : pd.DatetimeIndex, optional
            Timestamps for the data. If provided, will determine years from actual dates.
        aux_load_mw : np.ndarray or float
            Auxiliary load in MW (deducted from generation or served by BESS).
            Can be a scalar (constant load) or array (time-varying load profile)

        Returns
        -------
        pd.DataFrame
            Combined results for all years
        """
        self.reset()

        n_total = len(generation_mw)

        # Convert aux_load_mw to array if it's a scalar
        if np.isscalar(aux_load_mw):
            aux_load_array = np.full(n_total, aux_load_mw)
        else:
            aux_load_array = np.asarray(aux_load_mw)
            if len(aux_load_array) != n_total:
                raise ValueError(f"aux_load_mw array length ({len(aux_load_array)}) must match generation length ({n_total})")

        # Determine years from timestamps if provided
        if timestamps is not None:
            years = timestamps.year.unique()
            n_years = len(years)
            year_labels = years
        else:
            n_years = int(np.ceil(n_total / timesteps_per_year))
            year_labels = range(1, n_years + 1)

        all_results = []
        timesteps_per_month = int(30 * 24 / timestep_hours)  # Approximate month

        for year_idx, year_label in enumerate(year_labels):
            if timestamps is not None:
                # Use actual year boundaries
                year_mask = timestamps.year == year_label
                year_indices = np.where(year_mask)[0]
                if len(year_indices) == 0:
                    continue
                start_idx = year_indices[0]
                end_idx = year_indices[-1] + 1
            else:
                # Use fixed chunks
                start_idx = year_idx * timesteps_per_year
                end_idx = min((year_idx + 1) * timesteps_per_year, n_total)

            if verbose:
                print(f"Simulating year {year_label} ({year_idx + 1}/{n_years})...")

            year_gen = generation_mw[start_idx:end_idx]
            year_load = load_mw[start_idx:end_idx]
            year_aux = aux_load_array[start_idx:end_idx]

            # Optimize in monthly chunks for better performance
            n_timesteps = len(year_gen)
            year_results_list = []

            for month_start in range(0, n_timesteps, timesteps_per_month):
                month_end = min(month_start + timesteps_per_month, n_timesteps)

                month_gen = year_gen[month_start:month_end]
                month_load = year_load[month_start:month_end]
                month_aux = year_aux[month_start:month_end]

                results = self.optimize_dispatch(
                    month_gen,
                    month_load,
                    wire_capacity_mw,
                    timestep_hours,
                    verbose=False,
                    aux_load_mw=month_aux
                )

                # Update initial SoC for next chunk
                self.initial_soc = results['soc'][-1]

                # Store month results
                month_results = pd.DataFrame({
                    'charge_mw': results['charge_mw'],
                    'discharge_mw': results['discharge_mw'],
                    'soc': results['soc'],
                    'curtailment_mw': results['curtailment_mw'],
                    'grid_import_mw': results['grid_import_mw'],
                    'wire_flow_mw': results['wire_flow_mw'],
                    'unserved_aux_load_mw': results['unserved_aux_load_mw']
                })
                year_results_list.append(month_results)

            # Combine monthly results for the year
            year_results = pd.concat(year_results_list, ignore_index=True)
            year_results['year'] = year_label
            year_results['timestep'] = range(len(year_results))

            all_results.append(year_results)

            if verbose:
                total_cycles = np.sum([r['charge_mw'].sum() for r in year_results_list]) * timestep_hours / self.current_capacity_mwh
                print(f"  Year {year_label} cycles: {total_cycles:.2f}")
                print(f"  Cumulative degradation: {self.cumulative_degradation * 100:.2f}%")

        return pd.concat(all_results, ignore_index=True)


if __name__ == "__main__":
    # Example usage
    n_timesteps = 35040  # One year of 15-min data

    # Dummy data
    np.random.seed(42)
    generation = 50 + 30 * np.random.rand(n_timesteps)
    load = 40 + 10 * np.random.rand(n_timesteps)

    dispatcher = BESSDispatcher(
        energy_capacity_mwh=200.0,
        power_capacity_mw=50.0,
        min_soc=0.1,
        max_soc=0.9
    )

    results = dispatcher.optimize_dispatch(
        generation,
        load,
        wire_capacity_mw=60.0
    )

    print(f"Optimization completed")
    print(f"Total curtailment: {results['curtailment_mw'].sum() * 0.25:.2f} MWh")
    print(f"Total grid import: {results['grid_import_mw'].sum() * 0.25:.2f} MWh")
    print(f"Battery cycles: {results['cycles']:.2f}")
