"""
Main simulation runner for wind + solar + BESS + private wire system.

Integrates generation data, load profiles, and BESS dispatch to simulate
the complete system and calculate key performance metrics.
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

from src.load_generator import DatacenterLoadGenerator
from src.bess_dispatch import BESSDispatcher
from src.aux_load_generator import AuxiliaryLoadGenerator


def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types for YAML serialization.

    Parameters
    ----------
    obj : Any
        Object to convert (can be dict, list, numpy type, or native type)

    Returns
    -------
    Any
        Object with all numpy types converted to native Python types
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class SystemSimulator:
    """
    Simulates the integrated renewable energy + BESS + private wire system.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize simulator with configuration.

        Parameters
        ----------
        config_path : str
            Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.results = None

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def load_generation_data(self) -> pd.DataFrame:
        """
        Load wind and solar generation time series.

        Expected CSV format:
        - Interval_Date_Time (with timezone), Site_Generation_Wind_kWh/Solar_kWh
        OR timestamp, wind_mw, solar_mw

        Returns
        -------
        pd.DataFrame
            Combined generation data with 'total_generation_mw' column
        """
        wind_file = self.config['generation_data']['wind_file']
        solar_file = self.config['generation_data']['solar_file']

        # Add .csv extension if not present
        if not wind_file.endswith('.csv'):
            wind_file += '.csv'
        if not solar_file.endswith('.csv'):
            solar_file += '.csv'

        # Load wind data
        wind_df = pd.read_csv(wind_file)
        # Handle both 'Interval_Date_Time' and 'timestamp' column names
        timestamp_col = 'Interval_Date_Time' if 'Interval_Date_Time' in wind_df.columns else 'timestamp'
        wind_df[timestamp_col] = pd.to_datetime(wind_df[timestamp_col])
        wind_df.rename(columns={timestamp_col: 'timestamp'}, inplace=True)
        wind_df.set_index('timestamp', inplace=True)

        # Load solar data
        solar_df = pd.read_csv(solar_file)
        timestamp_col = 'Interval_Date_Time' if 'Interval_Date_Time' in solar_df.columns else 'timestamp'
        solar_df[timestamp_col] = pd.to_datetime(solar_df[timestamp_col])
        solar_df.rename(columns={timestamp_col: 'timestamp'}, inplace=True)
        solar_df.set_index('timestamp', inplace=True)

        # Combine - convert from kWh (15-min) to MW
        # kWh / 0.25 hours = kW, then kW / 1000 = MW
        # So: kWh / 250 = MW for 15-min intervals
        # Explicitly preserve the timestamp index
        combined_df = pd.DataFrame(index=wind_df.index)
        combined_df['wind_mw'] = wind_df.iloc[:, 0] / 250.0
        combined_df['solar_mw'] = solar_df.iloc[:, 0] / 250.0
        combined_df['total_generation_mw'] = (wind_df.iloc[:, 0] + solar_df.iloc[:, 0]) / 250.0

        return combined_df

    def generate_load_profile(self, n_years: int, load_csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate or load datacenter load profile.

        Parameters
        ----------
        n_years : int
            Number of years to simulate
        load_csv_path : str, optional
            Path to CSV file with load data. If provided, loads from CSV instead of generating.

        Returns
        -------
        pd.DataFrame
            Load profile with 'load_mw' column
        """
        if load_csv_path:
            # Load from CSV file
            generator = DatacenterLoadGenerator(
                annual_peak_mw=0,  # Not used when loading from CSV
                capacity_factor=0,
                base_load_ratio=0
            )
            load_df = generator.load_from_csv(load_csv_path)
            return load_df

        # Generate new load profile
        dc_config = self.config['datacenter']

        generator = DatacenterLoadGenerator(
            annual_peak_mw=dc_config['annual_peak_mw'],
            capacity_factor=dc_config['capacity_factor'],
            base_load_ratio=dc_config['base_load_ratio'],
            random_seed=dc_config.get('random_seed')
        )

        load_df = generator.generate_multiyear_profile(
            n_years=n_years,
            timestep_minutes=self.config['simulation']['timestep_minutes']
        )

        return load_df

    def generate_aux_load_profile(
        self,
        generation_df: pd.DataFrame,
        aux_load_csv_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate or load auxiliary load profile.

        Supports three modes:
        1. 'static': Constant auxiliary load (from config aux_load_kw)
        2. 'generated': Generate profile based on wind/solar generation
        3. 'csv': Load from CSV file

        Parameters
        ----------
        generation_df : pd.DataFrame
            Generation data with 'wind_mw' and 'solar_mw' columns
        aux_load_csv_path : str, optional
            Path to CSV file with aux load data. Overrides config setting.

        Returns
        -------
        np.ndarray
            Auxiliary load profile in MW (same length as generation_df)
        """
        aux_load_config = self.config.get('auxiliary_load', {})
        mode = aux_load_config.get('mode', 'static')

        # Override mode if CSV path is provided
        if aux_load_csv_path is not None:
            mode = 'csv'

        if mode == 'static':
            # Static constant load
            aux_load_kw = aux_load_config.get('aux_load_kw', 0.0)
            aux_load_mw = aux_load_kw / 1000.0
            return np.full(len(generation_df), aux_load_mw)

        elif mode == 'csv':
            # Load from CSV file
            csv_path = aux_load_csv_path or aux_load_config.get('aux_load_csv')
            if csv_path is None:
                raise ValueError("aux_load_csv path must be specified in config or as parameter when mode='csv'")

            generator = AuxiliaryLoadGenerator()
            aux_df = generator.load_from_csv(csv_path)

            # Align with generation data
            min_len = min(len(generation_df), len(aux_df))
            aux_load_mw = aux_df['total_aux_mw'].values[:min_len]

            # Pad or truncate to match generation data length
            if len(aux_load_mw) < len(generation_df):
                # Pad with zeros if aux load is shorter
                aux_load_mw = np.pad(aux_load_mw, (0, len(generation_df) - len(aux_load_mw)), mode='constant')
            elif len(aux_load_mw) > len(generation_df):
                # Truncate if aux load is longer
                aux_load_mw = aux_load_mw[:len(generation_df)]

            return aux_load_mw

        elif mode == 'generated':
            # Generate based on wind/solar profiles
            gen_config = aux_load_config.get('generator', {})

            generator = AuxiliaryLoadGenerator(
                num_wind_turbines=gen_config.get('num_wind_turbines', 21),
                wind_turbine_capacity_mw=gen_config.get('wind_turbine_capacity_mw', 8.0),
                solar_capacity_mw=gen_config.get('solar_capacity_mw'),  # None = auto-infer
                inverter_standby_kw_per_mw=gen_config.get('inverter_standby_kw_per_mw', 0.5),
                inverter_operating_kw_per_mw=gen_config.get('inverter_operating_kw_per_mw', 0.2),
                fixed_aux_load_kw=gen_config.get('fixed_aux_load_kw', 100.0),
                wind_turbine_base_aux_kw=gen_config.get('wind_turbine_base_aux_kw', 5.0),
                wind_turbine_aux_per_mw=gen_config.get('wind_turbine_aux_per_mw', 0.3),
                wind_low_generation_threshold_mw=gen_config.get('wind_low_generation_threshold_mw', 0.5),
                wind_nacelle_active_probability=gen_config.get('wind_nacelle_active_probability', 0.05),
                random_seed=gen_config.get('random_seed')
            )

            aux_df = generator.generate_profile(
                wind_generation_mw=generation_df['wind_mw'].values,
                solar_generation_mw=generation_df['solar_mw'].values,
                timestamps=generation_df.index
            )

            return aux_df['total_aux_mw'].values

        else:
            raise ValueError(f"Invalid auxiliary load mode: {mode}. Must be 'static', 'csv', or 'generated'")

    def run_simulation(
        self,
        bess_energy_mwh: Optional[float] = None,
        bess_power_mw: Optional[float] = None,
        wire_capacity_mw: Optional[float] = None,
        verbose: bool = True,
        load_csv_path: Optional[str] = None,
        aux_load_csv_path: Optional[str] = None,
        first_year_only: bool = False,
        assume_all_sellable: bool = False,
        skip_bess: bool = False
    ) -> Dict:
        """
        Run complete system simulation.

        Parameters
        ----------
        bess_energy_mwh : float, optional
            Override BESS energy capacity (uses config if not provided)
        bess_power_mw : float, optional
            Override BESS power capacity (uses config if not provided)
        wire_capacity_mw : float, optional
            Override wire capacity (uses config if not provided)
        verbose : bool
            Print progress and results
        load_csv_path : str, optional
            Path to CSV file with load data. If provided, loads from CSV instead of generating.
        aux_load_csv_path : str, optional
            Path to CSV file with aux load data. If provided, overrides config mode and loads from CSV.
        first_year_only : bool
            If True, run simulation for first year only (default: False)
        assume_all_sellable : bool
            If True, assume all energy can be sold by setting load to wire_capacity_mw * 8760 MWh/year
            (constant load at wire rating). Overrides load_csv_path and datacenter config. (default: False)
        skip_bess : bool
            If True, skip BESS simulation and run stats calculator only (no battery storage).
            All BESS-related values will be zero in the results. (default: False)

        Returns
        -------
        dict
            Dictionary with simulation results and metrics
        """
        # Use config values if not overridden
        bess_config = self.config['bess']
        wire_config = self.config['private_wire']
        sim_config = self.config['simulation']
        datacenter_config = self.config['datacenter']
        aux_load_config = self.config.get('auxiliary_load', {})

        bess_energy_mwh = bess_energy_mwh or bess_config['energy_capacity_mwh']
        bess_power_mw = bess_power_mw or bess_config['power_capacity_mw']
        wire_capacity_mw = wire_capacity_mw or wire_config['capacity_mw']

        # Use load_csv_path from config if not provided as parameter
        if load_csv_path is None:
            load_csv_path = datacenter_config.get('load_file')

        if verbose:
            print("=" * 60)
            print("SYSTEM SIMULATION")
            if first_year_only:
                print(" (FIRST YEAR ONLY)")
            if skip_bess:
                print(" (NO BESS - STATS ONLY)")
            print("=" * 60)
            if not skip_bess:
                print(f"BESS: {bess_energy_mwh:.1f} MWh / {bess_power_mw:.1f} MW")
            else:
                print("BESS: DISABLED")
            print(f"Wire: {wire_capacity_mw:.1f} MW")
            print()

        # Load generation data
        if verbose:
            print("Loading generation data...")
        generation_df = self.load_generation_data()

        # Generate or load load profile (match generation data length)
        n_years = sim_config['years']
        if assume_all_sellable:
            if verbose:
                print(f"Using maximum sellable load profile (constant at {wire_capacity_mw} MW)...")
            # Create flat load profile at wire capacity
            load_df = pd.DataFrame({
                'load_mw': np.full(len(generation_df), wire_capacity_mw)
            }, index=generation_df.index)
        elif load_csv_path:
            if verbose:
                print(f"Loading load profile from {load_csv_path}...")
            load_df = self.generate_load_profile(n_years, load_csv_path=load_csv_path)
        else:
            if verbose:
                print(f"Generating {n_years}-year load profile...")
            load_df = self.generate_load_profile(n_years)

        # Align time series (ensure same length)
        min_len = min(len(generation_df), len(load_df))
        generation_df = generation_df.iloc[:min_len]
        load_df = load_df.iloc[:min_len]

        # Generate or load auxiliary load profile
        if verbose:
            aux_mode = aux_load_config.get('mode', 'static')
            if aux_load_csv_path:
                aux_mode = 'csv'
            print(f"Generating auxiliary load profile (mode: {aux_mode})...")

        aux_load_profile_mw = self.generate_aux_load_profile(
            generation_df=generation_df,
            aux_load_csv_path=aux_load_csv_path
        )

        # Ensure aux load profile matches generation length
        if len(aux_load_profile_mw) != len(generation_df):
            aux_load_profile_mw = aux_load_profile_mw[:len(generation_df)]

        if verbose:
            aux_load_mean_mw = np.mean(aux_load_profile_mw)
            aux_load_max_mw = np.max(aux_load_profile_mw)
            aux_load_min_mw = np.min(aux_load_profile_mw)
            print(f"Auxiliary Load: {aux_load_mean_mw:.3f} MW avg ({aux_load_mean_mw * 1000:.1f} kW)")
            print(f"  Range: {aux_load_min_mw:.3f} - {aux_load_max_mw:.3f} MW")

        # Truncate to first year if requested
        if first_year_only:
            timestep_hours = sim_config['timestep_minutes'] / 60.0
            timesteps_per_year = int(365 * 24 / timestep_hours)
            generation_df = generation_df.iloc[:timesteps_per_year]
            load_df = load_df.iloc[:timesteps_per_year]
            aux_load_profile_mw = aux_load_profile_mw[:timesteps_per_year]
            if verbose:
                print(f"Limiting simulation to first year ({timesteps_per_year} timesteps)")

        timestep_hours = sim_config['timestep_minutes'] / 60.0
        timesteps_per_year = int(365 * 24 / timestep_hours)

        # Initialize and run BESS or create zero results
        if skip_bess:
            # Skip BESS simulation - create zero-valued results
            if verbose:
                print("Skipping BESS dispatch (stats only mode)...")

            n_timesteps = len(generation_df)
            generation_mw = generation_df['total_generation_mw'].values
            load_mw = load_df['load_mw'].values

            # Calculate wire flow and curtailment without BESS, accounting for aux load
            available_power = generation_mw - aux_load_profile_mw
            unserved_aux_load_mw = np.maximum(0, aux_load_profile_mw - generation_mw)
            wire_flow_mw = np.minimum(np.maximum(0, available_power), wire_capacity_mw)
            curtailment_mw = np.maximum(0, available_power - wire_capacity_mw)
            grid_import_mw = np.maximum(0, load_mw - wire_flow_mw)

            dispatch_results = pd.DataFrame({
                'charge_mw': np.zeros(n_timesteps),
                'discharge_mw': np.zeros(n_timesteps),
                'soc': np.zeros(n_timesteps),
                'curtailment_mw': curtailment_mw,
                'grid_import_mw': grid_import_mw,
                'wire_flow_mw': wire_flow_mw,
                'unserved_aux_load_mw': unserved_aux_load_mw
            })
            cumulative_degradation = 0.0
        else:
            # Initialize BESS dispatcher
            dispatcher = BESSDispatcher(
                energy_capacity_mwh=bess_energy_mwh,
                power_capacity_mw=bess_power_mw,
                min_soc=bess_config['min_soc'],
                max_soc=bess_config['max_soc'],
                initial_soc=bess_config['initial_soc'],
                charge_efficiency=bess_config['charge_efficiency'],
                discharge_efficiency=bess_config['discharge_efficiency'],
                max_daily_cycles=bess_config['max_daily_cycles'],
                degradation_per_cycle=bess_config['degradation_per_cycle']
            )

            # Run BESS optimization
            if verbose:
                print("Optimizing BESS dispatch...")

            # Pass timestamps to dispatcher to use actual years
            timestamps = load_df.index if hasattr(load_df.index, 'year') else None

            dispatch_results = dispatcher.simulate_multiyear(
                generation_mw=generation_df['total_generation_mw'].values,
                load_mw=load_df['load_mw'].values,
                wire_capacity_mw=wire_capacity_mw,
                timestep_hours=timestep_hours,
                timesteps_per_year=timesteps_per_year,
                verbose=verbose,
                timestamps=timestamps,
                aux_load_mw=aux_load_profile_mw
            )
            cumulative_degradation = dispatcher.cumulative_degradation

        # Combine all data
        results_df = pd.DataFrame({
            'timestamp': generation_df.index[:len(dispatch_results)].to_list(),
            'wind_mw': generation_df['wind_mw'].values[:len(dispatch_results)],
            'solar_mw': generation_df['solar_mw'].values[:len(dispatch_results)],
            'total_generation_mw': generation_df['total_generation_mw'].values[:len(dispatch_results)],
            'load_mw': load_df['load_mw'].values[:len(dispatch_results)],
            'bess_charge_mw': dispatch_results['charge_mw'].values,
            'bess_discharge_mw': dispatch_results['discharge_mw'].values,
            'bess_soc': dispatch_results['soc'].values,
            'curtailment_mw': dispatch_results['curtailment_mw'].values,
            'grid_import_mw': dispatch_results['grid_import_mw'].values,
            'wire_flow_mw': dispatch_results['wire_flow_mw'].values,
            'unserved_aux_load_mw': dispatch_results['unserved_aux_load_mw'].values
        })

        # Calculate metrics
        metrics = self._calculate_metrics(
            results_df,
            timestep_hours,
            bess_energy_mwh,
            bess_power_mw,
            wire_capacity_mw,
            cumulative_degradation,
            skip_bess
        )

        if verbose:
            self._print_metrics(metrics)

        self.results = {
            'timeseries': results_df,
            'metrics': metrics,
            'config': {
                'bess_energy_mwh': bess_energy_mwh,
                'bess_power_mw': bess_power_mw,
                'wire_capacity_mw': wire_capacity_mw
            }
        }

        return self.results

    def _calculate_metrics(
        self,
        results_df: pd.DataFrame,
        timestep_hours: float,
        bess_energy_mwh: float,
        bess_power_mw: float,
        wire_capacity_mw: float,
        degradation: float,
        skip_bess: bool = False
    ) -> Dict:
        """Calculate key performance metrics."""

        # Energy metrics (MWh)
        total_generation_mwh = results_df['total_generation_mw'].sum() * timestep_hours
        total_wind_mwh = results_df['wind_mw'].sum() * timestep_hours
        total_solar_mwh = results_df['solar_mw'].sum() * timestep_hours
        total_load_mwh = results_df['load_mw'].sum() * timestep_hours
        total_curtailment_mwh = results_df['curtailment_mw'].sum() * timestep_hours
        total_grid_import_mwh = results_df['grid_import_mw'].sum() * timestep_hours
        total_wire_delivery_mwh = results_df['wire_flow_mw'].sum() * timestep_hours
        total_bess_charge_mwh = results_df['bess_charge_mw'].sum() * timestep_hours
        total_bess_discharge_mwh = results_df['bess_discharge_mw'].sum() * timestep_hours
        total_unserved_aux_load_mwh = results_df['unserved_aux_load_mw'].sum() * timestep_hours

        # Utilization metrics
        curtailment_rate = total_curtailment_mwh / total_generation_mwh if total_generation_mwh > 0 else 0
        grid_import_rate = total_grid_import_mwh / total_load_mwh if total_load_mwh > 0 else 0
        renewable_fraction = (total_load_mwh - total_grid_import_mwh) / total_load_mwh if total_load_mwh > 0 else 0

        # Wire utilization
        avg_wire_flow_mw = results_df['wire_flow_mw'].mean()
        wire_utilization = avg_wire_flow_mw / wire_capacity_mw if wire_capacity_mw > 0 else 0

        # Load statistics (MW)
        load_min_mw = results_df['load_mw'].min()
        load_max_mw = results_df['load_mw'].max()
        load_avg_mw = results_df['load_mw'].mean()
        load_std_mw = results_df['load_mw'].std()

        # BESS metrics
        bess_cycles = total_bess_charge_mwh / bess_energy_mwh if bess_energy_mwh > 0 else 0
        bess_roundtrip_efficiency = total_bess_discharge_mwh / total_bess_charge_mwh if total_bess_charge_mwh > 0 else 0

        # Economic metrics
        econ_config = self.config['economics']
        ppa_price = econ_config['ppa_price_eur_per_mwh']
        wind_capex = econ_config.get('wind_capex_eur', 0.0)
        solar_capex = econ_config.get('solar_capex_eur', 0.0)
        bess_capex = econ_config['bess_capex_eur_per_mwh']
        wire_capex = econ_config['wire_capex_eur_per_mw']

        # Revenue from delivered renewable energy
        renewable_revenue_eur = total_wire_delivery_mwh * ppa_price

        # Capex
        total_bess_capex = 0 if skip_bess else bess_energy_mwh * bess_capex
        total_wire_capex = wire_capacity_mw * wire_capex
        total_capex = wind_capex + solar_capex + total_bess_capex + total_wire_capex

        # Simple payback (years)
        annual_revenue = renewable_revenue_eur / (len(results_df) * timestep_hours / 8760)
        simple_payback_years = total_capex / annual_revenue if annual_revenue > 0 else np.inf

        return {
            'energy': {
                'total_generation_mwh': total_generation_mwh,
                'total_wind_mwh': total_wind_mwh,
                'total_solar_mwh': total_solar_mwh,
                'total_load_mwh': total_load_mwh,
                'total_curtailment_mwh': total_curtailment_mwh,
                'total_grid_import_mwh': total_grid_import_mwh,
                'total_wire_delivery_mwh': total_wire_delivery_mwh,
                'total_unserved_aux_load_mwh': total_unserved_aux_load_mwh
            },
            'load_stats': {
                'min_mw': load_min_mw,
                'max_mw': load_max_mw,
                'avg_mw': load_avg_mw,
                'std_mw': load_std_mw
            },
            'utilization': {
                'curtailment_rate': curtailment_rate,
                'grid_import_rate': grid_import_rate,
                'renewable_fraction': renewable_fraction,
                'wire_utilization': wire_utilization
            },
            'bess': {
                'total_cycles': bess_cycles,
                'roundtrip_efficiency': bess_roundtrip_efficiency,
                'degradation': degradation
            },
            'economics': {
                'renewable_revenue_eur': renewable_revenue_eur,
                'wind_capex_eur': wind_capex,
                'solar_capex_eur': solar_capex,
                'bess_capex_eur': total_bess_capex,
                'wire_capex_eur': total_wire_capex,
                'total_capex_eur': total_capex,
                'simple_payback_years': simple_payback_years
            }
        }

    def _print_metrics(self, metrics: Dict):
        """Print formatted metrics."""
        print("\n" + "=" * 60)
        print("SIMULATION RESULTS")
        print("=" * 60)

        print("\nENERGY FLOWS (MWh):")
        print(f"  Total Generation:    {metrics['energy']['total_generation_mwh']:>12,.0f}")
        print(f"    Wind:              {metrics['energy']['total_wind_mwh']:>12,.0f}")
        print(f"    Solar:             {metrics['energy']['total_solar_mwh']:>12,.0f}")
        print(f"  Total Load:          {metrics['energy']['total_load_mwh']:>12,.0f}")
        print(f"  Wire Delivery:       {metrics['energy']['total_wire_delivery_mwh']:>12,.0f}")
        print(f"  Grid Import:         {metrics['energy']['total_grid_import_mwh']:>12,.0f}")
        print(f"  Curtailment:         {metrics['energy']['total_curtailment_mwh']:>12,.0f}")
        if metrics['energy']['total_unserved_aux_load_mwh'] > 0:
            print(f"  Unserved Aux Load:   {metrics['energy']['total_unserved_aux_load_mwh']:>12,.1f}")

        print("\nLOAD STATISTICS (MW):")
        print(f"  Minimum:             {metrics['load_stats']['min_mw']:>12.1f}")
        print(f"  Maximum:             {metrics['load_stats']['max_mw']:>12.1f}")
        print(f"  Average:             {metrics['load_stats']['avg_mw']:>12.1f}")
        print(f"  Std Deviation:       {metrics['load_stats']['std_mw']:>12.1f}")

        print("\nUTILIZATION:")
        print(f"  Renewable Fraction:  {metrics['utilization']['renewable_fraction']:>12.1%}")
        print(f"  Curtailment Rate:    {metrics['utilization']['curtailment_rate']:>12.1%}")
        print(f"  Grid Import Rate:    {metrics['utilization']['grid_import_rate']:>12.1%}")
        print(f"  Wire Utilization:    {metrics['utilization']['wire_utilization']:>12.1%}")

        print("\nBESS DISPATCH:")
        print(f"  Total Cycles:        {metrics['bess']['total_cycles']:>12.1f}")
        print(f"  Roundtrip Eff.:      {metrics['bess']['roundtrip_efficiency']:>12.1%}")
        print(f"  Degradation:         {metrics['bess']['degradation']:>12.1%}")

        print("\nECONOMICS (EUR):")
        print(f"  Wind Capex:          {metrics['economics']['wind_capex_eur']:>12,.0f}")
        print(f"  Solar Capex:         {metrics['economics']['solar_capex_eur']:>12,.0f}")
        print(f"  BESS Capex:          {metrics['economics']['bess_capex_eur']:>12,.0f}")
        print(f"  Wire Capex:          {metrics['economics']['wire_capex_eur']:>12,.0f}")
        print(f"  Total Capex:         {metrics['economics']['total_capex_eur']:>12,.0f}")
        print(f"  Revenue (PPA):       {metrics['economics']['renewable_revenue_eur']:>12,.0f}")
        print(f"  Simple Payback:      {metrics['economics']['simple_payback_years']:>12.1f} years")
        print("=" * 60 + "\n")

    def save_results(self, output_dir: str = "outputs", save_load_profile: bool = False):
        """
        Save simulation results to CSV.

        Parameters
        ----------
        output_dir : str
            Directory to save results in
        save_load_profile : bool
            If True, extract and save load profile to separate CSV file
        """
        if self.results is None:
            print("No results to save. Run simulation first.")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save timeseries
        ts_file = output_path / "timeseries_results.csv"
        self.results['timeseries'].to_csv(ts_file, index=False)
        print(f"Saved timeseries to {ts_file}")

        # Save load profile if requested
        if save_load_profile:
            load_profile = self.results['timeseries'][['timestamp', 'load_mw']].copy()
            load_profile.set_index('timestamp', inplace=True)
            load_file = output_path / "load_profile.csv"
            generator = DatacenterLoadGenerator(
                annual_peak_mw=0, capacity_factor=0, base_load_ratio=0
            )
            generator.save_to_csv(load_profile, str(load_file))
            print(f"Saved load profile to {load_file}")

        # Save metrics (convert numpy types to native Python types for readable YAML)
        metrics_file = output_path / "metrics.yaml"
        metrics_clean = convert_numpy_types(self.results['metrics'])
        with open(metrics_file, 'w') as f:
            yaml.dump(metrics_clean, f, default_flow_style=False, sort_keys=False)
        print(f"Saved metrics to {metrics_file}")


if __name__ == "__main__":
    # Example usage requires generation data files
    print("To run the simulator, you need to provide wind and solar generation data.")
    print("See config.yaml for expected file paths and formats.")
