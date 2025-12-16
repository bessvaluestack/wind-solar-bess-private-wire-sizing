"""
Auxiliary load generator module.

Generates realistic auxiliary (parasitic) load profiles for renewable energy systems
including inverter losses, monitoring equipment, and wind turbine control systems.
"""

import numpy as np
import pandas as pd
from typing import Optional
from pathlib import Path


class AuxiliaryLoadGenerator:
    """
    Generates realistic auxiliary load profiles based on generation patterns.

    Models three main components:
    1. Solar inverter parasitic loads (nighttime standby power)
    2. Fixed auxiliary equipment (monitoring, cameras, sensors, control systems)
    3. Wind turbine parasitic loads (nacelle positioning, pitch control, heating)
    """

    def __init__(
        self,
        num_wind_turbines: int = 21,
        wind_turbine_capacity_mw: float = 8.0,
        solar_capacity_mw: Optional[float] = None,
        inverter_standby_kw_per_mw: float = 0.5,
        inverter_operating_kw_per_mw: float = 0.2,
        fixed_aux_load_kw: float = 100.0,
        wind_turbine_base_aux_kw: float = 5.0,
        wind_turbine_aux_per_mw: float = 0.3,
        wind_low_generation_threshold_mw: float = 0.5,
        wind_nacelle_active_probability: float = 0.05,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the auxiliary load generator.

        Parameters
        ----------
        num_wind_turbines : int
            Number of wind turbines (default: 21)
        wind_turbine_capacity_mw : float
            Capacity of each wind turbine in MW (default: 8.0)
        solar_capacity_mw : float, optional
            Total solar capacity in MW. If None, inferred from generation data.
        inverter_standby_kw_per_mw : float
            Inverter standby power per MW of solar capacity when not generating (default: 0.5 kW/MW)
        inverter_operating_kw_per_mw : float
            Inverter parasitic load per MW of solar capacity when generating (default: 0.2 kW/MW)
        fixed_aux_load_kw : float
            Fixed auxiliary equipment load in kW (cameras, sensors, monitoring, control room, etc.)
            (default: 100 kW)
        wind_turbine_base_aux_kw : float
            Base auxiliary load per wind turbine for control systems, sensors, heating (default: 5 kW)
        wind_turbine_aux_per_mw : float
            Additional auxiliary load per MW of wind generation for active control (default: 0.3 kW/MW)
        wind_low_generation_threshold_mw : float
            Generation threshold per turbine below which nacelle/pitch control is more active (default: 0.5 MW)
        wind_nacelle_active_probability : float
            Probability per timestep that nacelle positioning/pitch control is active when generation is low
            (default: 0.05 = 5% of timesteps)
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.num_wind_turbines = num_wind_turbines
        self.wind_turbine_capacity_mw = wind_turbine_capacity_mw
        self.total_wind_capacity_mw = num_wind_turbines * wind_turbine_capacity_mw
        self.solar_capacity_mw = solar_capacity_mw

        # Inverter parameters
        self.inverter_standby_kw_per_mw = inverter_standby_kw_per_mw
        self.inverter_operating_kw_per_mw = inverter_operating_kw_per_mw

        # Fixed loads
        self.fixed_aux_load_kw = fixed_aux_load_kw

        # Wind turbine parameters
        self.wind_turbine_base_aux_kw = wind_turbine_base_aux_kw
        self.wind_turbine_aux_per_mw = wind_turbine_aux_per_mw
        self.wind_low_generation_threshold_mw = wind_low_generation_threshold_mw
        self.wind_nacelle_active_probability = wind_nacelle_active_probability

        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

    def generate_profile(
        self,
        wind_generation_mw: np.ndarray,
        solar_generation_mw: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> pd.DataFrame:
        """
        Generate auxiliary load profile based on wind and solar generation.

        Parameters
        ----------
        wind_generation_mw : np.ndarray
            Wind generation timeseries in MW
        solar_generation_mw : np.ndarray
            Solar generation timeseries in MW
        timestamps : pd.DatetimeIndex, optional
            Timestamps for the profile. If None, creates sequential index.

        Returns
        -------
        pd.DataFrame
            DataFrame with auxiliary load components and total
        """
        n_timesteps = len(wind_generation_mw)

        if len(solar_generation_mw) != n_timesteps:
            raise ValueError("Wind and solar generation arrays must have the same length")

        # Infer solar capacity if not provided
        if self.solar_capacity_mw is None:
            self.solar_capacity_mw = np.max(solar_generation_mw)
            if self.solar_capacity_mw == 0:
                self.solar_capacity_mw = 1.0  # Avoid division by zero

        # 1. Solar inverter parasitic loads
        inverter_load_kw = self._calculate_inverter_load(solar_generation_mw)

        # 2. Fixed auxiliary equipment loads
        fixed_load_kw = np.full(n_timesteps, self.fixed_aux_load_kw)

        # 3. Wind turbine parasitic loads
        wind_aux_load_kw = self._calculate_wind_turbine_load(wind_generation_mw)

        # Total auxiliary load
        total_aux_load_kw = inverter_load_kw + fixed_load_kw + wind_aux_load_kw
        total_aux_load_mw = total_aux_load_kw / 1000.0

        # Create DataFrame
        if timestamps is None:
            timestamps = pd.date_range(start='2024-01-01', periods=n_timesteps, freq='15min')

        df = pd.DataFrame({
            'timestamp': timestamps,
            'inverter_aux_kw': inverter_load_kw,
            'fixed_aux_kw': fixed_load_kw,
            'wind_turbine_aux_kw': wind_aux_load_kw,
            'total_aux_kw': total_aux_load_kw,
            'total_aux_mw': total_aux_load_mw
        })
        df.set_index('timestamp', inplace=True)

        return df

    def _calculate_inverter_load(self, solar_generation_mw: np.ndarray) -> np.ndarray:
        """
        Calculate solar inverter parasitic loads.

        Inverters consume power for:
        - Standby mode during nighttime (higher consumption)
        - Active mode parasitic losses during generation (lower consumption)

        Parameters
        ----------
        solar_generation_mw : np.ndarray
            Solar generation in MW

        Returns
        -------
        np.ndarray
            Inverter parasitic load in kW
        """
        n_timesteps = len(solar_generation_mw)
        inverter_load_kw = np.zeros(n_timesteps)

        # Standby load when not generating (nighttime)
        standby_mask = solar_generation_mw < 0.01  # Essentially zero generation
        inverter_load_kw[standby_mask] = self.solar_capacity_mw * self.inverter_standby_kw_per_mw

        # Operating parasitic load when generating
        operating_mask = ~standby_mask
        inverter_load_kw[operating_mask] = self.solar_capacity_mw * self.inverter_operating_kw_per_mw

        return inverter_load_kw

    def _calculate_wind_turbine_load(self, wind_generation_mw: np.ndarray) -> np.ndarray:
        """
        Calculate wind turbine parasitic loads.

        Wind turbines consume power for:
        - Base loads: control systems, sensors, heating, lighting (always on)
        - Active control: yaw/nacelle positioning, pitch control (more frequent at low wind)
        - Operating parasitic: small loads proportional to generation

        Parameters
        ----------
        wind_generation_mw : np.ndarray
            Total wind generation in MW

        Returns
        -------
        np.ndarray
            Wind turbine parasitic load in kW
        """
        n_timesteps = len(wind_generation_mw)

        # Base load for all turbines (control systems, sensors, heating)
        base_load_kw = self.num_wind_turbines * self.wind_turbine_base_aux_kw

        # Operating parasitic load (proportional to generation)
        operating_load_kw = wind_generation_mw * self.wind_turbine_aux_per_mw

        # Active control load (nacelle positioning, pitch control)
        # More active when generation is low (turbines adjusting to find wind)
        avg_generation_per_turbine = wind_generation_mw / self.num_wind_turbines
        low_wind_mask = avg_generation_per_turbine < self.wind_low_generation_threshold_mw

        # Probabilistic nacelle/pitch control activity
        nacelle_active = np.zeros(n_timesteps, dtype=bool)
        low_wind_indices = np.where(low_wind_mask)[0]

        if len(low_wind_indices) > 0:
            # Determine which timesteps have active control
            n_active = int(len(low_wind_indices) * self.wind_nacelle_active_probability)
            if n_active > 0:
                active_indices = np.random.choice(low_wind_indices, size=n_active, replace=False)
                nacelle_active[active_indices] = True

        # Nacelle/pitch control load (per turbine when active)
        # Assume 10-30 kW per turbine during active control
        control_load_per_turbine = np.random.uniform(10, 30, size=n_timesteps)
        control_load_kw = np.where(
            nacelle_active,
            control_load_per_turbine * self.num_wind_turbines,
            0
        )

        # Total wind turbine aux load
        total_wind_aux_kw = base_load_kw + operating_load_kw + control_load_kw

        return total_wind_aux_kw

    def save_to_csv(self, aux_load_df: pd.DataFrame, filepath: str) -> None:
        """
        Save auxiliary load profile to CSV file.

        Parameters
        ----------
        aux_load_df : pd.DataFrame
            Auxiliary load profile DataFrame
        filepath : str
            Path to save the CSV file
        """
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Reset index to include timestamp as a column
        df_to_save = aux_load_df.reset_index()
        df_to_save.to_csv(filepath, index=False)
        print(f"Saved auxiliary load profile to {filepath}")

    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load auxiliary load profile from CSV file.

        Expected columns:
        - timestamp or Interval_Date_Time
        - total_aux_mw (or will be calculated from total_aux_kw)

        Parameters
        ----------
        filepath : str
            Path to the CSV file containing aux load data

        Returns
        -------
        pd.DataFrame
            Auxiliary load profile DataFrame
        """
        df = pd.read_csv(filepath)

        # Handle both 'Interval_Date_Time' and 'timestamp' column names
        timestamp_col = 'Interval_Date_Time' if 'Interval_Date_Time' in df.columns else 'timestamp'
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df.rename(columns={timestamp_col: 'timestamp'}, inplace=True)
        df.set_index('timestamp', inplace=True)

        # Ensure total_aux_mw column exists
        if 'total_aux_mw' not in df.columns and 'total_aux_kw' in df.columns:
            df['total_aux_mw'] = df['total_aux_kw'] / 1000.0

        return df


if __name__ == "__main__":
    # Example usage
    print("Auxiliary Load Generator Example")
    print("=" * 60)

    # Generate sample wind and solar data
    n_timesteps = 35040  # One year at 15-min resolution

    # Simple sinusoidal pattern for demonstration
    hours = np.arange(n_timesteps) * 0.25

    # Solar: peaks during day
    solar_generation = 100 * np.maximum(0, np.sin(2 * np.pi * hours / 24))

    # Wind: more variable, 24/7
    wind_generation = 168 * (0.3 + 0.7 * np.abs(np.sin(2 * np.pi * hours / 24 + np.pi/4)))
    wind_generation += np.random.normal(0, 20, n_timesteps)
    wind_generation = np.maximum(0, wind_generation)

    # Create generator
    generator = AuxiliaryLoadGenerator(
        num_wind_turbines=21,
        wind_turbine_capacity_mw=8.0,
        solar_capacity_mw=100.0,
        random_seed=42
    )

    # Generate profile
    aux_profile = generator.generate_profile(
        wind_generation_mw=wind_generation,
        solar_generation_mw=solar_generation
    )

    # Print statistics
    print(f"\nAuxiliary Load Profile Statistics:")
    print(f"  Total Aux Load (kW):")
    print(f"    Mean:    {aux_profile['total_aux_kw'].mean():>10.2f}")
    print(f"    Min:     {aux_profile['total_aux_kw'].min():>10.2f}")
    print(f"    Max:     {aux_profile['total_aux_kw'].max():>10.2f}")
    print(f"    Std Dev: {aux_profile['total_aux_kw'].std():>10.2f}")

    print(f"\n  Component Breakdown (Mean kW):")
    print(f"    Inverter Aux:      {aux_profile['inverter_aux_kw'].mean():>10.2f}")
    print(f"    Fixed Aux:         {aux_profile['fixed_aux_kw'].mean():>10.2f}")
    print(f"    Wind Turbine Aux:  {aux_profile['wind_turbine_aux_kw'].mean():>10.2f}")

    print(f"\n  Total Aux Load (MW):")
    print(f"    Mean:    {aux_profile['total_aux_mw'].mean():>10.3f}")
    print(f"    Min:     {aux_profile['total_aux_mw'].min():>10.3f}")
    print(f"    Max:     {aux_profile['total_aux_mw'].max():>10.3f}")

    print(f"\n  First few timesteps:")
    print(aux_profile.head(10))
