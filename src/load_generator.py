"""
Datacenter load generator module.

Generates 15-minute resolution load profiles for hyperscale datacenters
with configurable peak loads and capacity factors.
"""

import numpy as np
import pandas as pd
from typing import Optional
from pathlib import Path


class DatacenterLoadGenerator:
    """
    Generates realistic datacenter load profiles.

    Datacenters typically have high capacity factors with relatively stable
    base loads and some variability for dynamic workloads.
    """

    def __init__(
        self,
        annual_peak_mw: float,
        capacity_factor: float,
        base_load_ratio: float = 0.7,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the datacenter load generator.

        Parameters
        ----------
        annual_peak_mw : float
            Peak load in MW
        capacity_factor : float
            Annual capacity factor (0-1)
        base_load_ratio : float, optional
            Ratio of base load to peak load (default: 0.7)
            Higher values mean more stable loads
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.annual_peak_mw = annual_peak_mw
        self.capacity_factor = capacity_factor
        self.base_load_ratio = base_load_ratio
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

    def generate_annual_profile(
        self,
        year: int = 2024,
        timestep_minutes: int = 15
    ) -> pd.DataFrame:
        """
        Generate annual load profile with 15-minute resolution.

        Parameters
        ----------
        year : int
            Year for the profile (for datetime index)
        timestep_minutes : int
            Timestep in minutes (default: 15)

        Returns
        -------
        pd.DataFrame
            DataFrame with datetime index and 'load_mw' column
        """
        # Create datetime index
        start_date = pd.Timestamp(f'{year}-01-01 00:00:00')
        end_date = pd.Timestamp(f'{year}-12-31 23:45:00')
        date_range = pd.date_range(
            start=start_date,
            end=end_date,
            freq=f'{timestep_minutes}min'
        )

        n_timesteps = len(date_range)

        # Calculate base and variable load components
        base_load = self.annual_peak_mw * self.base_load_ratio
        variable_capacity = self.annual_peak_mw - base_load

        # Generate daily patterns (datacenters have weekly patterns)
        # Lower loads on weekends, slight daily variation
        day_of_week = date_range.dayofweek  # 0=Monday, 6=Sunday
        hour_of_day = date_range.hour + date_range.minute / 60.0

        # Weekend reduction factor (10% lower on weekends)
        weekend_factor = np.where(day_of_week >= 5, 0.9, 1.0)

        # Daily pattern (slight peak during business hours)
        # Using sinusoidal pattern with peak at 14:00
        daily_pattern = 0.95 + 0.05 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)

        # Combine patterns to get target variable load
        pattern_factor = weekend_factor * daily_pattern

        # Add stochastic component for realistic variation
        # Using AR(1) process for temporal correlation
        noise = self._generate_ar1_noise(n_timesteps, phi=0.95, sigma=0.02)

        # Calculate variable load
        variable_load = variable_capacity * pattern_factor * (1 + noise)

        # Total load
        total_load = base_load + variable_load

        # Ensure we don't exceed peak
        total_load = np.clip(total_load, 0, self.annual_peak_mw)

        # Scale to match target capacity factor
        current_cf = total_load.mean() / self.annual_peak_mw
        scaling_factor = self.capacity_factor / current_cf
        total_load = total_load * scaling_factor

        # Final clipping
        total_load = np.clip(total_load, 0, self.annual_peak_mw)

        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': date_range,
            'load_mw': total_load
        })
        df.set_index('timestamp', inplace=True)

        return df

    def _generate_ar1_noise(
        self,
        n: int,
        phi: float = 0.95,
        sigma: float = 0.02
    ) -> np.ndarray:
        """
        Generate AR(1) noise process for temporal correlation.

        Parameters
        ----------
        n : int
            Number of samples
        phi : float
            Autocorrelation parameter (0-1)
        sigma : float
            Standard deviation of innovations

        Returns
        -------
        np.ndarray
            AR(1) noise series
        """
        noise = np.zeros(n)
        noise[0] = np.random.normal(0, sigma)

        for i in range(1, n):
            noise[i] = phi * noise[i-1] + np.random.normal(0, sigma)

        return noise

    def generate_multiyear_profile(
        self,
        n_years: int,
        start_year: int = 2024,
        annual_growth_rate: float = 0.0,
        timestep_minutes: int = 15
    ) -> pd.DataFrame:
        """
        Generate multi-year load profile with optional growth.

        Parameters
        ----------
        n_years : int
            Number of years to simulate
        start_year : int
            Starting year
        annual_growth_rate : float
            Annual load growth rate (e.g., 0.02 for 2% growth)
        timestep_minutes : int
            Timestep in minutes

        Returns
        -------
        pd.DataFrame
            Multi-year DataFrame with 'load_mw' column
        """
        profiles = []

        for year_offset in range(n_years):
            year = start_year + year_offset

            # Adjust peak for growth
            adjusted_peak = self.annual_peak_mw * (1 + annual_growth_rate) ** year_offset

            # Temporarily adjust peak
            original_peak = self.annual_peak_mw
            self.annual_peak_mw = adjusted_peak

            # Generate profile for this year
            year_profile = self.generate_annual_profile(year, timestep_minutes)
            profiles.append(year_profile)

            # Restore original peak
            self.annual_peak_mw = original_peak

        # Concatenate all years
        combined_profile = pd.concat(profiles)

        return combined_profile

    def save_to_csv(self, load_df: pd.DataFrame, filepath: str) -> None:
        """
        Save load profile to CSV file.

        Parameters
        ----------
        load_df : pd.DataFrame
            Load profile DataFrame with datetime index and 'load_mw' column
        filepath : str
            Path to save the CSV file
        """
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Reset index to include timestamp as a column
        df_to_save = load_df.reset_index()
        df_to_save.to_csv(filepath, index=False)

    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load load profile from CSV file.

        Parameters
        ----------
        filepath : str
            Path to the CSV file containing load data

        Returns
        -------
        pd.DataFrame
            Load profile DataFrame with datetime index and 'load_mw' column
        """
        df = pd.read_csv(filepath)
        # Handle both 'Interval_Date_Time' and 'timestamp' column names
        timestamp_col = 'Interval_Date_Time' if 'Interval_Date_Time' in df.columns else 'timestamp'
        # Parse timezone-aware datetimes
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df.rename(columns={timestamp_col: 'timestamp'}, inplace=True)
        df.set_index('timestamp', inplace=True)

        # Check if data is in kWh instead of MW (if max value > 100, assume it's in kWh for 15-min intervals)
        if 'load_mw' in df.columns and df['load_mw'].max() > 100:
            print(f"Warning: Load data appears to be in kWh (max={df['load_mw'].max():.2f}). Converting to MW for 15-min intervals.")
            # kWh / 0.25 hours = kW, then kW / 1000 = MW
            # So: kWh / 250 = MW for 15-min intervals
            df['load_mw'] = df['load_mw'] / 250.0

        return df


if __name__ == "__main__":
    # Example usage
    generator = DatacenterLoadGenerator(
        annual_peak_mw=100.0,
        capacity_factor=0.85,
        base_load_ratio=0.7,
        random_seed=42
    )

    load_profile = generator.generate_annual_profile(year=2024)

    print(f"Load profile shape: {load_profile.shape}")
    print(f"Load statistics:")
    print(f"  Mean: {load_profile['load_mw'].mean():.2f} MW")
    print(f"  Max: {load_profile['load_mw'].max():.2f} MW")
    print(f"  Min: {load_profile['load_mw'].min():.2f} MW")
    print(f"  Capacity Factor: {load_profile['load_mw'].mean() / 100.0:.3f}")
    print(f"\nFirst few rows:")
    print(load_profile.head())
