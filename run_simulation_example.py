#!/usr/bin/env python3
"""
Example script showing how to run simulations with first_year_only option.
"""

from src.simulator import SystemSimulator

# Initialize simulator with config file
simulator = SystemSimulator("config.yaml")

print("\n" + "="*70)
print("EXAMPLE 1: Running full 10-year simulation")
print("="*70)

# Run full simulation (all years from config)
results_full = simulator.run_simulation(
    load_csv_path="data/63_MW_datacenter_load.csv",
    first_year_only=False  # Use all years
)

print("\n" + "="*70)
print("EXAMPLE 2: Running first year only (faster for testing)")
print("="*70)

# Run first year only - useful for quick testing and optimization
results_first_year = simulator.run_simulation(
    load_csv_path="data/63_MW_datacenter_load.csv",
    first_year_only=True  # Only simulate first year
)

# Save results
print("\nSaving results...")
simulator.save_results(output_dir="outputs")

print("\nDone! Check the 'outputs' folder for results.")
