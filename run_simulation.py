#!/usr/bin/env python3
"""
Simple script to run a single simulation with current configuration.
"""

import argparse
from src.simulator import SystemSimulator


def main():
    parser = argparse.ArgumentParser(
        description="Run wind + solar + BESS + private wire simulation"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--bess-energy',
        type=float,
        help='Override BESS energy capacity (MWh)'
    )
    parser.add_argument(
        '--bess-power',
        type=float,
        help='Override BESS power capacity (MW)'
    )
    parser.add_argument(
        '--wire-capacity',
        type=float,
        help='Override wire capacity (MW)'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save results to outputs directory'
    )
    parser.add_argument(
        '--load-csv',
        type=str,
        help='Path to CSV file with datacenter load data (15-min resolution)'
    )
    parser.add_argument(
        '--aux-load-csv',
        type=str,
        help='Path to CSV file with auxiliary load data (overrides config mode)'
    )
    parser.add_argument(
        '--save-load-profile',
        action='store_true',
        help='Save generated load profile to CSV file'
    )
    parser.add_argument(
        '--first-year-only',
        action='store_true',
        help='Run simulation for first year only (faster for testing)'
    )
    parser.add_argument(
        '--assume-all-sellable',
        action='store_true',
        help='Assume all energy can be sold (creates flat load at wire capacity)'
    )
    parser.add_argument(
        '--no-bess',
        action='store_true',
        help='Skip BESS simulation and run stats calculator only (no battery storage)'
    )

    args = parser.parse_args()

    # Initialize and run simulator
    simulator = SystemSimulator(args.config)

    results = simulator.run_simulation(
        bess_energy_mwh=args.bess_energy,
        bess_power_mw=args.bess_power,
        wire_capacity_mw=args.wire_capacity,
        verbose=True,
        load_csv_path=args.load_csv,
        aux_load_csv_path=args.aux_load_csv,
        first_year_only=args.first_year_only,
        assume_all_sellable=args.assume_all_sellable,
        skip_bess=args.no_bess
    )

    if args.save:
        simulator.save_results(save_load_profile=args.save_load_profile)


if __name__ == "__main__":
    main()
