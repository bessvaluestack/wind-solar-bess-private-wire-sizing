# BESS + Private Wire Simulation Notebook - Summary

## What Was Created

This document summarizes the Google Colab notebook and supporting documentation created for a generic renewable energy park BESS simulation project.

---

## Files Created

### 1. BESS_and_PW_Simulation.ipynb
**Purpose**: Interactive Jupyter notebook for Google Colab
**Size**: ~1300 lines of code and markdown
**Target Audience**: End clients and stakeholders

**Sections**:
- **Section 0: Introduction** - Overview of the project, PPA structure, simulation objectives
- **Section 1: Data Connectivity** - Google Drive integration, data loading, statistics (3 visualizations)
- **Section 2: CAPEX Parameters** - Economic configuration with override capabilities
- **Section 3: No BESS Simulation** - Baseline scenario analysis (3 visualizations)
- **Section 4: BESS Simulation** - Optimized battery dispatch and comparison (3 visualizations)
- **Section 5: Conclusion & Export** - Summary and results export

**Total Visualizations**: 13 charts covering:
- Generation patterns (time series, distributions, monthly averages)
- Power flows (with and without BESS)
- Energy balance (Sankey-style diagrams)
- Duration curves
- Scenario comparisons
- BESS performance analytics

---

### 2. COLAB_QUICKSTART.md
**Purpose**: Quick 5-minute setup guide
**Length**: ~350 lines
**Audience**: Users who want to get started quickly

**Contents**:
- 5-minute setup procedure
- Data format requirements (concise)
- Common troubleshooting (quick fixes)
- First-time user tips (fast mode)
- What to expect from the notebook

---

### 3. COLAB_SETUP_GUIDE.md
**Purpose**: Comprehensive setup and troubleshooting guide
**Length**: ~650 lines
**Audience**: Users who need detailed instructions or encounter issues

**Contents**:
- Step-by-step setup instructions (8 detailed steps)
- Google Drive folder structure
- GitHub repository configuration
- Private repository authentication
- Data format specifications (detailed)
- 7 common issues with solutions
- Advanced configuration options
- Best practices
- Testing procedures

---

### 4. README.md Updates
**Changes**: Added Google Colab section
**New Content**:
- Link to notebook and guides
- Description of notebook capabilities
- Quick start instructions for Colab

---

### 5. Project Structure Updates
**Files Removed**:
- `generate_example_data.py` (test data generator - no longer needed)

**Files Added**:
- Notebook and documentation files listed above

---

## Notebook Features

### Data Loading
- **Google Drive Integration**: Direct loading from Google Drive folders
- **Automatic Conversion**: Handles kWh to MW conversions for 15-min intervals
- **Data Validation**: Statistics and quality checks for loaded data
- **Flexible Sources**: Supports both CSV files and auto-generated profiles

### Simulation Modes

#### No BESS Simulation
- Direct connection without battery storage
- Baseline curtailment and grid import analysis
- Wire capacity utilization
- Economic baseline

#### BESS Simulation
- Convex optimization-based battery dispatch
- Minimizes curtailment and grid import
- Respects all physical and operational constraints
- Tracks degradation over simulation period

### Configuration Options

**Speed Options**:
- First-year-only mode (5-10x faster for testing)
- Multi-year mode (full project lifetime)

**Load Profile Options**:
- Variable load profile (realistic datacenter consumption)
- Assume all sellable (merchant scenario)
- Custom CSV load data

**Economic Overrides**:
- CAPEX parameters (wind, solar, BESS, wire)
- PPA price
- Project lifetime and discount rate

**BESS Sizing Overrides**:
- Energy capacity (MWh)
- Power capacity (MW)
- Wire capacity (MW)

### Visualizations

**Section 1: Data Analysis** (3 charts)
1. Generation time series - sample week with wind, solar, total, wire capacity
2. Distribution histograms - wind and solar with correlation scatter plot
3. Monthly patterns - average generation by month

**Section 3: No BESS** (3 charts)
1. Power flows - generation, curtailment, wire flow, load coverage (sample week)
2. Energy balance - horizontal bar chart showing all energy flows
3. Duration curves - sorted power values for generation and load

**Section 4: BESS** (3 charts)
1. Power flows with BESS - generation, BESS operation, SoC, load coverage (sample week)
2. Scenario comparison - side-by-side bars comparing No BESS vs BESS
3. BESS performance - SoC distribution, power distribution, daily cycling, cumulative throughput

### Metrics Calculated

**Energy Metrics**:
- Total generation, curtailment, grid import
- Wire delivery
- Wind and solar breakdowns

**Utilization Metrics**:
- Curtailment rate (%)
- Grid import rate (%)
- Renewable fraction (%)
- Wire utilization (%)

**BESS Metrics**:
- Total cycles over project lifetime
- Round-trip efficiency
- Capacity degradation (%)
- Daily cycling statistics

**Economic Metrics**:
- CAPEX breakdown (wind, solar, BESS, wire)
- PPA revenue
- Simple payback period
- Comparison of scenarios

### Export Capabilities

**Results Export**:
- Timeseries CSV (full 15-min data)
- Metrics JSON (summary statistics)
- Saved to Google Drive for further analysis
- Downloadable to local machine

---

## How the Notebook Works

### Setup Phase
1. **Clone Repository**: Downloads code from GitHub
2. **Install Dependencies**: Installs required Python packages (numpy, pandas, cvxpy, matplotlib, etc.)
3. **Mount Google Drive**: Connects to user's Google Drive for data access
4. **Import Modules**: Loads simulation modules from the cloned repository

### Data Phase
1. **Load Configuration**: Reads default config from YAML
2. **Update Paths**: Points to Google Drive data location
3. **Load Generation Data**: Reads wind and solar CSV files
4. **Validate Data**: Displays statistics and checks for issues
5. **Generate/Load Aux Load**: Handles auxiliary loads (inverters, turbines)

### Simulation Phase

**No BESS**:
1. Generate or load datacenter load profile
2. Calculate wire flow (limited by capacity)
3. Calculate curtailment (excess generation)
4. Calculate grid import (deficit coverage)
5. Compute metrics and visualize

**With BESS**:
1. Initialize BESS dispatcher with constraints
2. Run convex optimization for each year
3. Track degradation across years
4. Calculate improved metrics
5. Compare with baseline
6. Visualize BESS operation

### Analysis Phase
1. Display comprehensive metrics
2. Generate visualizations
3. Compare scenarios
4. Export results

---

## Technical Implementation

### Data Format Handling

**Input CSVs**:
- Automatically detects column names (`Interval_Date_Time` or `timestamp`)
- Handles timezone-aware datetimes
- Converts kWh to MW for 15-minute intervals (kWh / 250 = MW)
- Aligns time series by truncating to shorter length

**Load Profiles**:
- Three modes: CSV, auto-generate, or assume all sellable
- Validates resolution (15-minute intervals)
- Checks for data consistency

### Optimization

**CVXPY Implementation**:
- Convex quadratic programming
- ECOS solver (guaranteed global optimum)
- ~35,000 timesteps for 10 years
- Solve time: 1-10 seconds per year

**Objective Function**:
- Minimize: curtailment + grid import
- Subject to: SoC limits, power limits, wire capacity, daily cycling limits

**Multi-Year Handling**:
- Year-by-year optimization
- Degradation tracking between years
- Reduced capacity in later years

### Visualization

**Matplotlib Styling**:
- Seaborn darkgrid theme
- Consistent color scheme (wind=blue, solar=orange, BESS=purple, etc.)
- Large readable fonts
- Legends and annotations

**Chart Types**:
- Time series line plots
- Filled area plots (stacked flows)
- Histograms (distributions)
- Bar charts (comparisons)
- Scatter plots (correlations)
- Dual-axis plots (power and SoC)

---

## Configuration Options

### Variables in Notebook

**Section 1 (Data)**:
```python
GDRIVE_DATA_PATH = '/content/drive/MyDrive/Park/data'
```

**Section 2 (CAPEX)**:
```python
WIND_CAPEX_EUR = None
SOLAR_CAPEX_EUR = None
BESS_CAPEX_EUR_PER_MWH = None
WIRE_CAPEX_EUR_PER_MW = None
PPA_PRICE_EUR_PER_MWH = None
PROJECT_LIFETIME_YEARS = None
DISCOUNT_RATE = None
```

**Section 3 (No BESS)**:
```python
FIRST_YEAR_ONLY = False
ASSUME_ALL_SELLABLE = False
LOAD_CSV_PATH = None
```

**Section 4 (BESS)**:
```python
BESS_ENERGY_MWH = None
BESS_POWER_MW = None
WIRE_CAPACITY_MW = None
FIRST_YEAR_ONLY_BESS = False
ASSUME_ALL_SELLABLE_BESS = False
LOAD_CSV_PATH_BESS = None
```

All variables default to `None` or `False`, using values from `config.yaml` when not overridden.

---

## Key Assumptions in Notebook

### Project Details
- **Wind**: 21 turbines × 8 MW = 168 MW installed
- **Solar**: ~177 MW installed
- **Private Wire**: x MW capacity
- **PPA Price**: € x /MWh 
- **Net Metering**: All delivered energy purchased at PPA price
- **Aux Power**: Power needed to operate the park when it's not producing is net metered

### Simulation Assumptions
- **Resolution**: 15-minute intervals
- **Duration**: 10-11 years of data (configurable)
- **Perfect Forecast**: Optimization assumes known generation and load
- **No Uncertainty**: Deterministic simulation
- **Simplified Degradation**: Throughput-based only (MWh cycled)

### BESS Defaults
- **Energy**: x MWh
- **Power**: x MW
- **SoC Range**: 5% - 95%
- **Efficiency**: 95% charge, 96% discharge (91% round-trip)
- **Max Daily Cycles**: 1.5
- **Degradation**: 0.01% per full cycle

---

## Use Cases

1. **Project Overview**: Understand system configuration and assumptions
2. **Performance Analysis**: See how BESS improves renewable delivery
3. **Economic Assessment**: Evaluate CAPEX, revenue, and payback
4. **Sensitivity Analysis**: Test different BESS sizes and wire capacities
5. **What-If Scenarios**: Adjust PPA price, project lifetime, etc.
6. **Quick Testing**: First-year-only mode for rapid iteration
7. **Parameter Tuning**: Test configurations before full optimization
8. **Validation**: Verify simulation results with visualizations
9. **Reporting**: Generate charts and metrics for presentations

---

### Recommendation
- **Use Notebook for**: Client presentations, exploratory analysis, visualization, sharing results
- **Use Command-Line Interface for**: Production runs, optimization, automation, parameter sweeps

---

## Customization Guide

### Adding New Visualizations

Add a new cell after existing visualization cells:

```python
# Visualization X.X: Your Title
fig, ax = plt.subplots(figsize=(12, 6))
# Your plotting code here
plt.tight_layout()
plt.show()
```

### Adding New Metrics

Edit the metrics calculation in the simulation results:

```python
# Calculate custom metric
my_metric = ts_bess['some_column'].sum() / len(ts_bess)
print(f"My Metric: {my_metric:.2f}")
```

### Changing Default Parameters

Edit the config override cells (Section 2 and 4):

```python
# Change default BESS size
BESS_ENERGY_MWH = 250  # Instead of None
BESS_POWER_MW = 60     # Instead of None
```

---

## Limitations and Future Work

### Current Limitations
- **Single Load Scenario**: Can only run one load profile per execution
- **No Comparison Mode**: Cannot easily compare multiple BESS sizes side-by-side
- **Manual Optimization**: Sizing still requires manual trial-and-error
- **Static Config**: Config file changes require re-running setup cells

### Future Enhancements
- [ ] Separate optimization notebook (wire-only and full optimization)
- [ ] Multi-scenario comparison in single run
- [ ] Interactive widgets for parameter adjustment
- [ ] Animated time series visualizations
- [ ] Automated report generation (PDF export)
- [ ] Cost breakdown waterfall charts
- [ ] Sensitivity tornado diagrams

---

## File Locations After Setup

### In GitHub Repository
```
wind-solar-bess-private-wire-sizing/
├── BESS_and_PW_Simulation.ipynb
├── COLAB_QUICKSTART.md
├── COLAB_SETUP_GUIDE.md
├── config.yaml
├── requirements.txt
└── src/
    └── (simulation modules)
```

### In Google Drive
```
My Drive/
└── Park/
    ├── data/
    │   ├── wind_generation_11_years.csv
    │   ├── solar_generation_11_years.csv
    │   └── (optional load CSVs)
    └── outputs/  (created by notebook)
        ├── timeseries_no_bess.csv
        ├── metrics_no_bess.json
        ├── timeseries_bess.csv
        └── metrics_bess.json
```

### In Colab Runtime (Temporary)
```
/content/
├── drive/MyDrive/Park/...  (mounted)
└── wind-solar-bess-private-wire-sizing/  (cloned)
    ├── config_colab.yaml  (modified config)
    └── src/...
```

---

## Support and Documentation

**Quick Help**: [COLAB_QUICKSTART.md](COLAB_QUICKSTART.md)
**Detailed Guide**: [COLAB_SETUP_GUIDE.md](COLAB_SETUP_GUIDE.md)
**Project Instructions**: [CLAUDE.md](CLAUDE.md)
**Repository README**: [README.md](README.md)

**Common Resources**:
- Google Colab FAQ: https://colab.research.google.com/faq
- CVXPY Documentation: https://www.cvxpy.org/
- Pandas Documentation: https://pandas.pydata.org/docs/