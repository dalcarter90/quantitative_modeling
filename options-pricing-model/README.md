# Options Pricing Model using Monte Carlo Simulation

This project implements an options pricing model that utilizes Monte Carlo simulation data to calculate option prices. The model is designed to integrate seamlessly with existing Monte Carlo simulation pipelines, providing a robust framework for financial analysis.

## Project Structure

```
options-pricing-model
├── src
│   ├── montecarlo_pricing.py        # Implementation of the options pricing model
│   ├── pipeline_integration.py       # Integration with the Monte Carlo simulation pipeline
│   └── types
│       └── index.py                  # Custom types and data structures
├── notebooks
│   └── options_pricing_analysis.ipynb # Jupyter notebook for analysis and visualization
├── data
│   └── sample_simulation_data.csv     # Sample data from Monte Carlo simulations
├── results
│   └── pricing_results.csv            # Output of the options pricing model
└── README.md                          # Documentation for the project
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd options-pricing-model
   ```

2. **Install required packages**:
   Ensure you have Python installed, then install the necessary packages using pip:
   ```
   pip install -r requirements.txt
   ```

3. **Data Preparation**:
   Place your Monte Carlo simulation data in the `data` directory as `sample_simulation_data.csv`.

## Usage

- **Monte Carlo Pricing Model**:
  The main functionality for pricing options is implemented in `src/montecarlo_pricing.py`. You can import this module and use the provided functions to calculate option prices based on the simulated price paths.

- **Pipeline Integration**:
  Use `src/pipeline_integration.py` to integrate the options pricing model with your existing Monte Carlo simulation pipeline. This script handles data retrieval and prepares it for pricing calculations.

- **Analysis and Visualization**:
  The Jupyter notebook `notebooks/options_pricing_analysis.ipynb` provides an interactive environment for exploratory data analysis and visualization of the pricing results. Load the simulation data and generate plots to analyze the results.

## Results

The output of the options pricing model will be stored in `results/pricing_results.csv`, which includes calculated option prices and associated metrics.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.