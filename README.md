# CosmixAna

A Dash-based cosmic detector analyzer for collecting, visualizing, and fitting cosmic-ray coincidence measurements.

## What it does

- Runs simulated detector measurements with operator name, duration, angle, count rates, and coincidences.
- Persists measurements to `detector_measurements.csv`.
- Provides multiple Dash pages:
  - Home page with measurement controls and recent entries
  - Poisson page with histogram and Poisson fit overlay
  - Trend page for count and rate evolution over time
  - Angle page with angular dependence plotting and fit
- Fits the angle dependence using a model of the form `f(θ) = A·cos(θ)^n + B` and displays fitted values with uncertainties.

## Install

1. Clone or open this repository.
2. Create a Python virtual environment:

```bash
cd CosmixAna
python3 -m venv .venv
```

3. Activate the virtual environment:

```bash
source .venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

If there is no `requirements.txt`, install the main dependencies manually:

```bash
pip install dash pandas plotly numpy scipy
```

## Run the app

With the virtual environment active:

```bash
python detector_app.py
```

Then open the local Dash URL shown in the terminal.

## Usage

- Use the home page to enter operator name, duration, and angle, then click `Run measurement`.
- Navigate to the Poisson page to inspect histograms and enable the Poisson fit.
- Navigate to the Trend page for temporal evolution of counts and rates.
- Navigate to the Angle page and enable the fit to see the angular model and fitted parameters.

## Notes

- Data is stored in `detector_measurements.csv` in the project root.
- The angle fit model is explicitly shown as `f(θ) = A·cos(θ)^n + B` in the UI when the fit is activated.
- The app supports English and French via the language selector.
