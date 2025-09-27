# European Spread Option Pricing and SABR/Heston Simulation (main.py)

This project centers on pricing a European spread option via Monte Carlo and comparing it against Black–Scholes and Bachelier approximations. It also includes a mini playground for stochastic volatility models (Heston and SABR): path simulations, distribution comparison, pricing via SABR Monte Carlo vs Black–Scholes, implied-volatility smile with Hagan (2002) SABR, parameter sensitivities, and a simple SABR calibration to synthetic data.


## What’s in here
- `main.py` — the main script with all logic and plots.
- A collection of PNG files — example outputs/illustrations.
- An assignment PDF (context only).


## Dependencies
- Python 3.9+ recommended
- numpy
- scipy
- matplotlib
- pysabr (Hagan 2002 lognormal SABR implementation)

You can install them with the provided requirements file (see Setup below).


## Quick start (Windows, cmd.exe)
1) Create and activate a virtual environment

```bat
python -m venv .venv
.venv\Scripts\activate
```

2) Install dependencies

```bat
pip install -r requirements.txt
```

3) Run the script

```bat
python main.py
```

The script opens multiple plots and prints key metrics to the console. Close figures to let the script continue to the next sections.


## What the script does (high level)
The main function is organized in parts. Each part prints metrics and/or opens plots.

- Part A — Monte Carlo price of a European spread call option on two GBM futures S1 and S2 with correlation ρ.
  - Simulates S1 and S2 paths using correlated normals (via eigen-decomposition of the correlation matrix).
  - Prices the spread call at-the-money (K = S1 − S2) from simulated payoffs.
  - Plots a histogram of payoffs.
  - If ρ = 0.3, it also plots sample S1 and S2 paths.

- Part B — Sensitivity to correlation.
  - Sweeps ρ in [−1, 1] and plots the option value vs correlation.

- Part C — Closed-form proxy for spread volatility.
  - Computes σ_p^2 = σ1^2 + σ2^2 − 2 σ1 σ2 ρ and prints σ_p.

- Part D — Black–Scholes approximation for the spread.
  - Uses σ_p to price a call using a Black–Scholes-style formula on S = S1 − S2; compares with Monte Carlo in a bar chart.

- Part E — Bachelier (normal) model approximation for the spread.
  - Prices a call with normal model; compares MC vs Black–Scholes vs Bachelier in a bar chart.

- Part F — Discrepancy vs strike.
  - For strikes around S, plots absolute pricing discrepancies |BS − MC| and |Bachelier − MC| vs strike.

- Exercise 10 — Stochastic volatility models and smiles
  - Heston single path: simulates price and variance paths; plots both.
  - SABR single path: simulates forward and volatility; plots both.
  - Compares log-return histograms from Heston and SABR to a matched normal.
  - SABR Monte Carlo pricing: simulates many SABR paths, prices European calls/puts at 25% ITM, 10% ITM, ATM, 10% OTM, 25% OTM; compares to Black–Scholes with a constant vol.
  - Implied-volatility smile: uses `pysabr.Hagan2002LognormalSABR` to generate a smile vs strike and compares to flat BS vol.
  - Parameter sensitivities: plots smiles while varying α (vol-of-vol), β (elasticity), and ρ (correlation).
  - Simple SABR fitting: fits α, ρ, and vol0 to synthetic “market” vols via Nelder–Mead and plots fitted vs market.


## Console outputs you’ll see
- Estimated MC value of the European spread option
- Volatility of the spread σ_p
- Black–Scholes price using σ_p
- Bachelier price


## Plots you’ll see
- (Optional when ρ = 0.3) Simulated paths for S1 and S2
- Histogram of spread payoffs
- Spread option value vs correlation ρ
- Bar charts comparing MC, Black–Scholes, and Bachelier prices
- Discrepancy vs strike (|BS − MC| and |Bachelier − MC|)
- Heston: asset price and variance paths
- SABR: forward and volatility paths
- Histograms of log-returns (Heston and SABR) vs normal
- Option prices vs moneyness: SABR MC vs Black–Scholes (calls and puts)
- Implied volatility smile: SABR vs flat BS
- Sensitivity smiles (varying α, β, ρ)
- SABR fit: market vs fitted implied vols


## Configuration knobs (edit in `main.py`)
Spread option :
- S1, S2 — current futures levels
- sigma1, sigma2 — volatilities of S1 and S2
- rho — correlation between the Brownian motions
- T — time to maturity (years)
- r — risk-free rate (0 for futures under Q)
- n_simulations — number of Monte Carlo paths (computationally expensive)
- n_steps — time steps per path

Stochastic volatility (Exercise 10):
- Heston: S0, r, nu0, theta, rho, xi, kappa, T, dt
- SABR: F0, sigma0, beta, rho, alpha, T, dt
- `pysabr.Hagan2002LognormalSABR` parameters: f, shift, t, v_atm_n, beta, rho, volvol

Notes:
- Random seeds aren’t fixed; results vary run-to-run.
- Many figures open; close them to progress through the script.


## Performance tips
- Decrease `n_simulations` or `n_steps` for faster iteration.
- Prefer a modern 64-bit Python.
- If you only want a subset of figures, comment out blocks in `main()`.


## Troubleshooting
- ImportError for `pysabr`: ensure it’s installed (`pip install pysabr`).
- Matplotlib backend issues: if figures don’t show, run in a local desktop environment. To save figures instead of showing them, you can replace `plt.show()` with `plt.savefig("name.png", dpi=200)`.
- Long runtime: lower `n_simulations` for the Monte Carlo parts.


## References (for theory)
- Hagan, P., Kumar, D., Lesniewski, A., and Woodward, D. (2002). Managing Smile Risk.
- Heston, S. (1993). A Closed-Form Solution for Options with Stochastic Volatility.
- Bachelier, L. (1900). Théorie de la Spéculation.



