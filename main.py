import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
import scipy.linalg as linalg
from scipy.stats import norm, skew, kurtosis
from scipy.optimize import brentq
from pysabr import Hagan2002LognormalSABR


def value_european_spread_option(S1, S2, sigma1, sigma2, rho, T, r, n_simulations, n_steps):
    """
    Value a European spread option using Monte Carlo simulation.

    Parameters:
    - S1, S2: Current prices of the underlying futures.
    - sigma1, sigma2: Volatilities of S1 and S2.
    - rho: Correlation coefficient between the Brownian motions of S1 and S2.
    - T: Time to maturity (in years).
    - r: Risk-free rate.
    - n_simulations: Number of Monte Carlo simulations.
    - n_steps: Number of steps in the simulation.

    Returns:
    - Estimated value of the European spread option.
    - Array of simulated payoffs at maturity.
    """
    K = S1 - S2  # ATM option
    cov_matrix = np.array([[1, rho], [rho, 1]])

    # Using eigenvalue decomposition to handle a wider range of correlation values
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    L = eigenvectors * np.sqrt(eigenvalues)

    z = np.random.normal(size=(n_steps, 2, n_simulations))
    correlated_randomness = L @ z

    dt = T / n_steps
    S1_paths = np.zeros((n_steps + 1, n_simulations))
    S2_paths = np.zeros((n_steps + 1, n_simulations))
    S1_paths[0] = S1
    S2_paths[0] = S2

    # Simulate paths for S1 and S2 using GBM
    for t in range(1, n_steps + 1):
        S1_paths[t] = S1_paths[t - 1] * np.exp(
            (r - 0.5 * sigma1 ** 2) * dt + sigma1 * np.sqrt(dt) * correlated_randomness[t - 1, 0])
        S2_paths[t] = S2_paths[t - 1] * np.exp(
            (r - 0.5 * sigma2 ** 2) * dt + sigma2 * np.sqrt(dt) * correlated_randomness[t - 1, 1])

    # Calculate payoffs and option value
    payoffs = np.maximum(S1_paths[-1] - S2_paths[-1] - K, 0)
    option_value = np.mean(payoffs)

    if rho == 0.3:
        # To plot the paths of S1
        plt.figure(figsize=(14, 7))
        for i in range(n_simulations):
            plt.plot(S1_paths[:, i], lw=1)  # Plot each path with a light weight
        plt.title('MC Simulation for S1')
        plt.xlabel('Time Steps')
        plt.ylabel('Option Price')
        plt.show()

        # Do the same for S2
        plt.figure(figsize=(14, 7))
        for i in range(n_simulations):
            plt.plot(S2_paths[:, i], lw=1)  # Plot each path with a light weight
        plt.title('MC Simulation for S2')
        plt.xlabel('Time Steps')
        plt.ylabel('Option Price')
        plt.show()

    return option_value, payoffs


def bs_value(S, K, T, r, sigma_spread):
    """
    Calculate Black-Scholes option value.

    Parameters:
    - S: Current price of the underlying
    - K: Strike price
    - T: Time to maturity
    - r: Risk-free rate
    - sigma_spread: Volatility of the spread

    Returns:
    - Call option price
    """
    d1 = (r + 0.5 * sigma_spread ** 2) * T / (sigma_spread * np.sqrt(T))
    d2 = d1 - sigma_spread * np.sqrt(T)
    call_price_BS = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price_BS


def calculate_log_returns(prices):
    """
    Calculate logarithmic returns from price series.

    Parameters:
    - prices: Array of price points

    Returns:
    - Array of log returns
    """
    return np.log(prices[1:] / prices[:-1])


def compare_dist(S_heston, F_sabr):
    """
    Function to compare the distribution of log-returns for Heston and SABR models.

    Parameters:
    - S_heston: Simulated price path from Heston model
    - F_sabr: Simulated price path from SABR model
    """
    # Calculating log-returns for Heston and SABR models
    log_returns_heston = calculate_log_returns(S_heston)
    log_returns_sabr = calculate_log_returns(F_sabr)

    # Calculate mean and variance of the log-returns for both models
    mean_heston, var_heston = np.mean(log_returns_heston), np.var(log_returns_heston)
    mean_sabr, var_sabr = np.mean(log_returns_sabr), np.var(log_returns_sabr)

    # Generate a normal distribution with the same mean and variance
    normal_dist_heston = np.random.normal(mean_heston, np.sqrt(var_heston), size=len(log_returns_heston))
    normal_dist_sabr = np.random.normal(mean_sabr, np.sqrt(var_sabr), size=len(log_returns_sabr))

    # Plotting histograms of log-returns for Heston and SABR models
    plt.figure(figsize=(12, 6))

    # Heston Model
    plt.subplot(2, 1, 1)
    plt.hist(log_returns_heston, bins=50, alpha=0.6, color='blue', label='Heston Log-Returns')
    plt.hist(normal_dist_heston, bins=50, alpha=0.6, color='orange', label='Normal Distribution')
    plt.title('Histogram of Log-Returns: Heston Model vs Normal Distribution')
    plt.legend()

    # SABR Model
    plt.subplot(2, 1, 2)
    plt.hist(log_returns_sabr, bins=50, alpha=0.6, color='green', label='SABR Log-Returns')
    plt.hist(normal_dist_sabr, bins=50, alpha=0.6, color='red', label='Normal Distribution')
    plt.title('Histogram of Log-Returns: SABR Model vs Normal Distribution')
    plt.legend()

    plt.tight_layout()
    plt.show()


def simulate_sabr_path(F0, sigma0, beta, rho, alpha, T, n_steps):
    """
    Function to simulate a path using the SABR model.

    Parameters:
    - F0: Initial forward rate
    - sigma0: Initial volatility
    - beta: Beta parameter controlling price dependence
    - rho: Correlation parameter
    - alpha: Volatility of volatility
    - T: Time horizon
    - n_steps: Number of simulation steps

    Returns:
    - F: Simulated forward rates
    - sigma: Simulated volatilities
    """
    dt = T / n_steps
    F = np.zeros(n_steps)
    sigma = np.zeros(n_steps)
    F[0] = F0
    sigma[0] = sigma0

    # Generate correlated random variables for forward rate and volatility
    z1 = np.random.normal(size=n_steps)
    z2 = np.random.normal(size=n_steps)
    z2 = rho * z1 + np.sqrt(1 - rho ** 2) * z2

    # Simulate the path
    for t in range(1, n_steps):
        F[t] = F[t - 1] + sigma[t - 1] * F[t - 1] ** beta * np.sqrt(dt) * z1[t]
        sigma[t] = sigma[t - 1] * np.exp(alpha * np.sqrt(dt) * z2[t])

    return F, sigma


def simulate_heston_path(S0, r, nu0, theta, rho, xi, kappa, T, n_steps):
    """
    Function to simulate a path using the Heston model.

    Parameters:
    - S0: Initial asset price
    - r: Risk-free rate
    - nu0: Initial variance
    - theta: Long-run mean of variance
    - rho: Correlation between price and variance processes
    - xi: Volatility of variance
    - kappa: Mean reversion speed of variance
    - T: Time horizon
    - n_steps: Number of simulation steps

    Returns:
    - S: Simulated asset prices
    - nu: Simulated variances
    """
    dt = T / n_steps
    S = np.zeros(n_steps)
    nu = np.zeros(n_steps)
    S[0] = S0
    nu[0] = nu0

    # Generate correlated random variables for asset price and volatility
    z1 = np.random.normal(size=n_steps)
    z2 = np.random.normal(size=n_steps)
    z2 = rho * z1 + np.sqrt(1 - rho ** 2) * z2

    # Simulate the path
    for t in range(1, n_steps):
        S[t] = S[t - 1] * np.exp((r - 0.5 * nu[t - 1]) * dt + np.sqrt(nu[t - 1] * dt) * z1[t])
        nu[t] = np.maximum(nu[t - 1] + kappa * (theta - nu[t - 1]) * dt + xi * np.sqrt(nu[t - 1] * dt) * z2[t], 0)

    # Plotting the Heston model path
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(S, label='Asset Price', color='green')
    plt.title('Heston Model: Asset Price Path')
    plt.ylabel('Price')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(nu, label='Variance', color='red')
    plt.title('Heston Model: Variance Path')
    plt.xlabel('Time Steps')
    plt.ylabel('Variance')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return S, nu


def european_option_mc_price(paths, strike, option_type='call'):
    """
    Function to calculate European option prices using Monte Carlo.

    Parameters:
    - paths: Simulated price paths
    - strike: Option strike price
    - option_type: 'call' or 'put'

    Returns:
    - Option price
    """
    payoffs = np.maximum(paths[:, -1] - strike, 0) if option_type == 'call' else np.maximum(strike - paths[:, -1], 0)
    price = np.mean(payoffs)
    return price


def sabr_volatility(F, K, T, alpha, beta, rho, vol0):
    """
    SABR model formula to calculate implied volatility.

    Parameters:
    - F: Forward price
    - K: Strike price
    - T: Time to maturity
    - alpha: Volatility of volatility
    - beta: Beta parameter
    - rho: Correlation
    - vol0: Initial volatility

    Returns:
    - Implied volatility
    """
    if F == K:
        # ATM formula
        fk_beta = (F * K) ** ((1 - beta) / 2)
        factor1 = alpha / fk_beta
        factor2 = (1 + ((1 - beta) ** 2 / 24) * (np.log(F / K) ** 2) + ((1 - beta) ** 4 / 1920) * (np.log(F / K) ** 4))
        return factor1 * factor2 * T
    else:
        # Non-ATM formula
        fk_beta = (F * K) ** ((1 - beta) / 2)
        z = alpha / vol0 * fk_beta * np.log(F / K)
        xz = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
        factor1 = alpha / (fk_beta * (1 + ((1 - beta) ** 2 / 24) * (np.log(F / K) ** 2) + ((1 - beta) ** 4 / 1920) * (
                    np.log(F / K) ** 4)))
        factor2 = z / xz
        factor3 = 1 + (
                    (1 - beta) ** 2 / 24 * alpha ** 2 / (fk_beta ** 2) + 0.25 * rho * beta * vol0 * alpha / fk_beta + (
                        2 - 3 * rho ** 2) / 24 * vol0 ** 2) * T
        return factor1 * factor2 * factor3


def plot_sabr_volatility(strike_range, F0, T, alpha, beta, rho, vol0, title):
    """
    Function to plot SABR implied volatilities with varying parameters.

    Parameters:
    - strike_range: Range of strike values to plot
    - F0: Forward price
    - T: Time to maturity
    - alpha: Volatility of volatility
    - beta: Beta parameter
    - rho: Correlation
    - vol0: Initial volatility
    - title: Plot title
    """
    sabr_vols = [sabr_volatility(F0, K, T, alpha, beta, rho, vol0) for K in strike_range]
    plt.plot(strike_range, sabr_vols, label=f'Alpha: {alpha}, Beta: {beta}, Rho: {rho}', marker='o')
    plt.title(title)
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.grid(True)


def black_scholes_price(F, K, sigma, T, r=0, option_type='call'):
    """
    Calculate Black-Scholes option price.

    Parameters:
    - F: Forward price
    - K: Strike price
    - sigma: Volatility
    - T: Time to maturity
    - r: Risk-free rate
    - option_type: 'call' or 'put'

    Returns:
    - Option price
    """
    d1 = (np.log(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = F * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - F * norm.cdf(-d1)

    return price


def simulate_sabr_paths(F0, sigma0, alpha, beta, rho, T, num_simulations, dt):
    """
    Function to simulate paths using the SABR model.

    Parameters:
    - F0: Initial forward rate
    - sigma0: Initial volatility
    - alpha: Volatility of volatility
    - beta: Beta parameter
    - rho: Correlation
    - T: Time horizon
    - num_simulations: Number of simulations
    - dt: Time step

    Returns:
    - F: Simulated forward rates
    - sigma: Simulated volatilities
    """
    num_steps = int(T / dt)
    F = np.zeros((num_simulations, num_steps))
    sigma = np.zeros((num_simulations, num_steps))

    # Initial values
    F[:, 0] = F0
    sigma[:, 0] = sigma0

    # Simulate paths
    for t in range(1, num_steps):
        Z1 = np.random.normal(size=num_simulations)
        Z2 = np.random.normal(size=num_simulations)
        W1 = Z1
        W2 = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2

        sigma[:, t] = sigma[:, t - 1] * np.exp(alpha * np.sqrt(dt) * W2 - 0.5 * alpha ** 2 * dt)
        F[:, t] = F[:, t - 1] * np.exp(
            sigma[:, t - 1] * np.power(F[:, t - 1], beta - 1) * np.sqrt(dt) * W1 - 0.5 * sigma[:,
                                                                                         t - 1] ** 2 * np.power(
                F[:, t - 1], 2 * (beta - 1)) * dt)

    return F, sigma


def sabr_error(params, strikes, market_vols, F, T):
    """
    Objective function for SABR model fitting.

    Parameters:
    - params: Model parameters to optimize [alpha, rho, vol0]
    - strikes: Strike prices
    - market_vols: Market implied volatilities
    - F: Forward price
    - T: Time to maturity

    Returns:
    - Sum of squared errors
    """
    alpha, rho, vol0 = params
    beta = 1  # Fixed beta
    errors = 0
    for K, market_vol in zip(strikes, market_vols):
        sabr_vol = sabr_volatility(F, K, T, alpha, beta, rho, vol0)
        errors += (sabr_vol - market_vol) ** 2
    return errors


def main():
    # Parameters
    S1 = 600  # Current price of S1
    S2 = 300  # Current price of S2
    sigma1 = 0.33  # Volatility of S1
    sigma2 = 0.57  # Volatility of S2
    rho = 0.3  # Correlation coefficient
    K = S1 - S2  # Strike price (ATM option)
    T = 1  # Time to maturity (1 year)
    r = 0  # Risk-free rate (0 for futures under risk-neutral measure)
    n_simulations = 10000  # Number of Monte Carlo simulations
    n_steps = 252  # Number of steps (trading days in a year)

    # PART A
    option_value, payoffs = value_european_spread_option(S1, S2, sigma1, sigma2, rho, T, r, n_simulations, n_steps)
    print(f"Estimated value of the European spread option: {option_value:.2f}")

    # Plotting the histogram of payoffs
    plt.figure(figsize=(10, 6))
    plt.hist(payoffs, bins=50, color='blue', alpha=0.7)
    plt.title('Distribution of Payoffs for European Spread Call Option')
    plt.xlabel('Payoff')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # PART B
    rho_values = np.linspace(-1, 1, 50)
    option_values = []

    # Calculate option values for each correlation value
    for rho in rho_values:
        option_value1, _ = value_european_spread_option(S1, S2, sigma1, sigma2, rho, T, r, n_simulations, n_steps)
        option_values.append(option_value1)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(rho_values, option_values, color='blue', marker='o')
    plt.title('European Spread Option Value vs Correlation')
    plt.xlabel('Correlation')
    plt.ylabel('Option Value')
    plt.grid(True)
    plt.show()

    # PART C
    sigma_p_squared = sigma1 ** 2 + sigma2 ** 2 - 2 * sigma1 * sigma2 * rho
    # Calculate volatility of the spread (standard deviation)
    sigma_p = np.sqrt(sigma_p_squared)
    print(f"Volatility of the spread: {sigma_p:.2f}")

    # PART D
    # Black-Scholes call option price
    S = S1 - S2  # Current value of the spread
    call_price_BS = bs_value(S, K, T, r, sigma_p)
    print(f"Black-Scholes call option price: {call_price_BS:.2f}")
    methods = ['Monte Carlo Simulation', 'Black-Scholes']
    values = [option_value, call_price_BS]

    # Creating the plot
    plt.figure(figsize=(8, 6))
    plt.bar(methods, values, color=['blue', 'green'])
    plt.title('Comparison of Spread Option Pricing Methods')
    plt.ylabel('Option Price')
    plt.ylim(0, max(values) + 50)  # Setting y-axis limit a bit higher for better visualization
    plt.grid(axis='y')
    for i, v in enumerate(values):
        plt.text(i, v + 10, f"€{v:.2f}", ha='center', va='bottom')  # Adding value labels
    plt.show()

    # PART E
    # Spread's current value and strike price (ATM)

    # Standard deviation (volatility) of the spread
    sigma = sigma_p  # Volatility from part c)

    # Bachelier model components
    d = (S - K) / (sigma * np.sqrt(T))
    call_price_Bachelier = (S - K) * norm.cdf(d) + sigma * np.sqrt(T) * norm.pdf(d)
    print(f"Bachelier call option price: {call_price_Bachelier:.2f}")

    methods = ['Monte Carlo Simulation', 'Black-Scholes', 'Bachelier']
    values = [option_value, call_price_BS, call_price_Bachelier]

    # Creating the plot
    plt.figure(figsize=(10, 6))
    plt.bar(methods, values, color=['blue', 'green', 'red'])
    plt.title('Comparison of Spread Option Pricing Methods')
    plt.ylabel('Option Price')
    plt.ylim(0, max(values) + 50)  # Setting y-axis limit a bit higher for better visualization
    plt.grid(axis='y')
    for i, v in enumerate(values):
        plt.text(i, v + 10, f"€{v:.2f}", ha='center', va='bottom')  # Adding value labels
    plt.show()

    # PART F
    # Define a range of strike prices
    strike_range = np.linspace(S - 100, S + 100, 20)

    # Initialize lists to store discrepancies
    discrepancies_BS_MC = []
    discrepancies_Bachelier_MC = []

    # Calculate discrepancies for each strike price
    for strike in strike_range:
        # Monte Carlo price
        mc_price, _ = value_european_spread_option(S1, S2, sigma1, sigma2, rho, T, r, n_simulations, n_steps)

        # Black-Scholes price
        bs_price = bs_value(S, strike, T, r, sigma_p)

        # Bachelier price
        d_bachelier = (S - strike) / (sigma * np.sqrt(T))
        bachelier_price = (S - strike) * norm.cdf(d_bachelier) + sigma * np.sqrt(T) * norm.pdf(d_bachelier)

        # Calculate discrepancies
        discrepancies_BS_MC.append(abs(bs_price - mc_price))
        discrepancies_Bachelier_MC.append(abs(bachelier_price - mc_price))

    # Plotting the discrepancies
    plt.figure(figsize=(10, 6))
    plt.plot(strike_range, discrepancies_BS_MC, label='BS vs MC', marker='o')
    plt.plot(strike_range, discrepancies_Bachelier_MC, label='Bachelier vs MC', marker='x')
    plt.xlabel('Strike Price')
    plt.ylabel('Price Discrepancy')
    plt.title('Discrepancy vs Strike Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Exercise 10
    # Heston model parameters
    S0_heston = 1
    r_heston = 0
    nu0_heston = 0.04
    theta_heston = nu0_heston
    rho_heston = -0.7
    xi_heston = 0.5
    kappa_heston = 1
    T = 1  # Time to maturity
    dt = 1 / 252  # Daily time step
    n_steps = int(T / dt)

    # SABR model parameters
    F0_sabr = 1
    sigma0_sabr = 0.2
    beta_sabr = 1
    rho_sabr = -0.7
    alpha_sabr = 0.5

    # Simulating a single path for the Heston model
    S_heston, nu_heston = simulate_heston_path(S0_heston, r_heston, nu0_heston, theta_heston, rho_heston, xi_heston,
                                               kappa_heston, T, n_steps)

    # Simulating a single path for the SABR model
    F_sabr, sigma_sabr = simulate_sabr_path(F0_sabr, sigma0_sabr, beta_sabr, rho_sabr, alpha_sabr, T, n_steps)

    # Plotting the SABR model path
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(F_sabr, label='Forward Rate', color='green')
    plt.title('SABR Model: Forward Rate Path')
    plt.ylabel('Rate')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(sigma_sabr, label='Volatility')
    plt.title('SABR Model: Volatility Path')
    plt.xlabel('Time Steps')
    plt.ylabel('Volatility')
    plt.legend()

    plt.tight_layout()
    plt.show()

    compare_dist(S_heston, F_sabr)

    # PART A
    sabr_model = Hagan2002LognormalSABR(f=F0_sabr, shift=0.1, t=T, v_atm_n=0.25,
                                        beta=1, rho=-0.7, volvol=sigma0_sabr)
    # Calculate the average volatility from the SABR simulations
    average_volatility = sabr_model.lognormal_vol(1)

    F_paths, sigma_paths = simulate_sabr_paths(F0_sabr, average_volatility, alpha_sabr, beta_sabr, rho_sabr, T,
                                               n_simulations, dt)

    # Strike prices for ATM, +/-10%, +/-25%
    strikes = [F0_sabr * factor for factor in [0.75, 0.9, 1, 1.1, 1.25]]

    # Calculate option prices using SABR Monte Carlo
    sabr_prices = {'calls': [], 'puts': []}
    for strike in strikes:
        sabr_prices['calls'].append(european_option_mc_price(F_paths, strike, 'call'))
        sabr_prices['puts'].append(european_option_mc_price(F_paths, strike, 'put'))

    # Calculate Black-Scholes prices
    bs_prices = {'calls': [], 'puts': []}
    for strike in strikes:
        bs_prices['calls'].append(black_scholes_price(F0_sabr, strike, average_volatility, T, option_type='call'))
        bs_prices['puts'].append(black_scholes_price(F0_sabr, strike, average_volatility, T, option_type='put'))

    # Setting up the data for plotting
    strike_percentages = ['25% ITM', '10% ITM', 'ATM', '10% OTM', '25% OTM']
    call_prices_sabr = sabr_prices['calls']
    put_prices_sabr = sabr_prices['puts']
    call_prices_bs = bs_prices['calls']
    put_prices_bs = bs_prices['puts']

    # Creating the plot
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))

    # Plotting call prices
    ax[0].plot(strike_percentages, call_prices_sabr, label='SABR - Calls', marker='o')
    ax[0].plot(strike_percentages, call_prices_bs, label='BS - Calls', marker='x')
    ax[0].set_title('European Call Option Prices')
    ax[0].set_ylabel('Option Price')
    ax[0].legend()

    # Plotting put prices
    ax[1].plot(strike_percentages, put_prices_sabr, label='SABR - Puts', marker='o', color='green')
    ax[1].plot(strike_percentages, put_prices_bs, label='BS - Puts', marker='x', color='red')
    ax[1].set_title('European Put Option Prices')
    ax[1].set_xlabel('Moneyness')
    ax[1].set_ylabel('Option Price')
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    # Creating the plot
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))

    # Plotting call prices
    ax[0].plot(strike_percentages, call_prices_sabr, label='SABR MC - Calls', marker='o')
    ax[0].plot(strike_percentages, call_prices_bs, label='BS - Calls', marker='x')
    ax[0].set_title('Comparison of European Call Option Prices')
    ax[0].set_ylabel('Option Price')
    ax[0].legend()

    # Plotting put prices
    ax[1].plot(strike_percentages, put_prices_sabr, label='SABR MC - Puts', marker='o', color='green')
    ax[1].plot(strike_percentages, put_prices_bs, label='BS - Puts', marker='x', color='red')
    ax[1].set_title('Comparison of European Put Option Prices')
    ax[1].set_xlabel('Moneyness')
    ax[1].set_ylabel('Option Price')
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    # PART B
    strike_range = np.linspace(0.75, 1.25, 50)  # Range of strike prices

    # Calculate SABR implied volatilities
    sabr_model = Hagan2002LognormalSABR(f=F0_sabr, shift=0.1, t=T, v_atm_n=0.25,
                                        beta=1, rho=-0.7, volvol=sigma0_sabr)
    # Calculate the average volatility from the SABR simulations

    sabr_implied_vols = [sabr_model.lognormal_vol(K) for K in strike_range]

    # Calculate Black-Scholes implied volatilities using the average volatility from previous part
    bs_implied_vols = [average_volatility for _ in strike_range]

    # Plot the implied volatilities
    plt.figure(figsize=(12, 6))
    plt.plot(strike_range, sabr_implied_vols, label='SABR Implied Volatility', marker='o')
    plt.plot(strike_range, bs_implied_vols, label='Black-Scholes Implied Volatility', linestyle='--')
    plt.title('Implied Volatility Smile - SABR vs Black-Scholes')
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.legend()
    plt.grid(True)
    plt.show()

    # PART C
    # Plot 1: Varying Alpha (Vol-of-vol)
    plt.figure(figsize=(15, 4))  # New figure for Plot 1
    for alpha in [0.3, 0.5, 0.7]:
        plot_sabr_volatility(strike_range, F0_sabr, T, alpha, beta_sabr, rho_sabr, sigma0_sabr,
                             "Varying Alpha (Vol-of-vol)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 2: Varying Beta (Elasticity parameter)
    plt.figure(figsize=(15, 4))  # New figure for Plot 2
    for beta in [0, 0.5, 1]:
        plot_sabr_volatility(strike_range, F0_sabr, T, alpha, beta, rho_sabr, sigma0_sabr,
                             "Varying Beta (Elasticity parameter)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 3: Varying Rho (Correlation)
    plt.figure(figsize=(15, 4))  # New figure for Plot 3
    for rho in [-0.9, -0.7, -0.5]:
        plot_sabr_volatility(strike_range, F0_sabr, T, alpha, beta, rho, sigma0_sabr, "Varying Rho (Correlation)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # PART D
    # Example market data (to be replaced with real data)
    market_vols = np.array([0.2, 0.205, 0.21, 0.22, 0.23])  # Example implied volatilities
    strikes = np.array([1500, 1650, 1750, 1800, 1900])  # Corresponding strike prices
    F = 1700  # Forward price
    T = 1  # Time to expiration

    # Initial parameter guesses
    initial_params = [0.2, 0, 0.2]  # [alpha, rho, vol0]

    # Optimization
    result = minimize(sabr_error, initial_params, args=(strikes, market_vols, F, T),
                      method='Nelder-Mead')

    optimized_params = result.x
    alpha_opt, rho_opt, vol0_opt = optimized_params

    # Calculating fitted SABR volatilities
    fitted_vols = [sabr_volatility(F, K, T, alpha_opt, 1, rho_opt, vol0_opt) for K in strikes]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(strikes, market_vols, 'o', label='Market Implied Vols')
    plt.plot(strikes, fitted_vols, label='Fitted SABR Vols')
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.title('SABR Model Fitting to Implied Volatility Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()