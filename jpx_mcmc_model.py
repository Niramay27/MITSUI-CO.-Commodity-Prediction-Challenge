"""
MCMC Model for JPX Futures Simulation
Bayesian simulation of JPX Platinum vs LME Zinc spread
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class JPXMCMCModel:
    """
    MCMC Model for simulating JPX futures spreads using Bayesian inference.
    
    This model implements:
    1. Bivariate Ornstein-Uhlenbeck process for mean reversion
    2. Dynamic correlation modeling
    3. Fat-tailed error distributions (Student-t)
    4. Bayesian parameter estimation via MCMC
    """
    
    def __init__(self, target_name="target_5"):
        self.target_name = target_name
        self.params = {}
        self.mcmc_samples = None
        self.simulation_results = None
        
    def load_data(self):
        """Load and prepare data for MCMC modeling."""
        print("📊 Loading JPX MCMC Data...")
        
        # Load datasets
        train = pd.read_csv('train.csv')
        train_labels = pd.read_csv('train_labels.csv')
        
        # Get target data
        target_data = train_labels[self.target_name].dropna()
        
        # Get feature data (LME Zinc and JPX Platinum)
        lme_zs = train['LME_ZS_Close'].dropna()
        jpx_pt = train['JPX_Platinum_Standard_Futures_Close'].dropna()
        
        # Align data by index
        common_idx = target_data.index.intersection(lme_zs.index).intersection(jpx_pt.index)
        
        self.target_data = target_data.loc[common_idx]
        self.lme_zs = lme_zs.loc[common_idx]
        self.jpx_pt = jpx_pt.loc[common_idx]
        
        # Calculate returns
        self.lme_returns = self.lme_zs.pct_change().dropna()
        self.jpx_returns = self.jpx_pt.pct_change().dropna()
        
        # Align target with returns
        common_returns_idx = self.target_data.index.intersection(self.lme_returns.index).intersection(self.jpx_returns.index)
        self.target_aligned = self.target_data.loc[common_returns_idx]
        self.lme_returns_aligned = self.lme_returns.loc[common_returns_idx]
        self.jpx_returns_aligned = self.jpx_returns.loc[common_returns_idx]
        
        print(f"   Aligned observations: {len(self.target_aligned)}")
        print(f"   Target mean: {self.target_aligned.mean():.6f}")
        print(f"   Target std: {self.target_aligned.std():.6f}")
        
        return self.target_aligned, self.lme_returns_aligned, self.jpx_returns_aligned
    
    def estimate_initial_parameters(self):
        """Estimate initial parameters using MLE."""
        print("\n⚙️ Estimating Initial Parameters...")
        
        # Basic statistics
        target_mean = self.target_aligned.mean()
        target_std = self.target_aligned.std()
        
        lme_mean = self.lme_returns_aligned.mean()
        lme_std = self.lme_returns_aligned.std()
        
        jpx_mean = self.jpx_returns_aligned.mean()
        jpx_std = self.jpx_returns_aligned.std()
        
        # Correlation
        correlation = np.corrcoef(self.lme_returns_aligned, self.jpx_returns_aligned)[0,1]
        target_lme_corr = np.corrcoef(self.target_aligned, self.lme_returns_aligned)[0,1]
        target_jpx_corr = np.corrcoef(self.target_aligned, self.jpx_returns_aligned)[0,1]
        
        # Mean reversion parameters (simple AR(1) estimation)
        def ar1_likelihood(params, data):
            mu, phi, sigma = params
            residuals = data[1:] - mu - phi * (data[:-1] - mu)
            return -np.sum(stats.norm.logpdf(residuals, 0, sigma))
        
        # Estimate AR(1) for target
        target_result = minimize(ar1_likelihood, [target_mean, 0.1, target_std], 
                               args=(self.target_aligned.values,), method='L-BFGS-B',
                               bounds=[(-0.1, 0.1), (-0.99, 0.99), (0.001, 1.0)])
        
        self.params = {
            'target_mu': target_result.x[0],
            'target_phi': target_result.x[1], 
            'target_sigma': target_result.x[2],
            'lme_mu': lme_mean,
            'lme_sigma': lme_std,
            'jpx_mu': jpx_mean,
            'jpx_sigma': jpx_std,
            'correlation': correlation,
            'target_lme_beta': target_lme_corr * target_std / lme_std,
            'target_jpx_beta': target_jpx_corr * target_std / jpx_std,
            'nu': 5.0  # degrees of freedom for t-distribution
        }
        
        print("   Initial parameter estimates:")
        for key, value in self.params.items():
            print(f"      {key}: {value:.6f}")
        
        return self.params
    
    def mcmc_sampler(self, n_samples=10000, burn_in=2000):
        """
        MCMC sampler using Metropolis-Hastings algorithm.
        """
        print(f"\n🎲 Running MCMC Sampling ({n_samples:,} samples)...")
        
        # Initialize parameters
        current_params = self.params.copy()
        samples = []
        n_accepted = 0
        
        # Proposal standard deviations
        proposal_stds = {
            'target_mu': 0.001,
            'target_phi': 0.05,
            'target_sigma': 0.001,
            'target_lme_beta': 0.1,
            'target_jpx_beta': 0.1,
            'nu': 1.0
        }
        
        for i in range(n_samples + burn_in):
            # Propose new parameters
            proposed_params = current_params.copy()
            
            # Update one parameter at a time
            param_to_update = np.random.choice(list(proposal_stds.keys()))
            proposed_params[param_to_update] += np.random.normal(0, proposal_stds[param_to_update])
            
            # Bounds checking
            proposed_params['target_phi'] = np.clip(proposed_params['target_phi'], -0.99, 0.99)
            proposed_params['target_sigma'] = max(proposed_params['target_sigma'], 0.001)
            proposed_params['nu'] = max(proposed_params['nu'], 2.1)
            
            # Calculate likelihood
            current_ll = self._log_likelihood(current_params)
            proposed_ll = self._log_likelihood(proposed_params)
            
            # Metropolis-Hastings acceptance
            alpha = min(1, np.exp(proposed_ll - current_ll))
            
            if np.random.random() < alpha:
                current_params = proposed_params
                n_accepted += 1
            
            # Store sample after burn-in
            if i >= burn_in:
                samples.append(current_params.copy())
            
            # Progress update
            if (i + 1) % 2000 == 0:
                acceptance_rate = n_accepted / (i + 1)
                print(f"   Sample {i+1:,}: Acceptance rate = {acceptance_rate:.3f}")
        
        self.mcmc_samples = pd.DataFrame(samples)
        
        final_acceptance = n_accepted / (n_samples + burn_in)
        print(f"   Final acceptance rate: {final_acceptance:.3f}")
        
        return self.mcmc_samples
    
    def _log_likelihood(self, params):
        """Calculate log-likelihood of the model."""
        try:
            # Predicted target values using the model
            predicted = (params['target_mu'] + 
                        params['target_lme_beta'] * self.lme_returns_aligned +
                        params['target_jpx_beta'] * self.jpx_returns_aligned)
            
            # Add AR(1) component
            ar_component = np.zeros_like(predicted)
            for i in range(1, len(predicted)):
                ar_component[i] = params['target_phi'] * (self.target_aligned.iloc[i-1] - params['target_mu'])
            
            predicted += ar_component
            
            # Residuals
            residuals = self.target_aligned.values - predicted
            
            # Student-t likelihood
            ll = np.sum(stats.t.logpdf(residuals, df=params['nu'], scale=params['target_sigma']))
            
            return ll if np.isfinite(ll) else -np.inf
            
        except:
            return -np.inf
    
    def analyze_mcmc_results(self):
        """Analyze MCMC convergence and parameter estimates."""
        if self.mcmc_samples is None:
            print("❌ No MCMC samples available. Run mcmc_sampler first.")
            return
        
        print("\n📈 MCMC Results Analysis:")
        
        # Parameter summaries
        param_summary = self.mcmc_samples.describe()
        print("\n   Parameter Posterior Summaries:")
        print(param_summary)
        
        # Effective sample sizes (simple autocorrelation-based estimate)
        print("\n   Effective Sample Sizes:")
        for param in self.mcmc_samples.columns:
            if param in ['target_mu', 'target_phi', 'target_sigma', 'target_lme_beta', 'target_jpx_beta']:
                autocorr = self._autocorrelation(self.mcmc_samples[param].values)
                ess = len(self.mcmc_samples) / (1 + 2 * np.sum(autocorr[1:50]))
                print(f"      {param}: {ess:.0f}")
        
        return param_summary
    
    def _autocorrelation(self, x, max_lag=50):
        """Calculate autocorrelation function."""
        n = len(x)
        x = x - np.mean(x)
        autocorr = np.correlate(x, x, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]
        return autocorr[:max_lag+1]
    
    def simulate_scenarios(self, n_scenarios=50000, forecast_horizon=1):
        """
        Simulate future scenarios using posterior parameter distribution.
        """
        print(f"\n🚀 Simulating {n_scenarios:,} Scenarios...")
        
        if self.mcmc_samples is None:
            print("❌ No MCMC samples available. Run mcmc_sampler first.")
            return None
        
        # Sample parameters from posterior
        n_mcmc_samples = len(self.mcmc_samples)
        scenario_results = []
        
        # Get last observed values for starting the simulation
        last_target = self.target_aligned.iloc[-1]
        last_lme = self.lme_returns_aligned.iloc[-1]
        last_jpx = self.jpx_returns_aligned.iloc[-1]
        
        for scenario in range(n_scenarios):
            # Sample parameters from MCMC chain
            param_idx = np.random.randint(0, n_mcmc_samples)
            params = self.mcmc_samples.iloc[param_idx]
            
            # Simulate LME and JPX returns (simple random walk with drift)
            lme_sim = np.random.normal(params['lme_mu'], params['lme_sigma'], forecast_horizon)
            jpx_sim = np.random.normal(params['jpx_mu'], params['jpx_sigma'], forecast_horizon)
            
            # Add correlation structure
            correlation_adjustment = params['correlation'] * (lme_sim - params['lme_mu'])
            jpx_sim += correlation_adjustment * params['jpx_sigma'] / params['lme_sigma']
            
            # Simulate target using the model
            for h in range(forecast_horizon):
                if h == 0:
                    # Use AR(1) component from last observed target
                    ar_component = params['target_phi'] * (last_target - params['target_mu'])
                    
                    # Predict target
                    target_pred = (params['target_mu'] + 
                                 params['target_lme_beta'] * lme_sim[h] +
                                 params['target_jpx_beta'] * jpx_sim[h] +
                                 ar_component)
                    
                    # Add noise
                    target_sim = target_pred + stats.t.rvs(df=params['nu'], scale=params['target_sigma'])
                
                else:
                    # For multi-step forecasting (though we're doing 1-step here)
                    ar_component = params['target_phi'] * (target_sim - params['target_mu'])
                    target_pred = (params['target_mu'] + 
                                 params['target_lme_beta'] * lme_sim[h] +
                                 params['target_jpx_beta'] * jpx_sim[h] +
                                 ar_component)
                    target_sim = target_pred + stats.t.rvs(df=params['nu'], scale=params['target_sigma'])
            
            scenario_results.append({
                'scenario': scenario,
                'target_prediction': target_sim,
                'lme_return': lme_sim[-1],
                'jpx_return': jpx_sim[-1],
                'param_sample': param_idx
            })
            
            # Progress update
            if (scenario + 1) % 10000 == 0:
                print(f"   Completed {scenario+1:,} scenarios")
        
        self.simulation_results = pd.DataFrame(scenario_results)
        
        print(f"✅ Simulation complete: {len(self.simulation_results)} scenarios")
        return self.simulation_results
    
    def calculate_consensus_metrics(self):
        """Calculate consensus-based metrics from simulation results."""
        if self.simulation_results is None:
            print("❌ No simulation results available.")
            return None
        
        print("\n📊 Consensus Metrics Analysis:")
        
        predictions = self.simulation_results['target_prediction']
        
        # Basic statistics
        consensus_mean = predictions.mean()
        consensus_std = predictions.std()
        consensus_median = predictions.median()
        
        # Confidence intervals
        ci_95 = np.percentile(predictions, [2.5, 97.5])
        ci_90 = np.percentile(predictions, [5, 95])
        ci_80 = np.percentile(predictions, [10, 90])
        
        # Probability metrics
        prob_positive = (predictions > 0).mean()
        prob_large_move = (np.abs(predictions) > 0.02).mean()  # 2% move
        
        # Risk metrics
        var_95 = np.percentile(predictions, 5)  # 5% VaR
        var_99 = np.percentile(predictions, 1)  # 1% VaR
        expected_shortfall = predictions[predictions <= var_95].mean()
        
        metrics = {
            'consensus_mean': consensus_mean,
            'consensus_median': consensus_median,
            'consensus_std': consensus_std,
            'ci_95_lower': ci_95[0],
            'ci_95_upper': ci_95[1],
            'ci_90_lower': ci_90[0],
            'ci_90_upper': ci_90[1],
            'ci_80_lower': ci_80[0],
            'ci_80_upper': ci_80[1],
            'prob_positive': prob_positive,
            'prob_large_move': prob_large_move,
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall': expected_shortfall,
            'skewness': stats.skew(predictions),
            'kurtosis': stats.kurtosis(predictions)
        }
        
        print("   Consensus Statistics:")
        print(f"      Mean: {consensus_mean:.6f}")
        print(f"      Median: {consensus_median:.6f}")
        print(f"      Std Dev: {consensus_std:.6f}")
        print(f"      95% CI: [{ci_95[0]:.6f}, {ci_95[1]:.6f}]")
        print(f"      P(target > 0): {prob_positive:.3f}")
        print(f"      P(|target| > 2%): {prob_large_move:.3f}")
        print(f"      VaR (95%): {var_95:.6f}")
        print(f"      Expected Shortfall: {expected_shortfall:.6f}")
        
        self.consensus_metrics = metrics
        return metrics
    
    def create_simulation_plots(self):
        """Create comprehensive plots of simulation results."""
        if self.simulation_results is None:
            print("❌ No simulation results available.")
            return
        
        from pathlib import Path
        Path('plots').mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('JPX MCMC Simulation Results (50,000 Scenarios)', fontsize=16)
        
        predictions = self.simulation_results['target_prediction']
        
        # 1. Histogram of predictions
        axes[0,0].hist(predictions, bins=100, alpha=0.7, color='blue', density=True)
        axes[0,0].axvline(predictions.mean(), color='red', linestyle='--', label=f'Mean: {predictions.mean():.6f}')
        axes[0,0].axvline(predictions.median(), color='orange', linestyle='--', label=f'Median: {predictions.median():.6f}')
        axes[0,0].set_title('Prediction Distribution')
        axes[0,0].set_xlabel('Target Prediction')
        axes[0,0].set_ylabel('Density')
        axes[0,0].legend()
        
        # 2. Q-Q plot vs normal
        stats.probplot(predictions, dist="norm", plot=axes[0,1])
        axes[0,1].set_title('Q-Q Plot vs Normal Distribution')
        
        # 3. MCMC parameter traces (if available)
        if self.mcmc_samples is not None:
            axes[0,2].plot(self.mcmc_samples['target_mu'])
            axes[0,2].set_title('MCMC Trace: Target Mean')
            axes[0,2].set_xlabel('Iteration')
            axes[0,2].set_ylabel('Value')
        
        # 4. Box plot of predictions
        axes[1,0].boxplot(predictions)
        axes[1,0].set_title('Prediction Box Plot')
        axes[1,0].set_ylabel('Target Prediction')
        
        # 5. Cumulative distribution
        sorted_predictions = np.sort(predictions)
        cumulative = np.arange(1, len(sorted_predictions) + 1) / len(sorted_predictions)
        axes[1,1].plot(sorted_predictions, cumulative)
        axes[1,1].set_title('Cumulative Distribution Function')
        axes[1,1].set_xlabel('Target Prediction')
        axes[1,1].set_ylabel('Cumulative Probability')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Risk metrics visualization
        var_95 = np.percentile(predictions, 5)
        var_99 = np.percentile(predictions, 1)
        axes[1,2].hist(predictions, bins=100, alpha=0.7, color='lightblue', density=True)
        axes[1,2].axvline(var_95, color='red', linestyle='--', label=f'VaR 95%: {var_95:.6f}')
        axes[1,2].axvline(var_99, color='darkred', linestyle='--', label=f'VaR 99%: {var_99:.6f}')
        axes[1,2].set_title('Risk Metrics')
        axes[1,2].set_xlabel('Target Prediction')
        axes[1,2].legend()
        
        plt.tight_layout()
        plt.savefig('plots/jpx_mcmc_simulation_results.png', dpi=300, bbox_inches='tight')
        print("📊 Simulation plots saved to plots/jpx_mcmc_simulation_results.png")

def main():
    """Main execution function."""
    print("🇯🇵 JPX MCMC Simulation Pipeline")
    print("="*60)
    
    # Initialize model
    model = JPXMCMCModel()
    
    # Load and prepare data
    target_data, lme_returns, jpx_returns = model.load_data()
    
    # Estimate initial parameters
    initial_params = model.estimate_initial_parameters()
    
    # Run MCMC sampling
    mcmc_samples = model.mcmc_sampler(n_samples=5000, burn_in=1000)  # Reduced for demo
    
    # Analyze MCMC results
    param_summary = model.analyze_mcmc_results()
    
    # Run simulation
    simulation_results = model.simulate_scenarios(n_scenarios=50000)
    
    # Calculate consensus metrics
    consensus_metrics = model.calculate_consensus_metrics()
    
    # Create plots
    model.create_simulation_plots()
    
    print("\n" + "="*60)
    print("🎉 JPX MCMC Simulation Complete!")
    
    return model

if __name__ == "__main__":
    model = main()
