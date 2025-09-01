"""
MCMC Model Evaluation and Competition Metrics Analysis
Evaluate the JPX MCMC simulation results against competition requirements
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class MCMCEvaluator:
    """
    Comprehensive evaluation of MCMC simulation results.
    """
    
    def __init__(self):
        self.model_results = None
        self.backtest_results = None
        self.competition_metrics = None
        
    def load_model_results(self, model):
        """Load results from the MCMC model."""
        print("📊 Loading MCMC Model Results...")
        
        self.model = model
        self.simulation_results = model.simulation_results
        self.consensus_metrics = model.consensus_metrics
        self.mcmc_samples = model.mcmc_samples
        
        print(f"   Loaded {len(self.simulation_results)} simulation scenarios")
        print(f"   Consensus mean prediction: {self.consensus_metrics['consensus_mean']:.6f}")
        
        return self.simulation_results
    
    def backtest_model(self, validation_split=0.2):
        """
        Backtest the model on historical data using walk-forward validation.
        """
        print(f"\n🔄 Backtesting Model (validation split: {validation_split:.0%})...")
        
        # Get aligned data
        target_data = self.model.target_aligned
        lme_returns = self.model.lme_returns_aligned
        jpx_returns = self.model.jpx_returns_aligned
        
        # Split into train/validation
        n_total = len(target_data)
        n_train = int(n_total * (1 - validation_split))
        
        train_target = target_data.iloc[:n_train]
        val_target = target_data.iloc[n_train:]
        
        train_lme = lme_returns.iloc[:n_train]
        val_lme = lme_returns.iloc[n_train:]
        
        train_jpx = jpx_returns.iloc[:n_train]
        val_jpx = jpx_returns.iloc[n_train:]
        
        print(f"   Training samples: {len(train_target)}")
        print(f"   Validation samples: {len(val_target)}")
        
        # Use posterior mean parameters for prediction
        posterior_means = self.mcmc_samples.mean()
        
        # Generate predictions for validation set
        predictions = []
        prediction_intervals = []
        
        for i in range(len(val_target)):
            # Get the previous target value (for AR component)
            if i == 0:
                prev_target = train_target.iloc[-1]
            else:
                prev_target = val_target.iloc[i-1]
            
            # AR(1) component
            ar_component = posterior_means['target_phi'] * (prev_target - posterior_means['target_mu'])
            
            # Base prediction
            base_pred = (posterior_means['target_mu'] + 
                        posterior_means['target_lme_beta'] * val_lme.iloc[i] +
                        posterior_means['target_jpx_beta'] * val_jpx.iloc[i] +
                        ar_component)
            
            predictions.append(base_pred)
            
            # Generate prediction interval using model uncertainty
            # Sample multiple predictions from posterior
            interval_preds = []
            n_interval_samples = 1000
            
            for _ in range(n_interval_samples):
                # Sample parameters
                param_idx = np.random.randint(0, len(self.mcmc_samples))
                params = self.mcmc_samples.iloc[param_idx]
                
                ar_comp = params['target_phi'] * (prev_target - params['target_mu'])
                pred = (params['target_mu'] + 
                       params['target_lme_beta'] * val_lme.iloc[i] +
                       params['target_jpx_beta'] * val_jpx.iloc[i] +
                       ar_comp)
                
                # Add noise
                pred += stats.t.rvs(df=params['nu'], scale=params['target_sigma'])
                interval_preds.append(pred)
            
            # Calculate prediction intervals
            ci_lower = np.percentile(interval_preds, 2.5)
            ci_upper = np.percentile(interval_preds, 97.5)
            prediction_intervals.append((ci_lower, ci_upper))
        
        # Calculate metrics
        predictions = np.array(predictions)
        actual = val_target.values
        
        mse = mean_squared_error(actual, predictions)
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mse)
        
        # Directional accuracy
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actual)
        directional_accuracy = np.mean(pred_direction == actual_direction)
        
        # Coverage of prediction intervals
        lower_bounds = np.array([pi[0] for pi in prediction_intervals])
        upper_bounds = np.array([pi[1] for pi in prediction_intervals])
        coverage = np.mean((actual >= lower_bounds) & (actual <= upper_bounds))
        
        self.backtest_results = {
            'predictions': predictions,
            'actual': actual,
            'prediction_intervals': prediction_intervals,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy,
            'interval_coverage': coverage,
            'val_indices': val_target.index
        }
        
        print("   Backtest Results:")
        print(f"      RMSE: {rmse:.6f}")
        print(f"      MAE: {mae:.6f}")
        print(f"      Directional Accuracy: {directional_accuracy:.3f}")
        print(f"      95% Interval Coverage: {coverage:.3f}")
        
        return self.backtest_results
    
    def evaluate_competition_metrics(self):
        """
        Evaluate against competition-specific metrics.
        Based on typical financial forecasting competitions.
        """
        print("\n🏆 Competition Metrics Evaluation...")
        
        if self.backtest_results is None:
            print("❌ Run backtest_model first")
            return None
        
        predictions = self.backtest_results['predictions']
        actual = self.backtest_results['actual']
        
        # Standard regression metrics
        mse = self.backtest_results['mse']
        mae = self.backtest_results['mae']
        rmse = self.backtest_results['rmse']
        
        # R-squared
        ss_res = np.sum((actual - predictions) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Information Coefficient (correlation between predictions and actual)
        ic = np.corrcoef(predictions, actual)[0, 1]
        
        # Sharpe Ratio of predictions (if we traded based on predictions)
        pred_returns = predictions  # These are already returns/changes
        sharpe_ratio = np.mean(pred_returns) / np.std(pred_returns) if np.std(pred_returns) > 0 else 0
        
        # Hit Rate (directional accuracy)
        hit_rate = self.backtest_results['directional_accuracy']
        
        # Maximum Drawdown of cumulative prediction error
        cumulative_error = np.cumsum(np.abs(actual - predictions))
        running_max = np.maximum.accumulate(cumulative_error)
        drawdown = (cumulative_error - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Precision and Recall for significant moves
        threshold = 0.01  # 1% move threshold
        
        # Significant actual moves
        significant_actual = np.abs(actual) > threshold
        significant_predicted = np.abs(predictions) > threshold
        
        if np.sum(significant_predicted) > 0:
            precision = np.sum(significant_actual & significant_predicted) / np.sum(significant_predicted)
        else:
            precision = 0
            
        if np.sum(significant_actual) > 0:
            recall = np.sum(significant_actual & significant_predicted) / np.sum(significant_actual)
        else:
            recall = 0
        
        # Quantile Score (for probabilistic forecasting)
        # Using prediction intervals from backtest
        quantile_scores = self._calculate_quantile_scores(actual, predictions, 
                                                        self.backtest_results['prediction_intervals'])
        
        self.competition_metrics = {
            'rmse': rmse,
            'mae': mae,
            'r_squared': r_squared,
            'information_coefficient': ic,
            'sharpe_ratio': sharpe_ratio,
            'hit_rate': hit_rate,
            'max_drawdown': max_drawdown,
            'precision': precision,
            'recall': recall,
            'quantile_score_mean': np.mean(quantile_scores),
            'interval_coverage': self.backtest_results['interval_coverage']
        }
        
        print("   Competition Metrics:")
        print(f"      RMSE: {rmse:.6f}")
        print(f"      R²: {r_squared:.4f}")
        print(f"      Information Coefficient: {ic:.4f}")
        print(f"      Hit Rate: {hit_rate:.3f}")
        print(f"      Precision (1% moves): {precision:.3f}")
        print(f"      Recall (1% moves): {recall:.3f}")
        print(f"      Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"      Mean Quantile Score: {np.mean(quantile_scores):.6f}")
        
        return self.competition_metrics
    
    def _calculate_quantile_scores(self, actual, predictions, intervals, alpha=0.05):
        """Calculate quantile scores for probabilistic evaluation."""
        scores = []
        
        for i, (lower, upper) in enumerate(intervals):
            # Pinball loss for quantiles
            y_true = actual[i]
            
            # Lower quantile (alpha/2)
            if y_true < lower:
                score_lower = (alpha/2 - 1) * (y_true - lower)
            else:
                score_lower = (alpha/2) * (y_true - lower)
            
            # Upper quantile (1 - alpha/2)
            if y_true < upper:
                score_upper = ((1-alpha)/2 - 1) * (y_true - upper)
            else:
                score_upper = ((1-alpha)/2) * (y_true - upper)
            
            scores.append(score_lower + score_upper)
        
        return np.array(scores)
    
    def analyze_model_robustness(self):
        """Analyze robustness of the MCMC model."""
        print("\n🛡️ Model Robustness Analysis...")
        
        # Parameter stability analysis
        if self.mcmc_samples is not None:
            # Calculate effective sample sizes
            key_params = ['target_mu', 'target_phi', 'target_sigma', 'target_lme_beta', 'target_jpx_beta']
            ess_results = {}
            
            for param in key_params:
                if param in self.mcmc_samples.columns:
                    chain = self.mcmc_samples[param].values
                    autocorr = self._autocorrelation(chain)
                    ess = len(chain) / (1 + 2 * np.sum(autocorr[1:min(50, len(autocorr))]))
                    ess_results[param] = ess
            
            print("   MCMC Diagnostics:")
            for param, ess in ess_results.items():
                print(f"      {param} ESS: {ess:.0f}")
            
            # Geweke diagnostic (stationarity test)
            geweke_results = self._geweke_diagnostic(self.mcmc_samples[key_params])
            print("   Geweke Diagnostic (should be ~N(0,1)):")
            for param, z_score in geweke_results.items():
                status = "✅" if abs(z_score) < 2 else "⚠️"
                print(f"      {param}: {z_score:.3f} {status}")
        
        # Prediction stability
        if self.simulation_results is not None:
            pred_std = self.simulation_results['target_prediction'].std()
            pred_mean = self.simulation_results['target_prediction'].mean()
            
            # Coefficient of variation
            cv = pred_std / abs(pred_mean) if pred_mean != 0 else np.inf
            
            print(f"\n   Prediction Stability:")
            print(f"      Coefficient of Variation: {cv:.3f}")
            
            # Monte Carlo standard error
            mc_se = pred_std / np.sqrt(len(self.simulation_results))
            print(f"      Monte Carlo SE: {mc_se:.6f}")
    
    def _autocorrelation(self, x, max_lag=50):
        """Calculate autocorrelation function."""
        n = len(x)
        x = x - np.mean(x)
        autocorr = np.correlate(x, x, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]
        return autocorr[:max_lag+1]
    
    def _geweke_diagnostic(self, chains, first=0.1, last=0.5):
        """Geweke convergence diagnostic."""
        results = {}
        
        for param in chains.columns:
            chain = chains[param].values
            n = len(chain)
            
            # First 10% and last 50% of chain
            first_part = chain[:int(n * first)]
            last_part = chain[int(n * (1 - last)):]
            
            # Means
            mean1 = np.mean(first_part)
            mean2 = np.mean(last_part)
            
            # Spectral densities (simplified)
            var1 = np.var(first_part, ddof=1)
            var2 = np.var(last_part, ddof=1)
            
            # Z-score
            se_diff = np.sqrt(var1/len(first_part) + var2/len(last_part))
            z_score = (mean1 - mean2) / se_diff if se_diff > 0 else 0
            
            results[param] = z_score
        
        return results
    
    def create_evaluation_plots(self):
        """Create comprehensive evaluation plots."""
        print("\n📊 Creating Evaluation Plots...")
        
        from pathlib import Path
        Path('plots').mkdir(exist_ok=True)
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Backtest predictions vs actual
        ax1 = plt.subplot(3, 4, 1)
        if self.backtest_results is not None:
            plt.scatter(self.backtest_results['actual'], self.backtest_results['predictions'], 
                       alpha=0.6, s=30)
            plt.plot([-0.1, 0.1], [-0.1, 0.1], 'r--', label='Perfect Prediction')
            plt.xlabel('Actual Returns')
            plt.ylabel('Predicted Returns')
            plt.title(f'Predictions vs Actual\nR² = {self.competition_metrics["r_squared"]:.3f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 2. Residuals plot
        ax2 = plt.subplot(3, 4, 2)
        if self.backtest_results is not None:
            residuals = self.backtest_results['actual'] - self.backtest_results['predictions']
            plt.scatter(self.backtest_results['predictions'], residuals, alpha=0.6, s=30)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Returns')
            plt.ylabel('Residuals')
            plt.title('Residuals vs Predicted')
            plt.grid(True, alpha=0.3)
        
        # 3. Time series of predictions
        ax3 = plt.subplot(3, 4, 3)
        if self.backtest_results is not None:
            indices = range(len(self.backtest_results['actual']))
            plt.plot(indices, self.backtest_results['actual'], label='Actual', alpha=0.7)
            plt.plot(indices, self.backtest_results['predictions'], label='Predicted', alpha=0.7)
            
            # Add prediction intervals
            lower_bounds = np.array([pi[0] for pi in self.backtest_results['prediction_intervals']])
            upper_bounds = np.array([pi[1] for pi in self.backtest_results['prediction_intervals']])
            plt.fill_between(indices, lower_bounds, upper_bounds, alpha=0.2, label='95% PI')
            
            plt.xlabel('Time')
            plt.ylabel('Returns')
            plt.title('Time Series Validation')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 4. MCMC parameter traces
        ax4 = plt.subplot(3, 4, 4)
        if self.mcmc_samples is not None:
            plt.plot(self.mcmc_samples['target_mu'], alpha=0.7)
            plt.title('MCMC Trace: Target Mean')
            plt.xlabel('Iteration')
            plt.ylabel('Value')
            plt.grid(True, alpha=0.3)
        
        # 5. Parameter posterior distributions
        ax5 = plt.subplot(3, 4, 5)
        if self.mcmc_samples is not None:
            plt.hist(self.mcmc_samples['target_phi'], bins=50, alpha=0.7, density=True)
            plt.axvline(self.mcmc_samples['target_phi'].mean(), color='red', linestyle='--')
            plt.title('Posterior: AR(1) Coefficient')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.grid(True, alpha=0.3)
        
        # 6. Consensus prediction distribution
        ax6 = plt.subplot(3, 4, 6)
        if self.simulation_results is not None:
            predictions = self.simulation_results['target_prediction']
            plt.hist(predictions, bins=100, alpha=0.7, density=True, color='green')
            plt.axvline(predictions.mean(), color='red', linestyle='--', 
                       label=f'Mean: {predictions.mean():.6f}')
            plt.title('Consensus Prediction Distribution')
            plt.xlabel('Predicted Return')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 7. Model performance metrics
        ax7 = plt.subplot(3, 4, 7)
        if self.competition_metrics is not None:
            metrics = ['Hit Rate', 'Precision', 'Recall', 'R²', 'IC']
            values = [
                self.competition_metrics['hit_rate'],
                self.competition_metrics['precision'],
                self.competition_metrics['recall'],
                self.competition_metrics['r_squared'],
                self.competition_metrics['information_coefficient']
            ]
            
            bars = plt.bar(metrics, values, alpha=0.7, color=['blue', 'green', 'orange', 'red', 'purple'])
            plt.title('Model Performance Metrics')
            plt.ylabel('Score')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # 8. Risk metrics
        ax8 = plt.subplot(3, 4, 8)
        if self.consensus_metrics is not None:
            risk_data = [
                self.consensus_metrics['var_95'],
                self.consensus_metrics['var_99'],
                self.consensus_metrics['expected_shortfall']
            ]
            risk_labels = ['VaR 95%', 'VaR 99%', 'Expected Shortfall']
            
            colors = ['orange', 'red', 'darkred']
            bars = plt.bar(risk_labels, risk_data, color=colors, alpha=0.7)
            plt.title('Risk Metrics')
            plt.ylabel('Value')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            for bar, value in zip(bars, risk_data):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.002,
                        f'{value:.4f}', ha='center', va='top', color='white', fontweight='bold')
        
        # 9. Cumulative error
        ax9 = plt.subplot(3, 4, 9)
        if self.backtest_results is not None:
            errors = np.abs(self.backtest_results['actual'] - self.backtest_results['predictions'])
            cumulative_error = np.cumsum(errors)
            plt.plot(cumulative_error)
            plt.title('Cumulative Absolute Error')
            plt.xlabel('Time')
            plt.ylabel('Cumulative Error')
            plt.grid(True, alpha=0.3)
        
        # 10. Directional accuracy over time
        ax10 = plt.subplot(3, 4, 10)
        if self.backtest_results is not None:
            pred_direction = np.sign(self.backtest_results['predictions'])
            actual_direction = np.sign(self.backtest_results['actual'])
            correct_direction = (pred_direction == actual_direction).astype(int)
            
            # Rolling accuracy
            window = 20
            rolling_accuracy = pd.Series(correct_direction).rolling(window).mean()
            plt.plot(rolling_accuracy)
            plt.axhline(y=0.5, color='r', linestyle='--', label='Random')
            plt.title(f'Rolling Directional Accuracy (window={window})')
            plt.xlabel('Time')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 11. Parameter correlation matrix
        ax11 = plt.subplot(3, 4, 11)
        if self.mcmc_samples is not None:
            key_params = ['target_mu', 'target_phi', 'target_sigma', 'target_lme_beta', 'target_jpx_beta']
            param_subset = self.mcmc_samples[key_params]
            corr_matrix = param_subset.corr()
            
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.2f')
            plt.title('Parameter Correlations')
        
        # 12. Model summary
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        
        if self.competition_metrics is not None:
            summary_text = f"""
            Model Summary
            ═══════════════
            
            Target: LME Zinc vs JPX Platinum
            Scenarios: 50,000
            
            Performance Metrics:
            • RMSE: {self.competition_metrics['rmse']:.6f}
            • Hit Rate: {self.competition_metrics['hit_rate']:.1%}
            • R²: {self.competition_metrics['r_squared']:.3f}
            • Info Coeff: {self.competition_metrics['information_coefficient']:.3f}
            
            Risk Metrics:
            • VaR 95%: {self.consensus_metrics['var_95']:.4f}
            • Coverage: {self.competition_metrics['interval_coverage']:.1%}
            
            Consensus Prediction:
            • Mean: {self.consensus_metrics['consensus_mean']:.6f}
            • Std: {self.consensus_metrics['consensus_std']:.6f}
            • P(positive): {self.consensus_metrics['prob_positive']:.1%}
            """
            
            ax12.text(0.1, 0.9, summary_text, transform=ax12.transAxes, 
                     fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('plots/mcmc_comprehensive_evaluation.png', dpi=300, bbox_inches='tight')
        print("📊 Comprehensive evaluation plots saved to plots/mcmc_comprehensive_evaluation.png")

def main():
    """Main evaluation pipeline."""
    print("🔍 MCMC Model Evaluation Pipeline")
    print("="*60)
    
    # Import the model from previous run
    from jpx_mcmc_model import JPXMCMCModel
    
    # Load the model (this will recreate it)
    model = JPXMCMCModel()
    model.load_data()
    model.estimate_initial_parameters()
    
    # For demo, use smaller MCMC for speed
    print("⚡ Running quick MCMC for evaluation...")
    model.mcmc_sampler(n_samples=2000, burn_in=500)
    model.simulate_scenarios(n_scenarios=10000)
    model.calculate_consensus_metrics()
    
    # Initialize evaluator
    evaluator = MCMCEvaluator()
    evaluator.load_model_results(model)
    
    # Run evaluations
    backtest_results = evaluator.backtest_model()
    competition_metrics = evaluator.evaluate_competition_metrics()
    evaluator.analyze_model_robustness()
    evaluator.create_evaluation_plots()
    
    print("\n" + "="*60)
    print("🎉 MCMC Evaluation Complete!")
    
    # Final summary
    print("\n📋 Final Model Assessment:")
    print(f"   • Model Type: Bayesian MCMC with Ornstein-Uhlenbeck process")
    print(f"   • Target: JPX Platinum vs LME Zinc spread")
    print(f"   • Validation RMSE: {competition_metrics['rmse']:.6f}")
    print(f"   • Directional Accuracy: {competition_metrics['hit_rate']:.1%}")
    print(f"   • Information Coefficient: {competition_metrics['information_coefficient']:.3f}")
    print(f"   • Prediction Interval Coverage: {competition_metrics['interval_coverage']:.1%}")
    
    return evaluator

if __name__ == "__main__":
    evaluator = main()
