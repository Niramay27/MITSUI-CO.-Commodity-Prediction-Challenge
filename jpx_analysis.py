"""
JPX Exchange Analysis for MCMC Simulation
Focus on Japan Exchange (JPX) futures data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_jpx_data():
    """Load and filter JPX-related data."""
    print("🇯🇵 Loading JPX Exchange Data Analysis")
    print("="*60)
    
    # Load datasets
    train = pd.read_csv('train.csv')
    train_labels = pd.read_csv('train_labels.csv')
    target_pairs = pd.read_csv('target_pairs.csv')
    
    # Identify JPX features
    jpx_features = [col for col in train.columns if col.startswith('JPX_')]
    print(f"📊 Found {len(jpx_features)} JPX features")
    
    return train, train_labels, target_pairs, jpx_features

def analyze_jpx_features(train, jpx_features):
    """Analyze JPX feature structure."""
    print("\n🔍 JPX Features Breakdown:")
    
    # Group by instrument type
    jpx_by_instrument = {}
    for feature in jpx_features:
        parts = feature.split('_')
        if len(parts) >= 3:
            instrument = '_'.join(parts[1:3])  # e.g., 'Gold_Mini', 'Platinum_Standard'
            if instrument not in jpx_by_instrument:
                jpx_by_instrument[instrument] = []
            jpx_by_instrument[instrument].append(feature)
    
    for instrument, features in jpx_by_instrument.items():
        print(f"   {instrument}: {len(features)} features")
        print(f"      {features[:3]}...")
    
    # Analyze missing data
    jpx_data = train[jpx_features]
    missing_analysis = jpx_data.isnull().sum().sort_values(ascending=False)
    print(f"\n📉 Missing Data Analysis:")
    print(f"   Features with missing data: {(missing_analysis > 0).sum()}")
    print(f"   Most complete features:")
    complete_features = missing_analysis[missing_analysis == 0]
    for feature in complete_features.head(10).index:
        print(f"      {feature}")
    
    return jpx_by_instrument, complete_features

def analyze_jpx_targets(target_pairs):
    """Analyze targets involving JPX."""
    print(f"\n🎯 JPX-Related Targets Analysis:")
    
    # Find targets with JPX in the pair column
    jpx_targets = target_pairs[
        target_pairs['pair'].str.contains('JPX_', na=False)
    ].copy()
    
    print(f"   Total JPX-related targets: {len(jpx_targets)}")
    
    # Parse pair column to extract features
    jpx_targets['feature_0'] = jpx_targets['pair'].apply(lambda x: x.split(' - ')[0] if ' - ' in x else x)
    jpx_targets['feature_1'] = jpx_targets['pair'].apply(lambda x: x.split(' - ')[1] if ' - ' in x else None)
    
    # Categorize target types
    jpx_target_types = {}
    for idx, row in jpx_targets.iterrows():
        if row['feature_1'] is None:
            # Single feature target
            target_type = 'JPX-Single'
        else:
            f0_is_jpx = 'JPX_' in str(row['feature_0'])
            f1_is_jpx = 'JPX_' in str(row['feature_1'])
            
            if f0_is_jpx and f1_is_jpx:
                target_type = 'JPX-JPX'
            elif f0_is_jpx:
                target_type = f"JPX-{row['feature_1'].split('_')[0]}"
            else:
                target_type = f"{row['feature_0'].split('_')[0]}-JPX"
        
        if target_type not in jpx_target_types:
            jpx_target_types[target_type] = []
        jpx_target_types[target_type].append(row['target'])
    
    print("   Target pair types:")
    for target_type, targets in jpx_target_types.items():
        print(f"      {target_type}: {len(targets)} targets")
    
    return jpx_targets, jpx_target_types

def select_mcmc_candidate(train, train_labels, jpx_features, jpx_targets, complete_features):
    """Select best candidate for MCMC simulation."""
    print(f"\n🎲 Selecting MCMC Candidate:")
    
    # Focus on JPX features with least missing data
    jpx_missing_data = train[jpx_features].isnull().sum().sort_values()
    print(f"   JPX features with least missing data:")
    for feature in jpx_missing_data.head(5).index:
        print(f"      {feature}: {jpx_missing_data[feature]} missing")
    
    # Find targets that use JPX features with minimal missing data
    best_jpx_features = jpx_missing_data.head(10).index.tolist()
    
    good_jpx_targets = []
    for idx, row in jpx_targets.iterrows():
        feature_0 = row['feature_0']
        feature_1 = row['feature_1']
        
        # Check if any of the features are in our best features list
        if (feature_0 in best_jpx_features) or (feature_1 and feature_1 in best_jpx_features):
            good_jpx_targets.append(row)
    
    print(f"   Viable JPX targets: {len(good_jpx_targets)}")
    
    if good_jpx_targets:
        # Select first viable target
        candidate = good_jpx_targets[0]
        target_name = candidate['target']
        feature_0 = candidate['feature_0']
        feature_1 = candidate['feature_1']
        lag = candidate['lag']
        
        print(f"\n✅ Selected MCMC Candidate:")
        print(f"   Target: {target_name}")
        print(f"   Feature 0: {feature_0}")
        print(f"   Feature 1: {feature_1}")
        print(f"   Lag: {lag}")
        
        # Analyze the target data
        target_data = train_labels[target_name].dropna()
        print(f"\n📈 Target Statistics:")
        print(f"   Non-null observations: {len(target_data)}")
        print(f"   Mean: {target_data.mean():.6f}")
        print(f"   Std: {target_data.std():.6f}")
        print(f"   Min: {target_data.min():.6f}")
        print(f"   Max: {target_data.max():.6f}")
        
        return candidate, target_data
    else:
        print("   ❌ No viable JPX targets found")
        return None, None

def analyze_feature_relationships(train, candidate):
    """Analyze relationships between features in the selected target."""
    if candidate is None:
        return None
    
    feature_0 = candidate['feature_0']
    feature_1 = candidate['feature_1']
    
    print(f"\n🔗 Feature Relationship Analysis:")
    print(f"   Analyzing: {feature_0} vs {feature_1}")
    
    # Get feature data
    data = train[[feature_0, feature_1, 'date_id']].dropna()
    
    if len(data) > 0:
        # Calculate correlation
        corr = data[feature_0].corr(data[feature_1])
        print(f"   Correlation: {corr:.4f}")
        
        # Calculate volatility
        data['f0_returns'] = data[feature_0].pct_change()
        data['f1_returns'] = data[feature_1].pct_change()
        
        f0_vol = data['f0_returns'].std()
        f1_vol = data['f1_returns'].std()
        
        print(f"   {feature_0} volatility: {f0_vol:.6f}")
        print(f"   {feature_1} volatility: {f1_vol:.6f}")
        
        return data
    else:
        print("   ❌ No overlapping data found")
        return None

def create_visualization(train, candidate, target_data):
    """Create visualization for the selected JPX target."""
    if candidate is None:
        return
    
    feature_0 = candidate['feature_0']
    feature_1 = candidate['feature_1']
    target_name = candidate['target']
    
    # Create plots directory
    Path('plots').mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'JPX MCMC Analysis: {target_name}', fontsize=16)
    
    # Plot 1: Target distribution
    axes[0,0].hist(target_data, bins=50, alpha=0.7, color='blue')
    axes[0,0].set_title(f'Target Distribution: {target_name}')
    axes[0,0].set_xlabel('Target Value')
    axes[0,0].set_ylabel('Frequency')
    
    # Plot 2: Target time series
    axes[0,1].plot(target_data.index, target_data.values)
    axes[0,1].set_title(f'Target Time Series: {target_name}')
    axes[0,1].set_xlabel('Date Index')
    axes[0,1].set_ylabel('Target Value')
    
    # Plot 3: Feature 0 time series
    f0_data = train[feature_0].dropna()
    axes[1,0].plot(f0_data.index, f0_data.values)
    axes[1,0].set_title(f'Feature 0: {feature_0}')
    axes[1,0].set_xlabel('Date Index')
    axes[1,0].set_ylabel('Value')
    
    # Plot 4: Feature 1 time series
    f1_data = train[feature_1].dropna()
    axes[1,1].plot(f1_data.index, f1_data.values)
    axes[1,1].set_title(f'Feature 1: {feature_1}')
    axes[1,1].set_xlabel('Date Index')
    axes[1,1].set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig('plots/jpx_mcmc_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved to plots/jpx_mcmc_analysis.png")

def main():
    # Load data
    train, train_labels, target_pairs, jpx_features = load_jpx_data()
    
    # Analyze JPX features
    jpx_by_instrument, complete_features = analyze_jpx_features(train, jpx_features)
    
    # Analyze JPX targets
    jpx_targets, jpx_target_types = analyze_jpx_targets(target_pairs)
    
    # Select MCMC candidate
    candidate, target_data = select_mcmc_candidate(
        train, train_labels, jpx_features, jpx_targets, complete_features
    )
    
    # Analyze feature relationships
    feature_data = analyze_feature_relationships(train, candidate)
    
    # Create visualization
    create_visualization(train, candidate, target_data)
    
    print("\n" + "="*60)
    print("✅ JPX Analysis Complete!")
    
    if candidate is not None:
        print("🚀 Ready for MCMC simulation...")
        return candidate, target_data, feature_data
    else:
        print("❌ No suitable candidate found for MCMC")
        return None, None, None

if __name__ == "__main__":
    candidate, target_data, feature_data = main()
