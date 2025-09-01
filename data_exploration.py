"""
MITSUI CO. Commodity Prediction Challenge - Data Exploration Script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load all competition data files."""
    print("Loading competition data...")
    
    data = {}
    
    # Load main datasets
    data['train'] = pd.read_csv('train.csv')
    data['train_labels'] = pd.read_csv('train_labels.csv')
    data['target_pairs'] = pd.read_csv('target_pairs.csv')
    data['test'] = pd.read_csv('test.csv')
    
    print(f"✅ Loaded {len(data)} datasets")
    for name, df in data.items():
        print(f"   {name}: {df.shape}")
    
    return data

def analyze_basic_structure(data):
    """Analyze basic data structure."""
    print("\n" + "="*50)
    print("BASIC DATA STRUCTURE ANALYSIS")
    print("="*50)
    
    # Training data analysis
    train = data['train']
    train_labels = data['train_labels']
    target_pairs = data['target_pairs']
    test = data['test']
    
    print(f"\n📊 Training Data:")
    print(f"   Shape: {train.shape}")
    print(f"   Date range: {train['date_id'].min()} to {train['date_id'].max()}")
    print(f"   Missing values: {train.isnull().sum().sum()}")
    
    print(f"\n🎯 Target Data:")
    print(f"   Shape: {train_labels.shape}")
    print(f"   Number of targets: {len([col for col in train_labels.columns if col.startswith('target_')])}")
    print(f"   Date range: {train_labels['date_id'].min()} to {train_labels['date_id'].max()}")
    
    print(f"\n🔗 Target Pairs:")
    print(f"   Total pairs: {len(target_pairs)}")
    print(f"   Lag values: {target_pairs['lag'].unique()}")
    
    print(f"\n🧪 Test Data:")
    print(f"   Shape: {test.shape}")
    print(f"   Date range: {test['date_id'].min()} to {test['date_id'].max()}")
    if 'is_scored' in test.columns:
        print(f"   Scored samples: {test['is_scored'].sum()}")

def analyze_features(data):
    """Analyze feature categories."""
    print("\n" + "="*50)
    print("FEATURE ANALYSIS")
    print("="*50)
    
    train = data['train']
    feature_cols = [col for col in train.columns if col != 'date_id']
    
    # Categorize features
    feature_categories = {
        'LME_Commodities': [col for col in feature_cols if col.startswith('LME_')],
        'JPX_Futures': [col for col in feature_cols if col.startswith('JPX_')],
        'US_Stocks': [col for col in feature_cols if col.startswith('US_Stock_')],
        'FX_Rates': [col for col in feature_cols if col.startswith('FX_')]
    }
    
    print(f"\n📈 Feature Categories:")
    for category, features in feature_categories.items():
        print(f"   {category}: {len(features)} features")
        if len(features) <= 10:
            print(f"     {features}")
        else:
            print(f"     {features[:3]} ... {features[-3:]}")
    
    # Missing data analysis
    print(f"\n❓ Missing Data Analysis:")
    missing_data = train.isnull().sum()
    missing_features = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if len(missing_features) > 0:
        print(f"   Features with missing data: {len(missing_features)}")
        print(f"   Most missing: {missing_features.head()}")
    else:
        print("   No missing data found!")

def analyze_targets(data):
    """Analyze target distribution and patterns."""
    print("\n" + "="*50)
    print("TARGET ANALYSIS")
    print("="*50)
    
    train_labels = data['train_labels']
    target_pairs = data['target_pairs']
    
    # Get target columns
    target_cols = [col for col in train_labels.columns if col.startswith('target_')]
    
    # Basic statistics
    target_data = train_labels[target_cols]
    
    print(f"\n📊 Target Statistics:")
    print(f"   Number of targets: {len(target_cols)}")
    print(f"   Missing values: {target_data.isnull().sum().sum()}")
    print(f"   Mean target value: {target_data.mean().mean():.6f}")
    print(f"   Std target value: {target_data.std().mean():.6f}")
    
    # Target pair types
    print(f"\n🔗 Target Pair Types:")
    pair_types = {}
    for _, row in target_pairs.iterrows():
        pair = row['pair']
        if ' - ' in pair:
            # It's a spread/difference
            instruments = pair.split(' - ')
            type1 = instruments[0].split('_')[0] if '_' in instruments[0] else instruments[0]
            type2 = instruments[1].split('_')[0] if '_' in instruments[1] else instruments[1]
            pair_type = f"{type1} - {type2}"
        else:
            # Single instrument
            pair_type = pair.split('_')[0] if '_' in pair else pair
        
        pair_types[pair_type] = pair_types.get(pair_type, 0) + 1
    
    print("   Most common pair types:")
    for pair_type, count in sorted(pair_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"     {pair_type}: {count}")

def generate_summary_plots(data):
    """Generate summary visualization plots."""
    print("\n" + "="*50)
    print("GENERATING SUMMARY PLOTS")
    print("="*50)
    
    # Create plots directory
    Path('plots').mkdir(exist_ok=True)
    
    train = data['train']
    train_labels = data['train_labels']
    
    # 1. Missing data heatmap
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Missing data pattern
    missing_data = train.isnull().sum()
    missing_features = missing_data[missing_data > 0]
    
    if len(missing_features) > 0:
        axes[0, 0].bar(range(len(missing_features)), missing_features.values)
        axes[0, 0].set_title('Missing Data by Feature')
        axes[0, 0].set_xlabel('Features (sorted by missing count)')
        axes[0, 0].set_ylabel('Missing Values')
        axes[0, 0].tick_params(axis='x', rotation=45)
    else:
        axes[0, 0].text(0.5, 0.5, 'No Missing Data', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Missing Data by Feature')
    
    # Plot 2: Target distribution
    target_cols = [col for col in train_labels.columns if col.startswith('target_')]
    sample_targets = train_labels[target_cols[:50]]  # Sample first 50 targets
    target_means = sample_targets.mean()
    
    axes[0, 1].hist(target_means, bins=20, alpha=0.7)
    axes[0, 1].set_title('Distribution of Target Means (First 50)')
    axes[0, 1].set_xlabel('Mean Target Value')
    axes[0, 1].set_ylabel('Frequency')
    
    # Plot 3: Data timeline
    axes[1, 0].plot(train['date_id'], train['LME_AH_Close'], alpha=0.7, label='LME Aluminum')
    axes[1, 0].plot(train['date_id'], train['LME_CA_Close'], alpha=0.7, label='LME Copper')
    axes[1, 0].set_title('Sample Commodity Prices Over Time')
    axes[1, 0].set_xlabel('Date ID')
    axes[1, 0].set_ylabel('Price')
    axes[1, 0].legend()
    
    # Plot 4: Target correlation sample
    sample_targets_corr = sample_targets.corr()
    im = axes[1, 1].imshow(sample_targets_corr, cmap='coolwarm', aspect='auto')
    axes[1, 1].set_title('Target Correlation Matrix (Sample)')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('plots/data_overview.png', dpi=300, bbox_inches='tight')
    print("✅ Saved overview plots to plots/data_overview.png")
    
    plt.show()

def main():
    """Main exploration function."""
    print("🔍 MITSUI CO. Commodity Prediction Challenge - Data Exploration")
    print("="*70)
    
    # Load data
    data = load_data()
    
    # Run analyses
    analyze_basic_structure(data)
    analyze_features(data)
    analyze_targets(data)
    
    # Generate plots
    generate_summary_plots(data)
    
    print("\n" + "="*70)
    print("✅ Data exploration completed!")
    print("\nKey Insights:")
    print("1. This is a financial time series prediction challenge")
    print("2. Predict 424 different target pairs/spreads")
    print("3. All targets use lag=1 (next day prediction)")
    print("4. Data includes commodities, futures, stocks, and FX rates")
    print("\nNext steps:")
    print("- Check plots/data_overview.png for visualizations")
    print("- Run feature engineering analysis")
    print("- Build baseline models")

if __name__ == "__main__":
    main()
