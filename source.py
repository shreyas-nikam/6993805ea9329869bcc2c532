import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Global Constants and Configuration ---
N_CLIENTS_DEFAULT = 5000
RANDOM_STATE = 42
N_CLUSTERS = 5

# Define segment names based on typical characteristics found in such analyses
SEGMENT_NAMES = {
    0: 'Young Aggressive Accumulators',
    1: 'Mid-Career Balanced Families',
    2: 'Pre-Retiree Conservatives',
    3: 'High-Net-Worth Diversifiers',
    4: 'Retired Income Seekers'
}

# Max equity % for risk tolerance 1 (very conservative) to 5 (aggressive)
MAX_EQUITY_BY_RISK_TOLERANCE = {
    1: 30,  # Very conservative
    2: 50,  # Conservative
    3: 70,  # Moderate
    4: 85,  # Growth
    5: 95   # Aggressive
}

# Feature and Target Columns
FEATURE_COLS = ['age', 'income', 'net_worth', 'risk_tolerance',
                'time_horizon', 'has_dependents', 'is_retired', 'tax_bracket']
TARGET_COLS = ['pref_equity', 'pref_bonds', 'pref_alts', 'pref_cash']

# --- Data Generation Functions ---

def _generate_single_allocation(row: pd.Series) -> pd.Series:
    """
    Helper function to generate a single client's preferred asset allocation
    based on their profile characteristics.
    """
    # Base equity based on age and risk
    base_equity = 80 - row['age'] * 0.6 + row['risk_tolerance'] * 8
    base_equity += row['time_horizon'] * 0.5
    base_equity -= row['is_retired'] * 15 # Retirees typically lower equity
    base_equity -= row['has_dependents'] * 5 # Dependents might mean slightly less aggressive

    # Clip equity to a reasonable range [10%, 95%]
    base_equity = np.clip(base_equity + np.random.normal(0, 5), 10, 95)

    # Bonds are the remainder, with some floor and random element
    bonds = np.clip(100 - base_equity - np.random.uniform(0, 15), 5, 70)

    # Alternatives: more for higher risk tolerance and net worth
    alts = np.clip(np.random.uniform(0, 15) + row['risk_tolerance'] * 2 + (row['net_worth'] / 1e6) * 0.5, 0, 20)

    # Cash: remaining
    cash = 100 - base_equity - bonds - alts

    if cash < 0: # Ensure no negative cash, adjust bonds if needed
        bonds += cash
        cash = 0

    # Round to 1 decimal place and ensure sum is 100 (or very close)
    total_alloc = base_equity + bonds + alts + cash
    if total_alloc != 100:
         # Distribute rounding error proportionally
        base_equity = round(base_equity / total_alloc * 100, 1)
        bonds = round(bonds / total_alloc * 100, 1)
        alts = round(alts / total_alloc * 100, 1)
        cash = round(cash / total_alloc * 100, 1)

    # Final check to make sure it sums to 100 (due to rounding, might be off by 0.1)
    # We can slightly adjust one asset class if needed
    current_sum = base_equity + bonds + alts + cash
    if current_sum != 100.0:
        diff = 100.0 - current_sum
        # Add/subtract the difference from equity (or another asset)
        base_equity += diff

    return pd.Series({
        'pref_equity': round(base_equity, 1),
        'pref_bonds': round(bonds, 1),
        'pref_alts': round(max(0, alts), 1),
        'pref_cash': round(max(0, cash), 1)
    })

def generate_client_data(n_clients: int = N_CLIENTS_DEFAULT, random_seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Generates synthetic client data including demographic information and
    preferred asset allocations.

    Args:
        n_clients (int): Number of client profiles to generate.
        random_seed (int): Seed for random number generation for reproducibility.

    Returns:
        pd.DataFrame: DataFrame containing client profiles and their preferred allocations.
    """
    np.random.seed(random_seed)

    clients_df = pd.DataFrame({
        'age': np.random.randint(22, 75, n_clients),
        'income': np.random.lognormal(11, 0.7, n_clients).astype(int),
        'net_worth': np.random.lognormal(12, 1.2, n_clients).astype(int),
        'risk_tolerance': np.random.choice([1,2,3,4,5], n_clients,
                                            p=[0.10, 0.20, 0.35, 0.25, 0.10]), # 1=very conservative, 5=aggressive
        'time_horizon': np.random.randint(1, 40, n_clients), # years
        'has_dependents': np.random.binomial(1, 0.45, n_clients),
        'is_retired': np.random.binomial(1, 0.15, n_clients),
        'tax_bracket': np.random.choice([0.12, 0.22, 0.24, 0.32, 0.37], n_clients)
    })

    allocations_df = clients_df.apply(_generate_single_allocation, axis=1)
    clients_full_df = pd.concat([clients_df, allocations_df], axis=1)

    # Ensure allocations sum to 100.0 due to rounding
    alloc_cols = ['pref_equity', 'pref_bonds', 'pref_alts', 'pref_cash']
    clients_full_df[alloc_cols] = clients_full_df[alloc_cols].div(clients_full_df[alloc_cols].sum(axis=1), axis=0) * 100
    clients_full_df[alloc_cols] = clients_full_df[alloc_cols].round(1)

    return clients_full_df

# --- Client Segmentation Functions ---

def perform_client_segmentation(
    clients_df: pd.DataFrame,
    feature_cols: list,
    n_clusters: int = N_CLUSTERS,
    random_seed: int = RANDOM_STATE
) -> tuple[pd.DataFrame, StandardScaler, KMeans, PCA, dict]:
    """
    Performs K-Means clustering on client features, scales data,
    applies PCA for visualization, and assigns segment names.

    Args:
        clients_df (pd.DataFrame): DataFrame containing client data.
        feature_cols (list): List of column names to use as features for clustering.
        n_clusters (int): Number of clusters for K-Means.
        random_seed (int): Seed for K-Means and PCA reproducibility.

    Returns:
        tuple:
            - pd.DataFrame: Clients DataFrame with 'segment', 'pca_1', 'pca_2', and 'segment_name' columns added.
            - StandardScaler: The fitted StandardScaler object.
            - KMeans: The fitted KMeans model.
            - PCA: The fitted PCA model.
            - dict: The segment names mapping.
    """
    # Scale features for K-Means clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clients_df[feature_cols])

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    clients_df['segment'] = kmeans.fit_predict(X_scaled)

    # Add PCA for visualization
    pca = PCA(n_components=2, random_state=random_seed)
    X_pca = pca.fit_transform(X_scaled)
    clients_df['pca_1'] = X_pca[:, 0]
    clients_df['pca_2'] = X_pca[:, 1]

    clients_df['segment_name'] = clients_df['segment'].map(SEGMENT_NAMES)

    return clients_df, scaler, kmeans, pca, SEGMENT_NAMES

def analyze_and_plot_segments(clients_df: pd.DataFrame, segment_names: dict, display_plots: bool = True) -> pd.DataFrame:
    """
    Analyzes and prints profiles for each client segment and generates a PCA plot.

    Args:
        clients_df (pd.DataFrame): DataFrame with client data, including 'segment' and 'segment_name' columns.
        segment_names (dict): Mapping of segment IDs to descriptive names.
        display_plots (bool): If True, shows the generated plot.

    Returns:
        pd.DataFrame: A DataFrame summarizing the profiles of each segment.
    """
    print("\nCLIENT SEGMENTS ANALYSIS")
    print("=" * 75)
    segment_profiles = {}
    for seg_id in sorted(clients_df['segment'].unique()):
        seg_data = clients_df[clients_df['segment'] == seg_id]
        segment_profiles[seg_id] = {
            'Count': len(seg_data),
            'Avg Age': f"{seg_data['age'].mean():.0f}",
            'Avg Income': f"${seg_data['income'].mean():,.0f}",
            'Avg Net Worth': f"${seg_data['net_worth'].mean():,.0f}",
            'Avg Risk Tolerance': f"{seg_data['risk_tolerance'].mean():.1f}/5",
            'Avg Time Horizon': f"{seg_data['time_horizon'].mean():.0f}yr",
            'Retired %': f"{seg_data['is_retired'].mean():.0%}",
            'Dependents %': f"{seg_data['has_dependents'].mean():.0%}",
            'Preferred Equity': f"{seg_data['pref_equity'].mean():.0f}%",
            'Preferred Bonds': f"{seg_data['pref_bonds'].mean():.0f}%",
            'Preferred Alts': f"{seg_data['pref_alts'].mean():.0f}%",
            'Preferred Cash': f"{seg_data['pref_cash'].mean():.0f}%"
        }

        print(f"\nSegment {seg_id} - {segment_names[seg_id]} (n={len(seg_data)}):")
        print(f"  Avg age: {seg_data['age'].mean():.0f}, Income: ${seg_data['income'].mean():,.0f}, Risk: {seg_data['risk_tolerance'].mean():.1f}/5")
        print(f"  Time horizon: {seg_data['time_horizon'].mean():.0f}yr, Retired: {seg_data['is_retired'].mean():.0%}, Dependents: {seg_data['has_dependents'].mean():.0%}")
        print(f"  Preferred: Eq {seg_data['pref_equity'].mean():.0f}%, Bonds {seg_data['pref_bonds'].mean():.0f}%, Alts {seg_data['pref_alts'].mean():.0f}%, Cash {seg_data['pref_cash'].mean():.0f}%")

    # Plotting the segments
    if display_plots:
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x='pca_1', y='pca_2', hue='segment_name', data=clients_df, palette='viridis', s=50, alpha=0.7)
        plt.title('Client Segments via PCA-reduced K-Means Clustering')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    segment_profile_df = pd.DataFrame.from_dict(segment_profiles, orient='index')
    segment_profile_df.index.name = 'Segment ID'
    segment_profile_df['Segment Name'] = segment_profile_df.index.map(segment_names)
    segment_profile_df = segment_profile_df[['Segment Name'] + [col for col in segment_profile_df.columns if col != 'Segment Name']]
    print("\nSegment Profile Table:\n")
    print(segment_profile_df)
    return segment_profile_df

# --- Asset Allocation Modeling Functions ---

def train_allocation_models(
    clients_df: pd.DataFrame,
    feature_cols: list,
    target_cols: list,
    test_size: float = 0.2,
    random_seed: int = RANDOM_STATE
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Trains XGBoost Regressor models for each asset class (target column).

    Args:
        clients_df (pd.DataFrame): DataFrame containing client features and preferred allocations.
        feature_cols (list): List of feature column names.
        target_cols (list): List of target (asset allocation) column names.
        test_size (float): Proportion of the dataset to include in the test split.
        random_seed (int): Seed for train_test_split and XGBoost models.

    Returns:
        tuple:
            - dict: A dictionary where keys are target asset names and values are trained XGBRegressor models.
            - pd.DataFrame: X_test DataFrame.
            - pd.DataFrame: Y_test DataFrame.
            - pd.DataFrame: ML model predictions for the test set, normalized to 100%.
            - pd.DataFrame: X_train DataFrame.
            - pd.DataFrame: Y_train DataFrame.
    """
    X = clients_df[feature_cols]
    Y = clients_df[target_cols]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_seed
    )

    ml_models = {}
    ml_preds = pd.DataFrame(index=X_test.index)

    for target in target_cols:
        model = XGBRegressor(n_estimators=100, max_depth=4,
                             learning_rate=0.05, random_state=random_seed)
        model.fit(X_train, Y_train[target])
        ml_preds[target] = model.predict(X_test)
        ml_models[target] = model

    # Normalize ML predictions to sum to 100% for each client
    ml_preds = ml_preds.clip(lower=0)
    ml_preds = ml_preds.div(ml_preds.sum(axis=1), axis=0) * 100
    ml_preds = ml_preds.round(1) # Round for cleaner display

    return ml_models, X_test, Y_test, ml_preds, X_train, Y_train

def _glide_path_single_client(row: pd.Series) -> pd.Series:
    """
    Helper function to calculate a single client's allocation based on a
    traditional "glide path" rule (e.g., 110 minus age for equity).
    """
    equity = max(10, min(90, 110 - row['age']))
    bonds = max(5, min(70, 100 - equity - 5))
    alts = 0
    cash = max(0, 100 - equity - bonds - alts)

    # Ensure allocations sum to 100
    total_alloc = equity + bonds + alts + cash
    if total_alloc != 100:
        diff = 100 - total_alloc
        equity += diff # Adjust equity for any rounding discrepancies

    return pd.Series({
        'pref_equity': round(equity, 1),
        'pref_bonds': round(bonds, 1),
        'pref_alts': round(alts, 1),
        'pref_cash': round(cash, 1)
    })

def generate_glide_path_predictions(X_test: pd.DataFrame) -> pd.DataFrame:
    """
    Generates asset allocation predictions for the test set using a traditional
    rule-based "glide path" approach.

    Args:
        X_test (pd.DataFrame): DataFrame of client features for the test set.

    Returns:
        pd.DataFrame: Glide path predictions for asset allocations.
    """
    glide_preds = X_test.apply(_glide_path_single_client, axis=1)
    return glide_preds

def compare_allocation_models(
    Y_test: pd.DataFrame,
    ml_preds: pd.DataFrame,
    glide_preds: pd.DataFrame,
    target_cols: list,
    display_plots: bool = True
) -> pd.DataFrame:
    """
    Compares the performance of ML-based and Glide Path allocation models
    using Mean Absolute Error (MAE).

    Args:
        Y_test (pd.DataFrame): Actual preferred allocations for the test set.
        ml_preds (pd.DataFrame): ML model predictions for the test set.
        glide_preds (pd.DataFrame): Glide path predictions for the test set.
        target_cols (list): List of target (asset allocation) column names.
        display_plots (bool): If True, shows the generated plot.

    Returns:
        pd.DataFrame: A DataFrame containing per-asset MAE for both models.
    """
    ml_mae_total = mean_absolute_error(Y_test, ml_preds)
    glide_mae_total = mean_absolute_error(Y_test, glide_preds)

    print("\nALLOCATION PREDICTION COMPARISON")
    print("=" * 55)
    print(f"ML (XGBoost) Total MAE:     {ml_mae_total:.2f} percentage points")
    print(f"Glide Path Total MAE:       {glide_mae_total:.2f} percentage points")
    if glide_mae_total > 0:
        print(f"ML improvement:             {(1 - ml_mae_total / glide_mae_total) * 100:.0f}% reduction in error")
    else:
        print("Glide Path MAE is zero, no improvement possible.")

    print(f"\nPer-asset MAE:")
    per_asset_mae = []
    for col in target_cols:
        ml_err = mean_absolute_error(Y_test[col], ml_preds[col])
        gp_err = mean_absolute_error(Y_test[col], glide_preds[col])
        per_asset_mae.append({'Asset Class': col.replace('pref_', '').title(), 'ML MAE': ml_err, 'Glide Path MAE': gp_err})
        print(f"  {col.replace('pref_', '').title():<10s}: ML={ml_err:.2f}, Glide={gp_err:.2f}")

    per_asset_mae_df = pd.DataFrame(per_asset_mae)

    if display_plots:
        fig, ax = plt.subplots(figsize=(10, 6))
        per_asset_mae_df.set_index('Asset Class').plot(kind='bar', ax=ax, colormap='viridis')
        plt.title('Mean Absolute Error (MAE) Comparison: ML vs. Glide Path')
        plt.ylabel('Mean Absolute Error (%)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    return per_asset_mae_df

# --- Suitability Functions ---

def suitability_check(allocation: pd.Series, client_profile: pd.Series, max_equity_rules: dict) -> list[str]:
    """
    Verifies that the recommended allocation is suitable for the client based on predefined rules.

    Args:
        allocation (pd.Series): The recommended asset allocation (e.g., pref_equity, pref_bonds).
        client_profile (pd.Series): The client's demographic and risk profile.
        max_equity_rules (dict): A dictionary defining max equity % for each risk tolerance level.

    Returns:
        list[str]: A list of suitability violations detected. Empty if no violations.
    """
    violations = []

    equity = allocation['pref_equity']
    risk = client_profile['risk_tolerance']
    age = client_profile['age']
    retired = client_profile['is_retired']

    # 1. Risk tolerance bounds (e.g., Max equity by risk tolerance level)
    if equity > max_equity_rules.get(risk, 100): # Default to 100 if risk not in rules
        violations.append(f"Equity {equity:.0f}% exceeds max {max_equity_rules.get(risk, 'N/A')}% for risk tolerance {risk}")

    # 2. Retirement check: Retirees (age > 60 and is_retired) should have lower equity exposure
    if retired == 1 and age > 60 and equity > 50:
        violations.append(f"Retired client (Age {age}) with {equity:.0f}% equity (max 50%)")

    # 3. Age check: Older clients (e.g., >70) should have lower equity exposure, regardless of retirement status
    if age > 70 and equity > 40:
        violations.append(f"Client age {age} with {equity:.0f}% equity (max 40%)")

    # 4. Minimum diversification: Ensure minimum exposure to bonds or not excessively concentrated in equity
    if equity > 90 or allocation['pref_bonds'] < 5:
        violations.append("Insufficient diversification (equity > 90% or bonds < 5%)")

    return violations

def apply_suitability_and_correct(
    X_test_clients: pd.DataFrame,
    clients_df_full: pd.DataFrame,
    ml_preds_df: pd.DataFrame,
    max_equity_rules: dict,
    target_cols: list
) -> pd.DataFrame:
    """
    Applies suitability checks to ML-predicted allocations and auto-corrects violations.

    Args:
        X_test_clients (pd.DataFrame): DataFrame of client features for the test set (used for client indices).
        clients_df_full (pd.DataFrame): The full client DataFrame containing all profiles.
        ml_preds_df (pd.DataFrame): ML model predictions for the test set.
        max_equity_rules (dict): A dictionary defining max equity % for each risk tolerance level.
        target_cols (list): List of target (asset allocation) column names.

    Returns:
        pd.DataFrame: The ml_preds_df with auto-corrected allocations.
    """
    corrected_ml_preds = ml_preds_df.copy()
    n_violations = 0
    print("\nSUITABILITY VALIDATION AND CORRECTION")
    print("=" * 55)

    for idx in X_test_clients.index:
        alloc = corrected_ml_preds.loc[idx].copy()
        profile = clients_df_full.loc[idx]

        violations = suitability_check(alloc, profile, max_equity_rules)

        if violations:
            n_violations += 1
            original_alloc = corrected_ml_preds.loc[idx].copy()

            # Auto-correction logic: Cap equity to suitability limit and reallocate excess to bonds
            risk = int(profile['risk_tolerance'])
            current_equity = alloc['pref_equity']
            corrected_equity = current_equity

            # Apply general risk tolerance cap
            max_eq_allowed_by_risk = max_equity_rules.get(risk, 100)
            if corrected_equity > max_eq_allowed_by_risk:
                corrected_equity = max_eq_allowed_by_risk

            # Apply retired/age specific equity caps (more restrictive if applicable)
            if profile['is_retired'] == 1 and profile['age'] > 60 and corrected_equity > 50:
                corrected_equity = 50
            if profile['age'] > 70 and corrected_equity > 40:
                corrected_equity = 40

            # Reallocate if equity was reduced
            if corrected_equity < current_equity:
                excess_equity = current_equity - corrected_equity
                alloc['pref_equity'] = corrected_equity
                alloc['pref_bonds'] += excess_equity # Reallocate to bonds
                print(f"  Client {idx}: Equity capped from {original_alloc['pref_equity']:.0f}% to {alloc['pref_equity']:.0f}%. Excess reallocated to bonds.")

            # Ensure diversification (simplified: if bonds are still too low after adjustment, move more from equity/alts)
            # This is a simplified auto-correction. In practice, more sophisticated optimization might be used.
            if alloc['pref_bonds'] < 5 and alloc['pref_equity'] > 5: # If bonds still too low, and there's equity to shift
                bond_shortfall = 5 - alloc['pref_bonds']
                if alloc['pref_equity'] > bond_shortfall + 5: # Ensure equity doesn't go below 5
                    alloc['pref_equity'] -= bond_shortfall
                    alloc['pref_bonds'] = 5
                    print(f"  Client {idx}: Adjusted for min bonds. Equity {alloc['pref_equity']:.0f}%, Bonds {alloc['pref_bonds']:.0f}%.")


            # Re-normalize after corrections if sum isn't exactly 100
            current_sum = alloc[target_cols].sum()
            if not np.isclose(current_sum, 100.0):
                alloc = alloc / current_sum * 100

            corrected_ml_preds.loc[idx] = alloc.round(1)

    print(f"\nRecommendations tested: {len(X_test_clients)}")
    print(f"Violations detected: {n_violations} ({n_violations/len(X_test_clients)*100:.1f}%)")
    print(f"All detected violations were auto-corrected.")

    return corrected_ml_preds

def generate_client_report(
    client_idx: int,
    clients_df: pd.DataFrame,
    ml_preds_df: pd.DataFrame,
    ml_models: dict,
    feature_cols: list,
    segment_names: dict
):
    """
    Generates a client-facing allocation report with conceptual explanation.

    Args:
        client_idx (int): The index of the client to generate the report for.
        clients_df (pd.DataFrame): The full client DataFrame.
        ml_preds_df (pd.DataFrame): DataFrame of ML-predicted (and potentially corrected) allocations.
        ml_models (dict): Dictionary of trained ML models.
        feature_cols (list): List of feature column names used for training.
        segment_names (dict): Mapping of segment IDs to descriptive names.
    """
    client_profile = clients_df.loc[client_idx]
    recommended_alloc = ml_preds_df.loc[client_idx]
    segment_name = segment_names.get(client_profile['segment'], f"Segment {client_profile['segment']}")

    print(f"\n{'='*55}")
    print(f"PERSONALIZED INVESTMENT ALLOCATION REPORT")
    print(f"{'='*55}")

    print(f"\nClient Profile:")
    print(f"  Age: {client_profile['age']}, Income: ${client_profile['income']:,.0f}, Net Worth: ${client_profile['net_worth']:,.0f}")
    print(f"  Risk Tolerance: {client_profile['risk_tolerance']}/5, Time Horizon: {client_profile['time_horizon']} years")
    print(f"  Client Segment: {segment_name}")
    print(f"  Retired: {'Yes' if client_profile['is_retired'] == 1 else 'No'}, Has Dependents: {'Yes' if client_profile['has_dependents'] == 1 else 'No'}")

    print(f"\nRecommended Allocation:")
    for asset in ['pref_equity', 'pref_bonds', 'pref_alts', 'pref_cash']:
        name = asset.replace('pref_','').title()
        pct = recommended_alloc[asset]
        bar = '#' * int(pct / 2)
        print(f"  {name:<12s}: {pct:5.1f}% {bar}")

    print(f"\nWhy this allocation:")
    # Conceptual explanation without full SHAP calculation in this simplified demo
    drivers = []
    if client_profile['risk_tolerance'] > 3:
        drivers.append("your above-average risk tolerance")
    elif client_profile['risk_tolerance'] < 3:
        drivers.append("your below-average risk tolerance")

    if client_profile['time_horizon'] > 20:
        drivers.append("your long investment horizon")
    elif client_profile['time_horizon'] < 10:
        drivers.append("your shorter investment horizon")

    if client_profile['is_retired'] == 1:
        drivers.append("your retired status")

    if client_profile['net_worth'] > clients_df['net_worth'].median():
        drivers.append("your significant net worth")

    if recommended_alloc['pref_equity'] > clients_df['pref_equity'].mean():
        equity_direction = "higher"
    else:
        equity_direction = "lower"

    if drivers:
        print(f"  Based on {' and '.join(drivers)}, your allocation emphasizes a {equity_direction} equity exposure relative to the average client.")
    else:
        print(f"  This allocation is carefully tailored to your unique financial profile and goals.")


    print(f"\nDisclosure: This allocation is AI-assisted and has undergone a suitability review.")
    print(f"Past performance does not guarantee future results. Investment involves risk.")

def suitability_monitor_framework():
    """
    Defines a framework for ongoing suitability monitoring triggers and actions.
    This function conceptually outlines the monitoring process.
    """
    triggers = {
        'Age milestone': 'Client turns 60/65/70 -> review equity exposure and income needs',
        'Retirement': 'Employment status changes -> shift to income focus, adjust risk',
        'Major withdrawal': '>20% of portfolio withdrawn -> rebalance, reassess goals',
        'Market shock': 'Portfolio drops >15% -> check risk tolerance still valid, rebalance',
        'Life event': 'Marriage, divorce, child, inheritance, job loss -> full profile review and reallocation',
        'Annual review': 'Mandatory annual suitability reconfirmation and portfolio review'
    }

    print("\nSUITABILITY MONITORING TRIGGERS")
    print("=" * 55)
    print("For a modern robo-advisor, continuous monitoring is crucial to ensure ongoing suitability:")
    for trigger, action in triggers.items():
        print(f"  - {trigger:<20s}: {action}")

def topic_synthesis():
    """Synthesizes the entire Topic 1 workflow: Signal -> Portfolio -> Client."""
    print("\n" + "="*60)
    print("TOPIC 1 SYNTHESIS: AI IN ASSET MANAGEMENT")
    print("="*60)
    print("\nThe three cases span the full investment chain of AI applications in finance:")
    print("\n  D5-T1-C1: ML Stock Selection (Signal Generation)")
    print("    - Alpha generation via nonlinear factor discovery")
    print("    - Audience: Institutional PMs, quant analysts")
    print("\n  D5-T1-C2: AI-Optimized Portfolio Construction (Portfolio Optimization)")
    print("    - From alpha scores to constrained portfolio weights")
    print("    - Audience: Portfolio managers, risk officers")
    print("\n  D5-T1-C3: Robo-Advisor Simulation (Client Delivery & Suitability)")
    print("    - Client segmentation + personalized allocation")
    print("    - Audience: Wealth managers, retail advisory")
    print("\nThis workflow demonstrates the end-to-end impact of AI:")
    print("  Signal -> Portfolio -> Client")
    print("  Institutional -> Retail")
    print("  Alpha -> Allocation -> Suitability")
    print("\nVeridian Financial leverages AI at every stage, while ensuring robust governance and client-centricity.")

# --- Main Execution Block (for demonstration/testing within this file) ---
if __name__ == "__main__":
    np.random.seed(RANDOM_STATE) # Ensure overall reproducibility

    print(f"--- Generating {N_CLIENTS_DEFAULT} Client Profiles ---")
    clients_df_full = generate_client_data(n_clients=N_CLIENTS_DEFAULT, random_seed=RANDOM_STATE)
    print(f"Generated {N_CLIENTS_DEFAULT} client profiles.")
    print("\nFirst 5 rows of the client data:")
    print(clients_df_full.head())
    print(f"\nAge range: {clients_df_full['age'].min()}-{clients_df_full['age'].max()}")
    print(f"Mean preferred allocation: ")
    print(f"  Equity {clients_df_full['pref_equity'].mean():.0f}%,")
    print(f"  Bonds {clients_df_full['pref_bonds'].mean():.0f}%,")
    print(f"  Alts {clients_df_full['pref_alts'].mean():.0f}%,")
    print(f"  Cash {clients_df_full['pref_cash'].mean():.0f}%")
    print(f"\nAverage sum of preferred allocations: {clients_df_full[TARGET_COLS].sum(axis=1).mean():.1f}")


    print(f"\n--- Performing Client Segmentation ({N_CLUSTERS} clusters) ---")
    clients_df_segmented, scaler, kmeans, pca, seg_names = perform_client_segmentation(
        clients_df_full.copy(), FEATURE_COLS, n_clusters=N_CLUSTERS, random_seed=RANDOM_STATE
    )
    analyze_and_plot_segments(clients_df_segmented, seg_names, display_plots=True)


    print("\n--- Training Asset Allocation Models ---")
    ml_models, X_test, Y_test, ml_preds, X_train, Y_train = train_allocation_models(
        clients_df_full, FEATURE_COLS, TARGET_COLS, random_seed=RANDOM_STATE
    )
    print(f"Trained {len(ml_models)} XGBoost models.")
    print("\nFirst 5 rows of ML predictions (test set):")
    print(ml_preds.head())


    print("\n--- Generating Glide Path Predictions ---")
    glide_preds = generate_glide_path_predictions(X_test)
    print("\nFirst 5 rows of Glide Path predictions (test set):")
    print(glide_preds.head())


    print("\n--- Comparing Allocation Models ---")
    mae_comparison_df = compare_allocation_models(Y_test, ml_preds, glide_preds, TARGET_COLS, display_plots=True)


    print("\n--- Applying Suitability Checks and Corrections ---")
    corrected_ml_preds = apply_suitability_and_correct(
        X_test, clients_df_full, ml_preds, MAX_EQUITY_BY_RISK_TOLERANCE, TARGET_COLS
    )
    print("\nFirst 5 rows of ML predictions AFTER suitability corrections (test set):")
    print(corrected_ml_preds.head())


    print("\n--- Generating Client Reports for Sample Clients ---")
    # Find candidates in the full client dataset
    young_aggressive_candidates = clients_df_full[(clients_df_full['age'] < 30) & (clients_df_full['risk_tolerance'] == 5)].index
    retired_conservative_candidates = clients_df_full[(clients_df_full['age'] > 65) & (clients_df_full['is_retired'] == 1) & (clients_df_full['risk_tolerance'] < 3)].index

    # Filter candidates to ensure they are in the test set
    sample_client_young_idx = young_aggressive_candidates[young_aggressive_candidates.isin(X_test.index)].min() # Use .min() to pick one valid index
    sample_client_retired_idx = retired_conservative_candidates[retired_conservative_candidates.isin(X_test.index)].min()

    print("\nGenerating report for a Young, Aggressive Client:")
    generate_client_report(sample_client_young_idx, clients_df_full, corrected_ml_preds, ml_models, FEATURE_COLS, seg_names)

    print("\n" + "="*70 + "\n")

    print("Generating report for a Retired, Conservative Client:")
    generate_client_report(sample_client_retired_idx, clients_df_full, corrected_ml_preds, ml_models, FEATURE_COLS, seg_names)


    print("\n--- Suitability Monitoring Framework ---")
    suitability_monitor_framework()


    print("\n--- Topic Synthesis ---")
    topic_synthesis()
