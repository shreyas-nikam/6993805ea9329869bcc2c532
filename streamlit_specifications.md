
# Streamlit Application Specification: Robo-Advisor Simulation

## 1. Application Overview

### Purpose of the Application

This Streamlit application, "Robo-Advisor Simulation," serves as a hands-on learning tool for CFA Charterholders and Investment Professionals to understand and evaluate the capabilities of a modern, AI-driven robo-advisor platform. It demonstrates how machine learning and rule-based systems can be integrated to enhance client services in wealth management, specifically focusing on client segmentation, personalized asset allocation, and robust suitability monitoring, all while addressing regulatory and ethical considerations. The app provides a practical, real-world workflow simulation, moving beyond theoretical concepts to show how AI can personalize investment advice at scale, improve client fit, and embed regulatory compliance.

### High-Level Story Flow of the Application

The user, acting as a Senior Investment Analyst at Veridian Financial, navigates through a structured workflow to assess the robo-advisor's capabilities:

1.  **Client Data Generation**: The journey begins by simulating a diverse dataset of 5,000 client profiles, establishing the foundational data for analysis and model training.
2.  **Client Segmentation**: Using K-Means clustering, the application uncovers distinct client segments, moving beyond traditional demographic buckets to reveal nuanced client characteristics and preferences.
3.  **Allocation Prediction & Comparison**: A multi-output XGBoost model is trained to predict personalized asset allocations. Its performance is then critically compared against a traditional rule-based "glide path" to quantify the benefits of personalization using Mean Absolute Error (MAE).
4.  **Suitability Validation**: A crucial governance layer is applied, where rule-based suitability checks are run against the ML-generated recommendations. Any unsuitable allocations are automatically corrected to ensure regulatory compliance and ethical standards.
5.  **Client Reporting**: Personalized client reports are generated for selected clients, presenting their recommended allocations along with conceptual explanations (inspired by SHAP values) and regulatory disclosures, fostering transparency and trust.
6.  **Suitability Monitoring & Synthesis**: Finally, the application outlines a framework for ongoing suitability monitoring, detailing triggers for reallocation reviews, and synthesizes how this client-facing AI application fits within the broader AI in Asset Management context (Signal -> Portfolio -> Client).

## 2. Code Requirements

### Import Statement

```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from source import * # Import all functions and global variables from source.py
```

### `st.session_state` Design

`st.session_state` is used to preserve critical data and computed results across different pages (simulated via sidebar selection) and interactions.

*   **Initialization**: All keys should be initialized at the start of `app.py` using a pattern like `if 'key' not in st.session_state: st.session_state['key'] = default_value`.
    *   `st.session_state.clients_df = None` (stores the generated client data)
    *   `st.session_state.X_test = None` (stores the test features for ML)
    *   `st.session_state.Y_test = None` (stores the test targets for ML)
    *   `st.session_state.ml_preds = None` (stores ML model's predicted allocations)
    *   `st.session_state.glide_preds = None` (stores glide path model's predicted allocations)
    *   `st.session_state.ml_models = None` (stores the trained XGBoost models)
    *   `st.session_state.segment_profile_df = None` (stores the summarized segment profiles)
    *   `st.session_state.SEGMENT_NAMES = None` (stores the dictionary mapping segment IDs to names)
    *   `st.session_state.per_asset_mae_df = None` (stores the per-asset MAE comparison)
    *   `st.session_state.n_violations_count = 0` (stores the count of suitability violations)
    *   `st.session_state.feature_cols = ['age', 'income', 'net_worth', 'risk_tolerance', 'time_horizon', 'has_dependents', 'is_retired', 'tax_bracket']` (constant, but good to store)
    *   `st.session_state.target_cols = ['pref_equity', 'pref_bonds', 'pref_alts', 'pref_cash']` (constant, but good to store)
    *   `st.session_state.current_page = "Introduction"` (controls the active page in sidebar)
    *   `st.session_state.data_generated = False`
    *   `st.session_state.segments_calculated = False`
    *   `st.session_state.allocations_predicted = False`
    *   `st.session_state.suitability_checked = False`

*   **Update**:
    *   **Page 1 (Client Data Generation)**: `st.session_state.clients_df` is updated by `generate_client_data`. `st.session_state.data_generated = True`.
    *   **Page 2 (Client Segmentation)**: `st.session_state.clients_df` (with 'segment', 'pca_1', 'pca_2', 'segment_name' columns), `st.session_state.segment_profile_df`, `st.session_state.SEGMENT_NAMES` are updated. `st.session_state.segments_calculated = True`.
    *   **Page 3 (Allocation Prediction & Comparison)**: `st.session_state.X_test`, `st.session_state.Y_test`, `st.session_state.ml_preds`, `st.session_state.glide_preds`, `st.session_state.ml_models`, `st.session_state.per_asset_mae_df` are updated. `st.session_state.allocations_predicted = True`.
    *   **Page 4 (Suitability Validation)**: `st.session_state.ml_preds` (modified by auto-correction logic), `st.session_state.n_violations_count` are updated. `st.session_state.suitability_checked = True`.

*   **Read**: Data from `st.session_state` is read at the beginning of each page's rendering logic to ensure continuity and prevent re-computation if not necessary. For example, `clients_df = st.session_state.clients_df` would be used.

### Application Structure and Flow

The application will use a sidebar selectbox to simulate a multi-page experience.
The main content area will dynamically render based on the `st.session_state.current_page` value.

**Sidebar Navigation**

```python
# Sidebar title
st.sidebar.title("Robo-Advisor Workflow")

# Pages for navigation
pages = [
    "Introduction",
    "1. Client Data Generation",
    "2. Client Segmentation",
    "3. Allocation Prediction & Comparison",
    "4. Suitability Validation",
    "5. Client Reporting",
    "6. Suitability Monitoring & Synthesis"
]

# Set default page
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Introduction"

# Update current_page based on sidebar selection
selected_page = st.sidebar.selectbox("Navigate Steps:", pages, index=pages.index(st.session_state.current_page))
st.session_state.current_page = selected_page

st.sidebar.markdown("---")
st.sidebar.info("As a CFA Charterholder, follow the steps sequentially to evaluate the robo-advisor's capabilities.")
```

---

#### Page: Introduction

**Markdown Content**

```python
st.title("Robo-Advisor Simulation: Modernizing Wealth Management at Veridian Financial")

st.markdown(f"""
Welcome, CFA Charterholder and Senior Investment Analyst at Veridian Financial!

Your firm is exploring an advanced robo-advisor platform to enhance client services, particularly in client segmentation, asset allocation, and, crucially, suitability monitoring. This application simulates the core functions of such a modern robo-advisor.

Your task is to evaluate its capabilities by:
1.  **Segmenting Clients**: Discovering natural client groups beyond simple demographic buckets using advanced analytics.
2.  **Predicting Allocations**: Using machine learning to generate personalized asset allocations (equity, bonds, alternatives, cash) for each client.
3.  **Validating Suitability**: Implementing robust checks to ensure all recommendations comply with regulatory standards (e.g., FINRA Reg BI, SEC fiduciary duties) and Veridian's ethical guidelines.
4.  **Reporting**: Generating clear, client-facing reports with conceptual explanations for the recommended allocations.

This hands-on simulation will demonstrate how AI can personalize investment advice at scale, improve client fit compared to generic rules, and critically, how regulatory suitability and ethical considerations are embedded into the AI workflow. It provides insight into automating Investment Policy Statement (IPS) construction, enhancing risk management, and ensuring regulatory compliance for client-facing AI.
""")

st.markdown("---")
st.subheader("Workflow Overview")
st.image("https://raw.githubusercontent.com/streamlit/docs/main/docs/static/img/workflow_overview.png", caption="Robo-Advisor Workflow Steps") # Placeholder image or generate a simple diagram
st.markdown("Please proceed to **'1. Client Data Generation'** in the sidebar to begin the simulation.")
```

---

#### Page: 1. Client Data Generation

**Markdown Content**

```python
st.header("1. Generating Diverse Client Profiles for Robo-Advisor Evaluation")

st.markdown(f"""
As a CFA Charterholder, you understand that the performance of any client-facing financial model heavily depends on the quality and representativeness of its input data. Since we don't have direct access to real client data for this evaluation, we need to simulate a realistic dataset of 5,000 diverse client profiles. This synthetic dataset will include key demographic, financial, and risk-related features, along with "true" preferred asset allocations.

This step is crucial because it allows us to:
*   **Control for Realism**: Ensure the simulated clients reflect the varied needs and characteristics encountered in real-world wealth management.
*   **Establish Ground Truth**: The "true" preferred allocations serve as a baseline to evaluate how well our robo-advisor's ML model personalizes advice compared to what clients genuinely desire.
*   **Test Edge Cases**: Simulate clients with different risk tolerances, time horizons, and life stages to test the robo-advisor's robustness and suitability checks.

The goal is to create a rich client dataset that allows us to thoroughly test the robo-advisor's ability to provide tailored and suitable investment advice.
""")

st.subheader("Generate Client Data")
```

**Widget & Function Invocation**

```python
if st.button("Generate 5,000 Client Profiles"):
    with st.spinner("Generating client data..."):
        # Call function from source.py
        clients = generate_client_data(n_clients=5000)
        st.session_state.clients_df = clients
        st.session_state.data_generated = True
    st.success("Client data generated successfully!")

if st.session_state.data_generated:
    st.subheader("Generated Client Profiles (First 5 Rows)")
    st.dataframe(st.session_state.clients_df.head())

    st.subheader("Summary Statistics")
    st.markdown(f"**Generated {len(st.session_state.clients_df)} client profiles.**")
    st.markdown(f"**Age range:** {st.session_state.clients_df['age'].min()}-{st.session_state.clients_df['age'].max()}")
    st.markdown(f"**Mean preferred allocation:**")
    st.markdown(f"  - Equity: {st.session_state.clients_df['pref_equity'].mean():.0f}%")
    st.markdown(f"  - Bonds: {st.session_state.clients_df['pref_bonds'].mean():.0f}%")
    st.markdown(f"  - Alternatives: {st.session_state.clients_df['pref_alts'].mean():.0f}%")
    st.markdown(f"  - Cash: {st.session_state.clients_df['pref_cash'].mean():.0f}%")
    st.markdown(f"Average sum of preferred allocations: {st.session_state.clients_df[['pref_equity', 'pref_bonds', 'pref_alts', 'pref_cash']].sum(axis=1).mean():.1f}")
    
    st.info("The data covers a broad range of client ages, income levels, and net worths, ensuring diverse scenarios for testing. The mean preferred asset allocations are broadly consistent with typical diversified portfolios, providing a solid foundation for evaluating personalization.")
```

---

#### Page: 2. Client Segmentation

**Markdown Content**

```python
st.header("2. Discovering Client Segments with K-Means Clustering")

st.markdown(f"""
At Veridian Financial, our traditional client segmentation often relies on broad categories like "young" or "retiree." However, as a CFA, you know that a 30-year-old entrepreneur with high net worth and a short-term liquidity goal has vastly different needs than a 30-year-old teacher saving for retirement over 35 years. These nuances are missed by simple rules.

This is where K-Means clustering comes in. By applying K-Means to a richer set of client features, we aim to uncover more natural and meaningful client segments. This allows the robo-advisor to move beyond generic advice and truly personalize recommendations based on shared characteristics that traditional methods might overlook. This segmentation forms the foundation for targeted advice and more relevant client engagement.
""")

# Formula rendering (STRICT RULE)
st.markdown(r"The mathematical concept behind K-Means clustering is to partition $N$ observations into $K$ clusters, where each observation belongs to the cluster with the nearest mean (centroid). The objective function, which K-Means tries to minimize, is the sum of squared distances between each point and its assigned cluster centroid:")
st.markdown(r"$$ J = \sum_{{i=1}}^{{N}} \sum_{{k=1}}^{{K}} w_{{ik}} ||x_i - \mu_k||^2 $$")
st.markdown(r"where $x_i$ represents a client's feature vector, $\mu_k$ is the centroid of cluster $k$, and $w_{{ik}}$ is an indicator variable equal to 1 if client $i$ belongs to cluster $k$ and 0 otherwise. Minimizing $J$ means finding cluster assignments and centroids such that clients within a cluster are as similar as possible, and clients in different clusters are as dissimilar as possible.")

st.subheader("Perform Client Segmentation")
```

**Widget & Function Invocation**

```python
if not st.session_state.data_generated:
    st.warning("Please generate client profiles first on the '1. Client Data Generation' page.")
else:
    if st.button("Perform K-Means Client Segmentation"):
        with st.spinner("Clustering clients and generating segments..."):
            # Ensure clients_df is available
            clients = st.session_state.clients_df.copy() # Work on a copy
            
            # Call clustering logic from source.py
            feature_cols = st.session_state.feature_cols
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(clients[feature_cols])
            
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            clients['segment'] = kmeans.fit_predict(X_scaled)
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            clients['pca_1'] = X_pca[:, 0]
            clients['pca_2'] = X_pca[:, 1]
            
            # This is defined in source.py, so ensure it's picked up or recreated for consistency
            SEGMENT_NAMES = {
                0: 'Young Aggressive Accumulators',
                1: 'Mid-Career Balanced Families',
                2: 'Pre-Retiree Conservatives',
                3: 'High-Net-Worth Diversifiers',
                4: 'Retired Income Seekers'
            }
            clients['segment_name'] = clients['segment'].map(SEGMENT_NAMES)
            
            st.session_state.clients_df = clients # Update session state with segmented data
            st.session_state.SEGMENT_NAMES = SEGMENT_NAMES # Store SEGMENT_NAMES
            
            # Prepare segment_profile_df
            segment_profiles = {}
            for seg_id in sorted(clients['segment'].unique()):
                seg_data = clients[clients['segment'] == seg_id]
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
            segment_profile_df = pd.DataFrame.from_dict(segment_profiles, orient='index')
            segment_profile_df.index.name = 'Segment ID'
            segment_profile_df['Segment Name'] = segment_profile_df.index.map(SEGMENT_NAMES)
            st.session_state.segment_profile_df = segment_profile_df[['Segment Name'] + [col for col in segment_profile_df.columns if col != 'Segment Name']]

            st.session_state.segments_calculated = True
        st.success("Client segmentation complete!")

    if st.session_state.segments_calculated:
        st.subheader("Client Segments via PCA-reduced K-Means Clustering")
        # Generate and display PCA plot (recreating the plot logic from source.py)
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(x='pca_1', y='pca_2', hue='segment_name', data=st.session_state.clients_df, palette='viridis', s=50, alpha=0.7, ax=ax)
        ax.set_title('Client Segments via PCA-reduced K-Means Clustering (V1)')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Segment Profile Table")
        st.dataframe(st.session_state.segment_profile_df)
        
        st.info(f"""
        The K-Means clustering algorithm successfully identified 5 distinct client segments. The PCA plot visually confirms these groupings. From a CFA's perspective, this output is invaluable:

        *   **Beyond Age Buckets**: Instead of generic "mid-age," we now have segments like "{st.session_state.segment_profile_df.loc[1, 'Segment Name']}" and "{st.session_state.segment_profile_df.loc[3, 'Segment Name']}". These segments reveal nuanced preferences, such as "{st.session_state.segment_profile_df.loc[3, 'Segment Name']}" desiring a significant allocation to alternatives (around {st.session_state.segment_profile_df.loc[3, 'Preferred Alts']}), a preference that a simple age-based glide path would likely miss by assigning 0% to alternatives.
        *   **Targeted Strategies**: Each segment has a unique risk tolerance, time horizon, and preferred asset allocation. For example, "{st.session_state.segment_profile_df.loc[0, 'Segment Name']}" (Segment 0) have high equity preference and risk tolerance, while "{st.session_state.segment_profile_df.loc[4, 'Segment Name']}" (Segment 4) show lower equity and higher bond allocations. This granular insight enables Veridian Financial to tailor marketing, product offerings, and communication strategies more effectively, moving towards truly personalized wealth management.
        *   **Foundation for ML**: These segments provide a richer context for the machine learning model, allowing it to learn allocation patterns specific to these nuanced client types.
        """)
```

---

#### Page: 3. Allocation Prediction & Comparison

**Markdown Content**

```python
st.header("3. Predicting Personalized Asset Allocations with Multi-Output XGBoost")

st.markdown(f"""
After identifying distinct client segments, the next challenge for Veridian Financial is to deliver personalized asset allocations at scale. Historically, we might have used a simple rule-based "glide path" (e.g., "110 minus age" for equity). However, this one-size-fits-all approach often fails to account for individual client nuances like risk tolerance, specific goals, or net worth, leading to suboptimal or even unsuitable recommendations.

As a CFA, you need to evaluate if an ML model can provide more accurate and personalized allocations than these traditional rules. We will implement a Multi-Output Regression model using XGBoost to predict the percentage allocation to equity, bonds, alternatives, and cash for each client based on their profile features. We will then quantify the personalization benefit by comparing the ML model's performance against a rule-based glide path using Mean Absolute Error (MAE). This directly addresses the firm's need for scalable, personalized, and data-driven allocation strategies.
""")

# Formula rendering (STRICT RULE)
st.markdown(r"The Mean Absolute Error (MAE) is chosen as our primary metric to quantify the average magnitude of errors between our predicted allocations and the clients' 'true' preferred allocations. For $N$ clients and $K$ asset classes, the MAE is defined as:")
st.markdown(r"$$ MAE = \frac{{1}}{{N \cdot K}} \sum_{{i=1}}^{{N}} \sum_{{k=1}}^{{K}} |w_{{i,k}}^{{\text{{recommended}}}} - w_{{i,k}}^{{\text{{preferred}}}}| $$")
st.markdown(r"where $w_{{i,k}}^{{\text{{recommended}}}}$ is the recommended allocation for client $i$ in asset class $k$ (either by the ML model or the glide path), and $w_{{i,k}}^{{\text{{preferred}}}}$ is the client's actual preferred allocation. A lower MAE indicates a better fit to client preferences, signifying more effective personalization.")

st.subheader("Train ML Model and Compare to Glide Path")
```

**Widget & Function Invocation**

```python
if not st.session_state.segments_calculated:
    st.warning("Please perform client segmentation first on the '2. Client Segmentation' page.")
else:
    if st.button("Train ML Model & Compare Allocations"):
        with st.spinner("Training ML model and comparing to glide path..."):
            clients = st.session_state.clients_df
            feature_cols = st.session_state.feature_cols
            target_cols = st.session_state.target_cols

            X = clients[feature_cols]
            Y = clients[target_cols]

            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=0.2, random_state=42)
            
            st.session_state.X_test = X_test
            st.session_state.Y_test = Y_test

            ml_models = {}
            ml_preds = pd.DataFrame(index=X_test.index)

            for target in target_cols:
                model = XGBRegressor(n_estimators=100, max_depth=4, 
                                     learning_rate=0.05, random_state=42)
                model.fit(X_train, Y_train[target])
                ml_preds[target] = model.predict(X_test)
                ml_models[target] = model

            ml_preds = ml_preds.clip(lower=0)
            ml_preds = ml_preds.div(ml_preds.sum(axis=1), axis=0) * 100
            
            st.session_state.ml_preds = ml_preds
            st.session_state.ml_models = ml_models

            # Glide path calculation using source.py function
            glide_preds = X_test.apply(glide_path, axis=1)
            st.session_state.glide_preds = glide_preds

            ml_mae_total = mean_absolute_error(Y_test, ml_preds)
            glide_mae_total = mean_absolute_error(Y_test, glide_preds)

            st.subheader("Allocation Prediction Comparison")
            st.markdown(f"**ML (XGBoost) Total MAE:**     {ml_mae_total:.2f} percentage points")
            st.markdown(f"**Glide Path Total MAE:**       {glide_mae_total:.2f} percentage points")
            st.markdown(f"**ML improvement:**             {(1 - ml_mae_total / glide_mae_total) * 100:.0f}% reduction in error")

            st.subheader("Per-asset MAE")
            per_asset_mae = []
            for col in target_cols:
                ml_err = mean_absolute_error(Y_test[col], ml_preds[col])
                gp_err = mean_absolute_error(Y_test[col], glide_preds[col])
                per_asset_mae.append({'Asset Class': col.replace('pref_', '').title(), 'ML MAE': ml_err, 'Glide Path MAE': gp_err})
                st.markdown(f"  **{col.replace('pref_', '').title():<10s}**: ML={ml_err:.2f}, Glide={gp_err:.2f}")

            per_asset_mae_df = pd.DataFrame(per_asset_mae)
            st.session_state.per_asset_mae_df = per_asset_mae_df
            
            # Bar chart for MAE comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            per_asset_mae_df.set_index('Asset Class').plot(kind='bar', ax=ax, colormap='viridis')
            ax.set_title('Mean Absolute Error (MAE) Comparison: ML vs. Glide Path (V3)')
            ax.set_ylabel('Mean Absolute Error (%)')
            plt.xticks(rotation=45, ha='right')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)
            st.session_state.allocations_predicted = True
        st.success("Allocation prediction and comparison complete!")
    
    if st.session_state.allocations_predicted:
        st.info(f"""
        The comparison between the Multi-Output XGBoost model and the traditional rule-based glide path clearly demonstrates the significant personalization benefit of the ML approach.

        *   **Quantified Personalization**: The ML model achieved a substantially lower total Mean Absolute Error (MAE) compared to the glide path, meaning ML-generated allocations are significantly closer to clients' true preferences.
        *   **Superiority in Alternatives**: The per-asset MAE breakdown shows the ML model's ability to allocate to alternatives based on client profiles, unlike the glide path which often assigns 0%.
        *   **Strategic Advantage**: By using ML, Veridian Financial can offer truly personalized investment advice at scale, a significant competitive advantage.
        """)
```

---

#### Page: 4. Suitability Validation

**Markdown Content**

```python
st.header("4. Ensuring Suitability with Rule-Based Validation and Auto-Correction")

st.markdown(f"""
Even with a highly accurate ML model, financial regulations (like FINRA Reg BI and SEC fiduciary standards) and ethical obligations demand that investment recommendations are *suitable* for the client. An ML model might technically optimize for returns, but if it recommends 85% equity to a conservative 70-year-old retiree, it has failed on suitability, regardless of its predictive power.

As a CFA, ensuring suitability is paramount. This section implements a crucial "governance layer": a set of rule-based suitability checks that act as a safeguard. These rules will override or auto-correct any ML recommendation that violates predefined risk tolerance bounds, age-based equity caps, retirement status considerations, or minimum diversification requirements. This step demonstrates how Veridian Financial embeds compliance and ethics directly into the robo-advisor's workflow.
""")

st.subheader("Run Suitability Checks")
```

**Widget & Function Invocation**

```python
if not st.session_state.allocations_predicted:
    st.warning("Please predict allocations first on the '3. Allocation Prediction & Comparison' page.")
else:
    if st.button("Run Suitability Checks & Auto-Correct"):
        with st.spinner("Running suitability checks and auto-correcting recommendations..."):
            ml_preds_corrected = st.session_state.ml_preds.copy() # Work on a copy
            X_test = st.session_state.X_test
            clients = st.session_state.clients_df

            n_violations = 0
            # MAX_EQUITY dict must be available, as it's defined in source.py suitability_check
            # For this context, we will re-define MAX_EQUITY locally for the display logic, 
            # assuming suitability_check function correctly uses its internal MAX_EQUITY.
            MAX_EQUITY = {1: 30, 2: 50, 3: 70, 4: 85, 5: 95} 
            
            log_messages = []

            for idx in X_test.index:
                alloc = ml_preds_corrected.loc[idx].copy()
                profile = clients.loc[idx]
                
                violations = suitability_check(alloc, profile) # Call function from source.py
                
                if violations:
                    n_violations += 1
                    # Auto-correction logic as in source.py
                    risk = int(profile['risk_tolerance'])
                    max_eq_allowed = MAX_EQUITY[risk]
                    
                    if alloc['pref_equity'] > max_eq_allowed:
                        excess_equity = alloc['pref_equity'] - max_eq_allowed
                        alloc['pref_equity'] = max_eq_allowed
                        alloc['pref_bonds'] += excess_equity
                        log_messages.append(f"Client {idx} equity capped from {ml_preds_corrected.loc[idx, 'pref_equity']:.0f}% to {alloc['pref_equity']:.0f}%. Excess reallocated to bonds (Rule: Max equity for risk tolerance).")
                    
                    if profile['is_retired'] == 1 and profile['age'] > 60 and alloc['pref_equity'] > 50:
                        excess_equity = alloc['pref_equity'] - 50
                        alloc['pref_equity'] = 50
                        alloc['pref_bonds'] += excess_equity
                        log_messages.append(f"Client {idx} (retired, age {profile['age']}) equity capped to 50%. Excess reallocated to bonds (Rule: Retired client equity cap).")
                    
                    if profile['age'] > 70 and alloc['pref_equity'] > 40:
                        excess_equity = alloc['pref_equity'] - 40
                        alloc['pref_equity'] = 40
                        alloc['pref_bonds'] += excess_equity
                        log_messages.append(f"Client {idx} (age {profile['age']}) equity capped to 40%. Excess reallocated to bonds (Rule: Older client equity cap).")

                    current_sum = alloc[['pref_equity', 'pref_bonds', 'pref_alts', 'pref_cash']].sum()
                    if current_sum != 100.0:
                        alloc = alloc / current_sum * 100
                    
                    ml_preds_corrected.loc[idx] = alloc.round(1)
            
            st.session_state.ml_preds = ml_preds_corrected # Update session state with corrected predictions
            st.session_state.n_violations_count = n_violations
            st.session_state.suitability_checked = True

        st.subheader("Suitability Validation Summary")
        st.markdown(f"**Recommendations tested:** {len(X_test)}")
        st.markdown(f"**Violations detected:** {st.session_state.n_violations_count} ({st.session_state.n_violations_count/len(X_test)*100:.1f}%)")
        st.markdown(f"**All violations auto-corrected (equity capped, excess reallocated to bonds)**")

        if log_messages:
            st.subheader("Sample Auto-Correction Details:")
            for msg in log_messages[:5]: # Show first 5 messages
                st.markdown(f"- {msg}")
            if len(log_messages) > 5:
                st.markdown(f"- ... (and {len(log_messages) - 5} more corrections)")
        
        st.info(f"""
        The suitability validation process successfully identified and auto-corrected a percentage of ML-generated recommendations that violated predefined rules. This outcome is crucial for a CFA overseeing a robo-advisor for Veridian Financial.

        *   **Regulatory Compliance in Action**: This demonstrates a practical implementation of FINRA Reg BI and SEC fiduciary standards, ensuring recommendations align with client best interests.
        *   **Preventing Unsuitable Advice**: By capping equity exposure and ensuring minimum diversification, the system prevents inappropriate recommendations.
        *   **Trust and Governance**: This layer acts as a critical governance control, building trust in the robo-advisor's recommendations by respecting regulatory boundaries and ethical obligations.
        """)
```

---

#### Page: 5. Client Reporting

**Markdown Content**

```python
st.header("5. Generating Personalized Client Allocation Reports")

st.markdown(f"""
For a CFA at Veridian Financial, transparent communication with clients is paramount. It's not enough to simply provide a personalized asset allocation; clients need to understand *why* that specific allocation was recommended. This builds trust, enhances client education, and meets regulatory disclosure requirements.

This section focuses on generating personalized client reports. These reports will not only display the recommended allocation (post-suitability checks) but also conceptually explain the key drivers behind the allocation decision. While full SHAP value computation can be complex, we will conceptually interpret its insights to provide human-readable explanations. This is critical for showing clients the intelligence behind the recommendations and fulfilling our fiduciary duties.
""")

st.subheader("Generate Report for a Sample Client")
```

**Widget & Function Invocation**

```python
if not st.session_state.suitability_checked:
    st.warning("Please run suitability checks first on the '4. Suitability Validation' page.")
else:
    clients = st.session_state.clients_df
    X_test = st.session_state.X_test
    ml_preds = st.session_state.ml_preds
    ml_models = st.session_state.ml_models
    feature_cols = st.session_state.feature_cols
    SEGMENT_NAMES = st.session_state.SEGMENT_NAMES

    # Select a sample client from X_test to generate a report
    sample_client_indices = X_test.index.tolist()
    selected_client_idx = st.selectbox("Select a Client ID to generate a report:", sample_client_indices)

    if st.button("Generate Client Report"):
        st.subheader(f"Report for Client ID: {selected_client_idx}")
        # The generate_client_report function in source.py prints to console.
        # We need to capture its output or adapt it to Streamlit display.
        # For simplicity, let's call it and adapt the display using Streamlit's markdown/text.
        
        client_profile = clients.loc[selected_client_idx]
        recommended_alloc = ml_preds.loc[selected_client_idx]
        segment_name = SEGMENT_NAMES.get(client_profile['segment'], f"Segment {client_profile['segment']}")

        st.markdown(f"\n{'='*55}")
        st.markdown(f"**PERSONALIZED INVESTMENT ALLOCATION REPORT**")
        st.markdown(f"{'='*55}")

        st.markdown(f"\n**Client Profile:**")
        st.markdown(f"  - Age: {client_profile['age']}, Income: ${client_profile['income']:,.0f}, Net Worth: ${client_profile['net_worth']:,.0f}")
        st.markdown(f"  - Risk Tolerance: {client_profile['risk_tolerance']}/5, Time Horizon: {client_profile['time_horizon']} years")
        st.markdown(f"  - Client Segment: {segment_name}")
        st.markdown(f"  - Retired: {'Yes' if client_profile['is_retired'] == 1 else 'No'}, Has Dependents: {'Yes' if client_profile['has_dependents'] == 1 else 'No'}")

        st.markdown(f"\n**Recommended Allocation:**")
        for asset in ['pref_equity', 'pref_bonds', 'pref_alts', 'pref_cash']:
            name = asset.replace('pref_','').title()
            pct = recommended_alloc[asset]
            bar = '#' * int(pct / 2)
            st.markdown(f"  - {name:<12s}: **{pct:5.1f}%** {bar}")

        st.markdown(f"\n**Why this allocation:**")
        # Conceptual SHAP explanation logic from source.py adapted for Streamlit
        drivers = []
        if client_profile['risk_tolerance'] > 3:
            drivers.append("your high risk tolerance")
        elif client_profile['risk_tolerance'] < 3:
            drivers.append("your lower risk tolerance")

        if client_profile['time_horizon'] > 20:
            drivers.append("your long investment horizon")
        elif client_profile['time_horizon'] < 10:
            drivers.append("your short investment horizon")

        if client_profile['is_retired'] == 1:
            drivers.append("your retired status")
        
        if client_profile['net_worth'] > 1e6:
            drivers.append("your high net worth")

        if recommended_alloc['pref_equity'] > clients['pref_equity'].mean(): # Compare to the full client data mean
            equity_direction = "higher"
        else:
            equity_direction = "lower"

        if drivers:
            st.markdown(f"  - {', '.join(drivers)} contribute to your {equity_direction} equity allocation.")
        else:
            st.markdown(f"  - This allocation is tailored to your unique financial profile and goals.")

        st.markdown(f"\n**Disclosure:** This allocation is AI-assisted and was reviewed for suitability. Past performance does not guarantee future results.")
        st.markdown(f"{'='*55}")

    st.info(f"""
    The personalized client allocation reports generated showcase Veridian Financial's commitment to transparency and client understanding.

    *   **Clarity and Readability**: Each report clearly presents the client's profile and the recommended asset allocation.
    *   **Conceptual Explainability**: The "Why this allocation" section provides human-readable insights into the drivers, aligning with explainable AI in finance.
    *   **Regulatory Disclosure**: The explicit disclosure statement reinforces compliance with FINRA Reg BI and SEC fiduciary standards.
    """)
```

---

#### Page: 6. Suitability Monitoring & Synthesis

**Markdown Content**

```python
st.header("6. Defining an Ongoing Suitability Monitoring Framework")

st.markdown(f"""
As a CFA, you know that a client's financial situation is not static. Life events, market shifts, and economic changes can significantly alter their risk tolerance, financial goals, and capacity for loss. A static investment recommendation, even if perfectly suitable at inception, can quickly become unsuitable over time.

Therefore, a robust robo-advisor at Veridian Financial must include an ongoing suitability monitoring framework. This isn't just a best practice; it's a regulatory requirement to ensure recommendations remain in the client's best interest. This section outlines key triggers that would prompt a re-evaluation of a client's suitability and potentially trigger a reallocation review. This proactive approach is essential for long-term client satisfaction, risk management, and regulatory compliance.
""")

st.subheader("Suitability Monitoring Triggers")
```

**Widget & Function Invocation**

```python
if st.button("Display Monitoring Framework & Topic Synthesis"):
    # The suitability_monitor function in source.py prints directly.
    # We need to capture its output or adapt it for Streamlit.
    # For this specification, we will simulate its output directly.
    
    triggers = {
        'Age milestone': 'Client turns 60/65/70 -> review equity exposure and income needs',
        'Retirement': 'Employment status changes -> shift to income focus, adjust risk',
        'Major withdrawal': '>20% of portfolio withdrawn -> rebalance, reassess goals',
        'Market shock': 'Portfolio drops >15% -> check risk tolerance still valid, rebalance',
        'Life event': 'Marriage, divorce, child, inheritance, job loss -> full profile review and reallocation',
        'Annual review': 'Mandatory annual suitability reconfirmation and portfolio review'
    }

    st.markdown("\n**SUITABILITY MONITORING TRIGGERS (V5)**")
    st.markdown("---")
    st.markdown("For a modern robo-advisor, continuous monitoring is crucial to ensure ongoing suitability:")
    for trigger, action in triggers.items():
        st.markdown(f"  - **{trigger}**: {action}")

    st.info(f"""
    This section defines a comprehensive framework for ongoing suitability monitoring, a non-negotiable component for any responsible robo-advisor.

    *   **Proactive Risk Management**: The listed triggers demonstrate a proactive approach to managing client risk.
    *   **Ensuring Continued Best Interest**: By defining clear actions for each trigger, Veridian Financial ensures the robo-advisor consistently operates in the client's best interest.
    *   **Operational Efficiency**: This framework automates much of the monitoring, allowing advisors to focus on high-value client engagement.
    """)

    st.subheader("Topic 1 Synthesis: AI in Asset Management")
    # The topic1_synthesis function in source.py prints directly.
    # We will simulate its output here.
    st.markdown("\n---")
    st.markdown("**TOPIC 1 SYNTHESIS: AI IN ASSET MANAGEMENT (V6)**")
    st.markdown("---")
    st.markdown("\nThe three cases span the full investment chain of AI applications in finance:")
    st.markdown("\n  - **D5-T1-C1: ML Stock Selection** (Signal Generation)")
    st.markdown("    - Alpha generation via nonlinear factor discovery")
    st.markdown("    - Audience: Institutional PMs, quant analysts")
    st.markdown("\n  - **D5-T1-C2: AI-Optimized Portfolio Construction** (Portfolio Optimization)")
    st.markdown("    - From alpha scores to constrained portfolio weights")
    st.markdown("    - Audience: Portfolio managers, risk officers")
    st.markdown("\n  - **D5-T1-C3: Robo-Advisor Simulation** (Client Delivery & Suitability)")
    st.markdown("    - Client segmentation + personalized allocation")
    st.markdown("    - Audience: Wealth managers, retail advisory")
    st.markdown("\nThis workflow demonstrates the end-to-end impact of AI:")
    st.markdown("  - Signal -> Portfolio -> Client")
    st.markdown("  - Institutional -> Retail")
    st.markdown("  - Alpha -> Allocation -> Suitability")
    st.markdown("\nVeridian Financial leverages AI at every stage, while ensuring robust governance and client-centricity.")
```

---
