# QuLab: Lab 51: Robo-Advisor Simulation

## Modernizing Wealth Management at Veridian Financial

[![QuLab Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)](https://www.quantuniversity.com/)

This Streamlit application, "QuLab: Lab 51: Robo-Advisor Simulation," serves as a hands-on lab project designed for CFA Charterholders and Senior Investment Analysts at a fictional firm, Veridian Financial. The project simulates the core functions of a modern robo-advisor platform, focusing on leveraging AI to enhance client services, particularly in client segmentation, personalized asset allocation, and robust suitability monitoring.

The application guides the user through a structured workflow to evaluate how AI can personalize investment advice at scale, improve client fit compared to generic rules, and critically, how regulatory suitability and ethical considerations are embedded into the AI workflow. It provides insight into automating Investment Policy Statement (IPS) construction, enhancing risk management, and ensuring regulatory compliance for client-facing AI.

## Features

The application provides a step-by-step workflow for simulating and evaluating a robo-advisor:

1.  **Client Data Generation**:
    *   Generates a synthetic dataset of 5,000 diverse client profiles, including demographic, financial, risk-related features, and "true" preferred asset allocations, to serve as a realistic testbed.
2.  **Client Segmentation**:
    *   Applies **K-Means Clustering** and **PCA visualization** to uncover natural and meaningful client segments beyond traditional demographic buckets.
    *   Provides a detailed segment profile table showing average characteristics and preferred allocations for each segment.
3.  **Allocation Prediction & Comparison**:
    *   Trains a **Multi-Output XGBoost Regressor** to predict personalized asset allocations (equity, bonds, alternatives, cash) for clients based on their profiles.
    *   Compares the ML model's performance against a traditional **rule-based glide path** using **Mean Absolute Error (MAE)** to quantify the personalization benefit.
4.  **Suitability Validation**:
    *   Implements a crucial "governance layer" of **rule-based suitability checks** to ensure all ML recommendations comply with regulatory standards (e.g., FINRA Reg BI, SEC fiduciary duties) and ethical guidelines.
    *   Includes an **auto-correction mechanism** to adjust unsuitable allocations (e.g., capping equity for conservative retirees).
5.  **Client Reporting**:
    *   Generates **personalized client reports** displaying the recommended allocation (post-suitability checks) and providing conceptual explanations for the allocation decision, enhancing transparency and trust.
6.  **Suitability Monitoring & Synthesis**:
    *   Outlines a framework for **ongoing suitability monitoring**, defining key triggers (e.g., age milestones, market shocks, life events) that would prompt a re-evaluation of client suitability.
    *   Provides a **synthesis of AI applications in asset management**, connecting this robo-advisor simulation to broader AI use cases in the financial sector.

## Getting Started

Follow these instructions to set up and run the Streamlit application on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/YourUsername/your-robo-advisor-repo.git
    cd your-robo-advisor-repo
    ```
    *(Note: Replace `YourUsername/your-robo-advisor-repo.git` with the actual repository URL if available, otherwise assume a local project setup.)*

2.  **Create a virtual environment** (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install the required packages**:
    Create a `requirements.txt` file in your project root with the following content:

    ```
    streamlit
    pandas
    matplotlib
    numpy
    seaborn
    scikit-learn
    xgboost
    ```

    Then install the packages:

    ```bash
    pip install -r requirements.txt
    ```

### Project Structure

The project is organized as follows:

```
├── app.py                  # Main Streamlit application file
├── source.py               # Contains helper functions (data generation, glide path, suitability checks)
├── requirements.txt        # Python dependencies
└── README.md               # This README file
```

## Usage

To run the Streamlit application:

1.  Ensure you have followed the installation steps and activated your virtual environment.
2.  From the project's root directory, execute the following command:

    ```bash
    streamlit run app.py
    ```

3.  Your web browser will automatically open to the Streamlit application (usually at `http://localhost:8501`).

### Navigating the Application

The application is structured as a sequential workflow, accessible via the **sidebar navigation**:

*   **Introduction**: Provides an overview of the project and your role as a CFA Charterholder.
*   **1. Client Data Generation**: Start here to create the synthetic client profiles. Click the "Generate 5,000 Client Profiles" button.
*   **2. Client Segmentation**: Proceed to segment the generated client data using K-Means. Click "Perform K-Means Client Segmentation".
*   **3. Allocation Prediction & Comparison**: Train the ML model and compare its asset allocations with a traditional glide path. Click "Train ML Model & Compare Allocations".
*   **4. Suitability Validation**: Run the rule-based suitability checks and observe auto-corrections. Click "Run Suitability Checks & Auto-Correct".
*   **5. Client Reporting**: Select a client ID and generate a personalized report explaining their recommended allocation. Click "Generate Client Report".
*   **6. Suitability Monitoring & Synthesis**: Review the framework for ongoing suitability monitoring and a synthesis of AI in asset management. Click "Display Monitoring Framework & Topic Synthesis".

Follow the steps sequentially to fully experience the robo-advisor simulation.

### Workflow Overview
The application's workflow visually represented:
![Robo-Advisor Workflow](https://raw.githubusercontent.com/streamlit/docs/main/docs/static/img/workflow_overview.png)

## Data Persistence & Caching

To optimize performance and avoid lengthy model training times, the application implements automatic data caching:

*   **Automatic Caching**: When you generate client data, perform segmentation, or train models, the results are automatically saved to the `cached_data/` directory.
*   **Automatic Loading**: On subsequent runs or page navigations, the app automatically detects and loads cached data, eliminating the need to regenerate or retrain.
*   **Smart Button Behavior**: 
    *   The "Train ML Model & Compare Allocations" button checks for cached models first. If found, it loads them instantly. If not, it trains new models and saves them.
    *   Use the "Retrain" button to force model retraining even when cached models exist.
    *   Use the "Clear Cached Data" button to remove all cached files and start fresh.

**Cached Files**:
*   `clients_data.csv` - Generated client profiles
*   `segment_profile.csv` & `segment_names.pkl` - Segmentation results
*   `ml_models.pkl` - Trained XGBoost models
*   `ml_predictions.csv` - ML model predictions
*   `glide_predictions.csv` - Glide path predictions
*   `ml_predictions_corrected.csv` - Suitability-corrected predictions
*   `X_test.csv` & `Y_test.csv` - Test dataset

## Technology Stack

*   **Python**: Programming language
*   **Streamlit**: For building interactive web applications and user interfaces.
*   **Pandas**: For data manipulation and analysis.
*   **NumPy**: For numerical operations.
*   **Matplotlib** & **Seaborn**: For data visualization (e.g., PCA plots, MAE comparison charts).
*   **Scikit-learn**: For machine learning tasks including:
    *   `StandardScaler` for feature scaling.
    *   `KMeans` for client clustering.
    *   `PCA` for dimensionality reduction and visualization.
    *   `train_test_split` for data partitioning.
    *   `mean_absolute_error` for model evaluation.
*   **XGBoost**: For the high-performance gradient boosting machine learning model used in allocation prediction.

## Contributing

This project is primarily a lab assignment. However, if you have suggestions for improvements or find any issues, feel free to open an issue in the repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details (or assume MIT if no explicit file exists).

## Contact

This QuLab project is developed by QuantUniversity.
For more information, please visit [QuantUniversity](https://www.quantuniversity.com/).