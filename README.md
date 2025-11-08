# Market & Pricing Strategist 📊

An interactive analytics and prediction tool for e-commerce product performance optimization. This application helps analyze market segments and predict product success probability using machine learning.

## Project Objective 🎯

The Market & Pricing Strategist provides two core capabilities:
1. **Market Analysis:** Segment and analyze product performance across the electronics marketplace
2. **Success Prediction:** Predict whether a new product will be a high performer

A **High-Performer** is defined as a product meeting both criteria:
- Rating ≥ 4.5 stars
- Review Count ≥ 1,000 reviews

## System Requirements 🔧

- Python 3.8 or higher
- Required Python packages:
  ```
  streamlit>=1.28.0
  pandas>=2.0.0
  numpy>=1.24.0
  scikit-learn>=1.3.0
  plotly>=5.18.0
  joblib>=1.3.0
  ```

## Project Structure 📁

```
Market_Pricing_Strategist/
├── app.py                 # Streamlit web application
├── requirements.txt       # Python dependencies
├── data/
│   ├── amazon_all_electronics_data.csv    # Raw input data
│   ├── processed_data.csv                 # Cleaned data with features
│   ├── final_processed_data_with_cluster.csv  # Including segments
│   └── user_input_log.csv                # Prediction inputs log
├── model/
│   ├── scaler.joblib             # Feature standardization
│   ├── clustering_model.joblib   # K-Means segmentation
│   └── classification_model.joblib # Random Forest classifier
├── notebooks/
│   ├── data_cleaning_eda.ipynb   # Data exploration & cleaning
│   └── model_development.ipynb   # Model training & evaluation
└── src/
    ├── __init__.py
    ├── data_cleaning.py     # Feature engineering pipeline
    ├── model_training.py    # Model training pipeline
    └── utils.py            # Shared utilities
```

## Development & MLOps Approach 🔄

### Feature Engineering (`data_cleaning.py`)
The feature extraction pipeline:
1. **Data Cleaning:** Handles missing values and standardizes formats
2. **Feature Creation:** 
   - Product characteristics (Wireless, 5G, Ultra HD, etc.)
   - Brand grouping (top 15 brands)
   - Storage capacity extraction
3. **Target Definition:** Labels high-performers (Rating ≥ 4.5 AND Reviews ≥ 1000)

### Model Pipeline (`model_training.py`)
Two-stage modeling approach:
1. **Market Segmentation** (K-Means, k=5):
   - 🥇 Premium Performers
   - 🚀 High-Value Growth
   - 💰 Mid-Range Standard
   - 📉 Budget Clearance
   - 📦 Volume & Low-End

2. **Success Prediction** (Random Forest):
   - Balanced class weights
   - Probability threshold optimization
   - Feature importance analysis

## Model Mechanism & Feature Importance 🎯

### Random Forest Classification
The success prediction model uses a hierarchical importance structure:

**Primary Drivers** (Highest Impact):
- **Rating:** Direct indicator of product quality and customer satisfaction
- **Review Count:** Measures market validation and product maturity

**Secondary Modifiers** (Supporting Features):
- **Price:** Influences success within market segments
- **Brand Presence:** Top-15 brand association
- **Product Features:** Wireless, 5G, Ultra HD, etc.

The model uses balanced class weights to handle the natural imbalance in high-performing products.

### K-Means Segmentation Logic
Market segmentation is primarily driven by:
1. **Price Bands:** Primary factor in segment assignment
2. **Performance Metrics:** Secondary clustering criteria
   - Rating distribution
   - Review volume
   - Success rate within price range

This creates natural market tiers that align with common customer decision patterns.

## Strategic Test Cases 📋

| Scenario | Input Values | Expected Output | Rationale |
|----------|--------------|-----------------|-----------|
| **Max Success Case** | Price: ₹49,999<br>Rating: 4.8<br>Reviews: 2,500 | Prob: >85%<br>Segment: Premium Performers | High-end product with strong metrics should be identified as likely success |
| **Target Failure Case** | Price: ₹999<br>Rating: 3.5<br>Reviews: 50 | Prob: <30%<br>Segment: Volume & Low-End | Budget product with weak metrics should show low success probability |
| **Mid-Range Check** | Price: ₹15,999<br>Rating: 4.3<br>Reviews: 800 | Prob: ~50%<br>Segment: Mid-Range Standard | Borderline case tests model's threshold sensitivity |

### MLOps & Retraining

To retrain models with new data (Windows PowerShell):
```powershell
# 1. Clean & process new data
python -c "from src.data_cleaning import extract_features; import pandas as pd; df = pd.read_csv('data/amazon_all_electronics_data.csv'); processed = extract_features(df); processed.to_csv('data/processed_data.csv', index=False)"

# 2. Retrain models
python -c "from src.model_training import train_and_save_models; train_and_save_models()"
```

## How to Use the Application 🚀

### Setup & Launch

1. Create virtual environment (recommended):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

3. Launch the application:
   ```powershell
   streamlit run app.py
   ```

### Tab 1: Performance Analysis & Segmentation 📊

The Analysis tab provides:
- **Market Segmentation Summary:** Interactive visualization of product distribution across segments
- **Comparative Segment Performance:** 
  - Average Price, Rating, and Review Count by segment
  - Success Rate analysis
  - Top products list with filtering options

### Tab 2: Predict New Product Success 🔮

The Prediction tab offers:
1. **Input Form:**
   - Product specifications
   - Optional brand selection
   - Price and estimated metrics

2. **Prediction Output:**
   - Success probability
   - Market segment assignment
   - Performance analysis

3. **Similar Product Suggestions:**
   - Top 3 performing products in the assigned segment
   - Direct product links
   - Key metrics comparison

> **⚠️ Data Privacy Notice:** User inputs are logged locally for model retraining purposes. This data helps improve prediction accuracy over time but is never transmitted externally.

### Test Case Guidelines
When using the prediction tab, consider these validation approaches:
1. **Cross-Segment Validation:**
   - Test similar products across different price points
   - Observe segment shifts and probability changes
2. **Feature Sensitivity:**
   - Modify one metric at a time to understand impact
   - Pay special attention to the Rating/Review threshold boundaries
3. **Brand Impact Testing:**
   - Compare predictions with/without known brand selection
   - Test both top-15 brands and "Other" category

## Sidebar Controls 🎛️

The application sidebar provides:
- Dataset override via CSV upload
- Model status monitoring
- Raw data preview option
- Similar product suggestions in a collapsible panel

## Contributing 🤝

To contribute:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with:
   - Clear description of changes
   - Updated tests if needed
   - Verification that all existing tests pass

<!-- ## License 📄

MIT License - See LICENSE file for details
-->
## Acknowledgments 💝

Thank you for exploring the Market & Pricing Strategist project! 

Created and maintained by [Aryan Srivastava](https://www.linkedin.com/in/aryan-srivastava-529840252/) 
