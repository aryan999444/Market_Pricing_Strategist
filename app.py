import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.utils import load_model
from src.data_cleaning import extract_features
import os # NEW: Import os for file path checking

# Configuration and Asset
st.set_page_config(layout="wide", page_title="Market & Pricing Strategist") # Fixed typo: Startgist -> Strategist

@st.cache_data
def load_data():
    try:
        """Final Processed for visualization"""
        df = pd.read_csv('data/final_processed_data_with_cluster.csv')
        return df
    except FileNotFoundError:
        st.error("Error: final_processed_data_with_cluster.csv not found.")
        return pd.DataFrame()
    
@st.cache_resource
def load_ml_assests():
    try:
        scaler = load_model('model/scaler.joblib')
        kmeans_model = load_model('model/clustering_model.joblib')
        rf_model = load_model('model/classification_model.joblib')
        return scaler, kmeans_model, rf_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# --- FIXED FUNCTION: log_user_input ---
def log_user_input(data: pd.DataFrame):
    """Saves the user input data to a log file for future retraining."""
    
    LOG_FILE = 'data/user_input_log.csv'
    # Use os.path.exists instead of pd.io.common.file_exists (more reliable)
    file_exists = os.path.exists(LOG_FILE) 
    
    log_data = data[['Product_Name', 'Price', 'Rating', 'Review_Count', 'Availability', 'ASIN', 'Product_URL']].copy()

    if not file_exists:
        # ERROR 1 FIX: Must write data to the file path, not a message string.
        log_data.to_csv(LOG_FILE, mode='w', index=False, header=True)
        st.success(f"Log file created and first input saved to {LOG_FILE}")
    else:
        # Append the data without header
        log_data.to_csv(LOG_FILE, mode='a', header=False, index=False)
        st.info("User input saved for future retraining.")
# --- END FIXED FUNCTION ---
    
df_viz = load_data()
scaler, kmeans, rf_classifier = load_ml_assests()

CLUSTER_MAPPING = {
    0: "🥇 Premium Performers",
    1: "🚀 High-Value Growth",
    2: "💰 Mid-Range Standard",
    3: "📉 Budget Clearance",
    4: "📦 Volume & Low-End"
}

# --- Sidebar controls (file upload, model status, quick preview) ---
st.sidebar.title("Controls")
uploaded_file = st.sidebar.file_uploader("Upload processed CSV to override (optional)", type=['csv'])
if uploaded_file is not None:
    try:
        df_viz = pd.read_csv(uploaded_file)
        st.sidebar.success("Uploaded dataset loaded")
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded file: {e}")

model_status = "Loaded" if (scaler is not None and kmeans is not None and rf_classifier is not None) else "Not Loaded"
st.sidebar.markdown(f"**Model status:** {model_status}")

show_raw = st.sidebar.checkbox("Show raw data preview", False)
if show_raw:
    if df_viz.empty:
        st.sidebar.info("No data available to preview")
    else:
        st.sidebar.dataframe(df_viz.head(100))

if not df_viz.empty and 'Cluster_ID' in df_viz.columns:
    df_viz['Segment_Name'] = df_viz['Cluster_ID'].map(CLUSTER_MAPPING).fillna('unknown')

# Header
st.title("📈 Market & Pricing Strategist")
st.markdown("Use this tool to analyze product performance and predict high-performing status for new products.")

# Tabs for different functionlities
tab1, tab2 = st.tabs(["📊 Performance Analysis & Segmentation", "🔮 Predict New Product Success"])

# Tab1 (Code is fine here)
with tab1:
    st.header("Product Segment and Performance Analysis")
    st.subheader("🛒 Market Segmentation Summary")

    if df_viz.empty:
        st.info("No dataset available. Upload a processed CSV in the sidebar or generate processed data from the notebooks.")
    else:
        # Interactive filters
        segments = ['All'] + sorted(df_viz['Segment_Name'].dropna().unique().tolist())
        selected_segment = st.selectbox("Filter by Segment", segments, index=0)
        top_n = st.slider("Show top N products by rating", min_value=3, max_value=20, value=5)

        if selected_segment == 'All':
            filtered = df_viz.copy()
        else:
            filtered = df_viz[df_viz['Segment_Name'] == selected_segment].copy()

        cluster_Counts = df_viz['Segment_Name'].value_counts().reset_index()
        cluster_Counts.columns = ['Segment_Name', 'Product Count']

        fig_counts = px.bar(
            cluster_Counts,
            x='Segment_Name',
            y='Product Count',
            title="Distribution of Products Across Market Segments",
            color='Segment_Name', color_discrete_sequence=px.colors.qualitative.Dark24
        )
        fig_counts.update_xaxes(title_text="Market Segment")
        fig_counts.update_yaxes(title_text="Total Products")
        st.plotly_chart(fig_counts, use_container_width=True)
        st.markdown("---")

        st.subheader("Comparative Segment Performance")
        cluster_summary = df_viz.groupby('Segment_Name')[['Price', 'Rating', 'Review_Count', 'Is_High_Performer']].mean().reset_index()
        cluster_summary.rename(columns={
            'Segment_Name': 'Market Segment',
            'Review_Count': 'Avg. Customer Reviews',
            'Is_High_Performer': 'Avg. Success Rate (0-1)'
        }, inplace=True)
        cluster_summary = cluster_summary[['Market Segment', 'Avg. Customer Reviews', 'Rating', 'Price', 'Avg. Success Rate (0-1)']]
        st.dataframe(cluster_summary.style.format({
            'Price': '₹{:,.0f}',
            'Rating': '{:.2f}',
            'Avg. Success Rate (0-1)': '{:.2f}'
        }).highlight_max(axis=0), use_container_width=True)

        st.markdown("""
            **Interpretation:** This table shows the typical profile of products in each segment.
            * **Price & Reviews:** Indicate the segment's typical cost and market exposure.
            * **Rating & Success Rate:** Indicate the quality and overall performance of the segment.
        """)

        st.markdown("---")
        st.subheader("Top products in selection")
        if filtered.empty:
            st.info("No products match the current selection.")
        else:
            display_cols = ['Product_Name', 'Brand', 'Price', 'Rating', 'Review_Count', 'Product_URL']
            display_cols = [c for c in display_cols if c in filtered.columns]
            top_products = filtered.sort_values(by=['Rating', 'Review_Count'], ascending=False).head(top_n)
            # format price if present
            if 'Price' in top_products.columns:
                top_products = top_products.copy()
                top_products['Price'] = top_products['Price'].apply(lambda x: f"₹{x:,.0f}")
            # Provide clickable links where possible
            if 'Product_URL' in top_products.columns:
                top_products['Product_Link'] = top_products.apply(lambda r: f"[{r['Product_Name']}]({r['Product_URL']})" if pd.notna(r.get('Product_URL')) else r['Product_Name'], axis=1)
                show_cols = ['Product_Link'] + [c for c in ['Brand', 'Price', 'Rating', 'Review_Count'] if c in top_products.columns]
                st.markdown("\n".join(top_products[show_cols].apply(lambda row: ' | '.join([str(x) for x in row]), axis=1).tolist()))
            else:
                st.dataframe(top_products[display_cols], use_container_width=True)

# Tab2
with tab2:
    st.header("Predict High-Performance Status")

    if rf_classifier is None or scaler is None or kmeans is None:
        st.warning("Prediction models or scaler failed to load. Please check your 'model/' folder.")
    else:
        with st.form("prediction_form"):
            st.markdown("**Enter a new product's specifications:**")
            col_in1, col_in2 = st.columns([2, 1])
            product_name = col_in1.text_input("Product Name (e.g., Apple iPhone 15 Pro Max 256GB)", "XYZ Budget Earbuds")
            price = col_in2.number_input("Price (INR)", min_value=1.0, value=5000.0)

            rating = col_in1.slider("Estimated Rating", 1.0, 5.0, 4.0)
            review_count = col_in2.number_input("Estimated Review Count", min_value=0, value=100)

            # Allow user to optionally pick Brand and Storage from existing data for better alignment
            TOP_N_BRANDS = 15
            if not df_viz.empty and 'Brand' in df_viz.columns:
                top_brands = df_viz['Brand'].value_counts().nlargest(TOP_N_BRANDS).index.tolist()
            else:
                top_brands = ["Other"]
            brand = col_in1.selectbox("Brand (optional)", options=["Auto"] + top_brands, index=0)
            storage = col_in2.selectbox("Storage/Capacity (optional)", options=["Unknown", "32GB", "64GB", "128GB", "256GB", "512GB"])

            submitted = st.form_submit_button("Predict Success")

        st.info("🔒 User inputs are logged locally for retraining use; no external transmission.")

        if submitted:
            input_data = pd.DataFrame([{ 'Product_Name': product_name, 'Price': price, 'Rating': rating, 'Review_Count': review_count, 'Availability': 'In Stock', 'ASIN': 'N/A', 'Product_URL': 'N/A', 'Brand': (brand if brand != 'Auto' else np.nan), 'Storage_Capacity': storage }])

            # Log the user input
            log_user_input(input_data)

            # Feature Engineering
            input_df_engineered = extract_features(input_data)
            # Group brand to align with training
            if not df_viz.empty and 'Brand' in df_viz.columns:
                top_brands = df_viz['Brand'].value_counts().nlargest(TOP_N_BRANDS).index.tolist()
            else:
                top_brands = []
            input_df_engineered['Brand_Grouped'] = np.where(input_df_engineered['Brand'].isin(top_brands), input_df_engineered['Brand'], 'Other')

            # One-Hot Encode
            features_to_encode = ['Brand_Grouped', 'Storage_Capacity']
            input_df_encoded = pd.get_dummies(input_df_engineered, columns=features_to_encode, drop_first=True)

            # Align Columns with Training Data
            all_training_features = scaler.feature_names_in_.tolist()
            final_input = input_df_encoded.reindex(columns=all_training_features, fill_value=0)

            # Scale Input
            scaled_array = scaler.transform(final_input)
            scaled_input = pd.DataFrame(scaled_array, columns=scaler.feature_names_in_)

            # Predictions
            prob = rf_classifier.predict_proba(scaled_input)[0, 1]
            segment_id = int(kmeans.predict(scaled_input.values)[0])
            segment_name = CLUSTER_MAPPING.get(segment_id, f"Cluster {segment_id} (Unknown)")

            st.markdown("---")
            col_res1, col_res2 = st.columns(2)
            col_res1.metric(label="Probability of High Performance", value=f"{prob*100:.2f}%")
            col_res2.metric(label="Product Segment", value=segment_name)

            if prob > 0.65:
                st.success("Analysis: This product has a strong likelihood of being a high performer!")
                st.balloons()
            elif prob > 0.4:
                st.info("Analysis: Moderate likelihood of high performance. Consider further validation tests.")
            else:
                st.warning("Analysis: Low likelihood of high performance.")

            # Similar Product Suggestions (interactive)
            st.markdown("### 🛍️ Similar Product Suggestions from the Market")
            if df_viz.empty or 'Cluster_ID' not in df_viz.columns:
                st.info("No market data available to provide suggestions.")
            else:
                suggestion_df = df_viz[df_viz['Cluster_ID'] == segment_id].copy()
                suggestion_df = suggestion_df[suggestion_df.get('Is_Available', 1) == 1]
                suggestion_df = suggestion_df.sort_values(by=['Rating', 'Review_Count'], ascending=False).head(5)

                if not suggestion_df.empty:
                    st.markdown(f"**Top products in the '{segment_name}' segment:**")
                    display_cols = ['Product_Name', 'Brand', 'Price', 'Rating', 'Product_URL']
                    display_cols = [c for c in display_cols if c in suggestion_df.columns]
                    suggestion_df = suggestion_df[display_cols].copy()
                    if 'Price' in suggestion_df.columns:
                        suggestion_df['Price'] = suggestion_df['Price'].apply(lambda x: f"₹{x:,.0f}")

                    # Build markdown list with clickable links when URL exists
                    suggestion_list = []
                    for _, row in suggestion_df.iterrows():
                        name = row.get('Product_Name', 'Unknown')
                        url = row.get('Product_URL', '')
                        link = f"[{name}]({url})" if pd.notna(url) and url != '' else name
                        brand_text = f" by *{row.get('Brand','')}*" if 'Brand' in row.index else ''
                        price_text = f" | Price: {row.get('Price','N/A')}" if 'Price' in row.index else ''
                        rating_text = f" | Rating: {row.get('Rating','N/A')}" if 'Rating' in row.index else ''
                        suggestion_list.append(f"- **{link}**{brand_text}{price_text}{rating_text}")

                    # Display suggestions in the sidebar only
                    with st.sidebar.expander("Similar Product Suggestions", expanded=True):
                        st.markdown(f"**Top products in the '{segment_name}' segment:**")
                        st.markdown("\n".join(suggestion_list))
                else:
                    st.info(f"No available products found in the '{segment_name}' segment for suggestion.")