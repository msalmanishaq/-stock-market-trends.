import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Set page configuration
st.set_page_config(
    page_title="Stock Market Prediction",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

# Load data
df = pd.read_csv('gld_price_data.csv', parse_dates=['Date'])

# Show the table data when checkbox is ON
if st.checkbox('Show the dataset as table data'):
    st.table(df.head())

# Set sidebar options
st.sidebar.title("Settings")
selected_features = st.sidebar.multiselect("Select Features", df.columns[1:], default=['GLD', 'USO', 'SLV', 'EUR/USD'])
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, step=0.1)

# Prepare the feature matrix (X) and the target variable (y)
x = df[selected_features]
y = df['SPX']

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=test_size, random_state=2)

# Create an instance of the random forest regressor model
regressor = RandomForestRegressor(n_estimators=100)

# Fit the model using the training data
regressor.fit(X_train, Y_train)

# Make predictions on the test set
test_data_pre = regressor.predict(X_test)

# Calculate R-squared error
error_score = metrics.r2_score(Y_test, test_data_pre)

# Convert Y_test to a list for plotting
y_test = list(Y_test)

# Set 3D background
st.markdown(
    """
    <style>
    body {
        background-color: #e6f7ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the actual and predicted data
st.title("Stock Market Prediction")
st.subheader("Actual Data vs Predicted Data")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(len(y_test)), y_test, label='Actual Data', marker='o')
ax.plot(range(len(test_data_pre)), test_data_pre, label='Predicted Data', marker='s')
ax.set_title('Actual Data vs Predicted Data')
ax.set_xlabel('Index')
ax.set_ylabel('Value')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Display the R-squared error
st.subheader("R-squared Error")
st.write(f"The R-squared error is: {error_score}")

