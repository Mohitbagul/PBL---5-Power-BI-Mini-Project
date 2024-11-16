import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import matplotlib.pyplot as plt

# Define the path to the CSV file
csv_file = './Datasets/Bank_Churn.csv'

# Load existing data (if any)
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    df = pd.DataFrame(columns=[
        'RowNumber', 'CustomerId', 'CreditScore', 'GeographyID', 'GenderID',
        'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
        'EstimatedSalary', 'Exited', 'Bank DOJ'
    ])

# Drop rows where 'Exited' is NaN
df = df.dropna(subset=['Exited'])

# Convert 'Bank DOJ' to datetime format
df['Bank DOJ'] = pd.to_datetime(df['Bank DOJ'], errors='coerce')

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Entry", "Graphs & Analysis"])

# Define a function to train and predict using Random Forest
def train_and_predict(df, new_data):
    # Prepare the data
    X = df.drop(columns=['Exited', 'RowNumber', 'CustomerId', 'Bank DOJ'])
    y = df['Exited']

    # Handle NaN values
    X = X.fillna(0)
    y = y.fillna(0)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    st.write(f"Model accuracy: {accuracy * 100:.2f}%")
    st.write(f"Model precision: {precision * 100:.2f}%")
    st.write(f"Model F1 score: {f1:.2f}")
    st.write(f"Model recall: {recall:.2f}")

    # Align new_data to the model's input features
    new_data_aligned = new_data.drop(columns=['Exited', 'RowNumber', 'CustomerId', 'Bank DOJ']).fillna(0)
    new_data_aligned = new_data_aligned[X_train.columns]

    # Predict Exited for the new data
    prediction = clf.predict(new_data_aligned)
    return prediction[0]

# Page 1: Data Entry Form
if page == "Data Entry":
    st.title('Customer Data Entry Form')

    with st.form(key='data_entry_form'):
        row_number = len(df) + 1  # New row number
        customer_id = st.text_input('Customer ID')
        credit_score = st.number_input('Credit Score (0-850)', min_value=0, max_value=850)
        geography_id = st.selectbox('Geography', [1, 2, 3])
        gender_id = st.selectbox('Gender', [1, 2])  # Assuming 1=Male, 2=Female
        age = st.number_input('Age', min_value=0)
        tenure = st.number_input('Tenure (in years)', min_value=0)
        balance = st.number_input('Balance', format="%.2f")  # Two decimal places
        num_of_products = st.number_input('Number of Products', min_value=0)
        has_cr_card = st.selectbox('Has Credit Card', [0, 1])  # 0 = No, 1 = Yes
        is_active_member = st.selectbox('Is Active Member', [0, 1])  # 0 = No, 1 = Yes
        estimated_salary = st.number_input('Estimated Salary', format="%.2f")  # Two decimal places
        bank_doj = st.date_input('Bank Date of Joining', datetime.today())

        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        # Prepare new data
        new_data = pd.DataFrame({
            'RowNumber': [row_number],
            'CustomerId': [customer_id],
            'CreditScore': [credit_score],
            'GeographyID': [geography_id],
            'GenderID': [gender_id],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary],
            'Exited': [0],  # Placeholder, will be predicted
            'Bank DOJ': [bank_doj]
        })

        # Append new data to the DataFrame
        df = pd.concat([df, new_data], ignore_index=True)

        # Predict Exited
        predicted_exit_id = train_and_predict(df, new_data)

        # Update Exited value in DataFrame
        df.at[df.index[-1], 'Exited'] = predicted_exit_id

        # Save the updated DataFrame
        df.to_csv(csv_file, index=False)

        st.success(f"Data submitted successfully! Predicted Exited: {predicted_exit_id}")

# Page 2: Graphs & Analysis
elif page == "Graphs & Analysis":
    st.title('Customer Data Analysis')

    # Add dropdown filters
    year_filter = st.selectbox("Select Year", options=["All", 2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016], index=0)
    month_filter = st.selectbox("Select Month", options=["All"] + list(range(1, 13)), index=0)
    geography_filter = st.selectbox("Select Geography", options=["All", 1, 2, 3], index=0)
    active_filter = st.selectbox("Select Active Category", options=["All", 1, 0], index=0)
    exit_filter = st.selectbox("Select Exit Category", options=["All", 1, 0], index=0)
    gender_filter = st.selectbox("Select Gender", options=["All", 1, 2], index=0)

    # Apply filters to the DataFrame
    filtered_df = df.copy()

    if year_filter != "All":
        filtered_df = filtered_df[filtered_df['Bank DOJ'].dt.year == year_filter]
    
    if month_filter != "All":
        filtered_df = filtered_df[filtered_df['Bank DOJ'].dt.month == month_filter]
    
    if geography_filter != "All":
        filtered_df = filtered_df[filtered_df['GeographyID'] == geography_filter]
    
    if active_filter != "All":
        filtered_df = filtered_df[filtered_df['IsActiveMember'] == active_filter]
    
    if exit_filter != "All":
        filtered_df = filtered_df[filtered_df['Exited'] == exit_filter]
    
    if gender_filter != "All":
        filtered_df = filtered_df[filtered_df['GenderID'] == gender_filter]

    # Plotting the graphs based on filtered data

    # 1. Customer by year and active category (Bar chart)
    st.subheader("Customer by Year and Active Category")
    active_df = filtered_df.groupby([filtered_df['Bank DOJ'].dt.year, 'IsActiveMember']).size().unstack(fill_value=0)
    
    if not active_df.empty:
        active_df.plot(kind='bar', stacked=True)
        plt.xlabel('Year')
        plt.ylabel('Total Customers')
        plt.title('Customers by Year and Active Category')
        st.pyplot(plt)
        plt.clf()

    # 2. Exit customers by gender category (Donut Chart)
    st.subheader("Exit Customers by Gender Category")
    gender_exit_df = filtered_df[filtered_df['Exited'] == 1].groupby('GenderID').size()

    if not gender_exit_df.empty:
        fig, ax = plt.subplots()
        ax.pie(gender_exit_df, labels=['Male', 'Female'], autopct='%1.1f%%', startangle=90, wedgeprops={'width': 0.4})
        plt.title('Exit Customers by Gender')
        plt.axis('equal')
        st.pyplot(fig)
        plt.clf()

    # 3. Exit customers and previous month exit customers by month (Line chart)
    st.subheader("Exit Customers and Previous Month Exit Customers by Month")
    monthly_exit_df = filtered_df[filtered_df['Exited'] == 1].groupby(filtered_df['Bank DOJ'].dt.month).size()

    previous_month_exit_df = monthly_exit_df.shift(1).fillna(0)

    if not monthly_exit_df.empty:
        plt.plot(monthly_exit_df.index, monthly_exit_df.values, label='Exit Customers', marker='o')
        plt.plot(previous_month_exit_df.index, previous_month_exit_df.values, label='Previous Month Exit Customers', linestyle='--', marker='x')
        plt.xlabel('Month')
        plt.ylabel('Number of Exit Customers')
        plt.title('Exit Customers by Month')
        plt.legend()
        st.pyplot(plt)
        plt.clf()

    # 4. Exit customers by credit score (Clustered bar chart)
    st.subheader("Exit Customers by Credit Score")
    credit_score_exit_df = filtered_df[filtered_df['Exited'] == 1].groupby('CreditScore').size()

    if not credit_score_exit_df.empty:
        credit_score_exit_df.plot(kind='bar')
        plt.xlabel('Credit Score')
        plt.ylabel('Number of Exit Customers')
        plt.title('Exit Customers by Credit Score')
        st.pyplot(plt)
        plt.clf()

    # 5. Exit customers by category (Pie chart: credit card holder or not)
    st.subheader("Exit Customers by Category (Credit Card Holder or Not)")
    cr_card_exit_df = filtered_df[filtered_df['Exited'] == 1].groupby('HasCrCard').size()

    if not cr_card_exit_df.empty:
        fig, ax = plt.subplots()
        ax.pie(cr_card_exit_df, labels=['No Credit Card', 'Has Credit Card'], autopct='%1.1f%%', startangle=90)
        plt.title('Exit Customers by Credit Card Category')
        plt.axis('equal')
        st.pyplot(fig)
        plt.clf()
