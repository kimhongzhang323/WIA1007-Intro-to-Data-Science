import streamlit as st
import os
import pandas as pd
from nbconvert import HTMLExporter
import nbformat
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib

# Function to render the notebook
def render_notebook(notebook_path):
    with open(notebook_path) as f:
        notebook = nbformat.read(f, as_version=4)
    html_exporter = HTMLExporter()
    (body, resources) = html_exporter.from_notebook_node(notebook)
    return body

# Sidebar navigation using selectbox for a cleaner look
st.sidebar.header("Navigation")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Select a Stage",
    [
        "Main Page",
        "Data Preprocessing 1",
        "Data Preprocessing 2",
        "EDA",
        "Model Building",
        "Model Evaluation",
        "Documentation",
    ],
)

st.sidebar.markdown("---")
st.sidebar.write("Select a stage to view its content.")

# Define content for each page
if page == "Main Page":
        st.header("WIA1007 Intro to Data Science Assignment")
        st.write(""" 
            Welcome to our data science project! In this assignment, we have chosen an uncleaned dataset from Kaggle to carry out our work. 
            We will go through various stages including data preprocessing, exploratory data analysis (EDA), model building, and model evaluation.
        """)
        
        # Display sample dataset from CSV (data/laptopData.csv)
        st.subheader("Dataset from laptopData.csv")
        
        # Load the dataset
        laptop_data_path = "../data/laptopData.csv"  # Adjust the path to your CSV file
        if os.path.exists(laptop_data_path):
            df = pd.read_csv(laptop_data_path)
            st.dataframe(df)  # Display the dataframe in Streamlit
        else:
            st.error(f"File {laptop_data_path} not found.")
    
elif page == "Data Preprocessing 1":
    with st.expander("Data Preprocessing Steps", expanded=True):
        st.header("Data Preprocessing Steps")
        st.write("Perform initial data cleaning, handle missing values, and normalize data.")
        
        # Load and display a specific dataset for Data Preprocessing 1
        preprocessing_data_path_1 = "../data/laptopData_alvi.csv"  # Adjust the path to your first data preprocessing CSV file
        if os.path.exists(preprocessing_data_path_1):
            df_preprocessing_1 = pd.read_csv(preprocessing_data_path_1)
            st.dataframe(df_preprocessing_1)  # Display the dataframe for Data Preprocessing 1
        else:
            st.error(f"File {preprocessing_data_path_1} not found.")
        
        # Example of Data Preprocessing:
        st.subheader("Example Code for Data Preprocessing")

        st.markdown("""
        Here's an example of how the data preprocessing was done:

        ```python
        # Handle missing values
        df['CPU'] = df['CPU'].fillna(df['CPU'].mode()[0])  # Fill missing CPU values with the most frequent value
        df['RAM'] = df['RAM'].fillna(df['RAM'].mean())  # Fill missing RAM values with the column mean

        # Normalize data (for columns like 'CPU_Speed', 'RAM', 'Storage', etc.)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df[['CPU_Speed', 'RAM', 'Storage']] = scaler.fit_transform(df[['CPU_Speed', 'RAM', 'Storage']])

        # Encode categorical data
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder()
        encoded_columns = encoder.fit_transform(df[['Brand', 'OS']])
        ```
        """)

        st.write("""
        The preprocessing steps included handling missing data, normalizing features, and encoding categorical variables to prepare the dataset for the model.
        """)

        # Automatically display the Jupyter Notebook for this stage
        notebook_path_1 = "WIA1007_Data_C&PP_Alvi_Version1.ipynb"  # Adjust the path accordingly
        if os.path.exists(notebook_path_1):
            notebook_html_1 = render_notebook(notebook_path_1)
            st.components.v1.html(notebook_html_1, height=600, scrolling=True)
        else:
            st.error(f"Notebook {notebook_path_1} not found.")
    
    # Documentation Section
    with st.expander("Documentation for Data Preprocessing 1", expanded=True):
        st.subheader("Documentation for Data Preprocessing 1")
        st.markdown(""" 
        ### Data Collection:
        For the project “Laptop Price Prediction”, the data has been collected from “Kaggle”. The author of the data specified that the data is “Uncleaned”...
        The data consists of information on different laptops, including attributes such as CPU, RAM, storage, and other laptop specifications along with their prices.
        
        The dataset has missing values and potential inconsistencies which we will address in the preprocessing steps. Some columns may require normalization, encoding of categorical variables, and filling of missing values.

        ### Data Review:
        First of all, the data has been downloaded in “CSV” format. We checked for missing values and inconsistencies in column values. The data set contains various features such as:
        
        - Brand of the laptop
        - Screen size
        - Processor speed
        - Amount of RAM and storage
        - Operating system

        ### Data Preprocessing Steps:
        In this stage of data preprocessing, we performed several tasks:
        
        1. **Handling Missing Data**:
           Missing values are handled using different strategies depending on the column. Some columns will have missing numeric data, and these will be filled with the mean or median of the column. Categorical variables with missing values will be filled with the mode (most frequent value).

        2. **Normalization**:
           Certain numerical features like CPU speed, RAM, storage, and others may need to be scaled so that they are on the same scale. We normalize features like CPU speed and storage size to ensure that the machine learning models don’t become biased toward certain features with larger magnitudes.
        
        3. **Removing Irrelevant Features**:
           Some columns may not contribute to the predictive model (for example, columns like 'Image URLs' that do not provide meaningful information). These are removed to streamline the dataset.

        4. **Encoding Categorical Data**:
           Categorical variables (like 'Brand', 'Operating System', etc.) are encoded into numerical values using techniques like One-Hot Encoding or Label Encoding. This step is essential for feeding data into machine learning algorithms that require numerical input.
        
        5. **Data Splitting**:
           The final step involves splitting the dataset into training and testing sets to ensure that our models can be validated properly.

        ### Conclusion:
        After preprocessing, the data will be ready to be used for model training. Handling missing values, normalization, and encoding will ensure that the data is clean and consistent, which is critical for building robust machine learning models.
        
        **Next Step**: Once the data is cleaned and preprocessed, we will proceed to the next stage of feature engineering and model building.
        """)

    
elif page == "Data Preprocessing 2":
    with st.expander("Data Preprocessing 2", expanded=True):
        st.header("Data Preprocessing 2")
        st.write("Feature engineering, encoding categorical variables, and splitting the dataset.")
        
        # Load and display a specific dataset for Data Preprocessing 2
        preprocessing_data_path_2 = "../data/processed_laptopData.csv"  # Adjust the path to your second data preprocessing CSV file
        if os.path.exists(preprocessing_data_path_2):
            df_preprocessing_2 = pd.read_csv(preprocessing_data_path_2)
            st.dataframe(df_preprocessing_2)  # Display the dataframe for Data Preprocessing 2
        else:
            st.error(f"File {preprocessing_data_path_2} not found.")
        
        # Automatically display the WIA1007_Group_Assignment_Table.ipynb notebook for this stage
        notebook_path_2 = "WIA1007_Group_Assignment_Table.ipynb"  # Adjust the path accordingly
        if os.path.exists(notebook_path_2):
            notebook_html_2 = render_notebook(notebook_path_2)
            st.components.v1.html(notebook_html_2, height=800, scrolling=True)
        else:
            st.error(f"Notebook {notebook_path_2} not found.")
    
elif page == "EDA":
    with st.expander("EDA", expanded=True):
        st.header("Exploratory Data Analysis (EDA)")
        st.write("Analyze patterns, visualize data, and generate insights.")
        
        # Render the EDA notebook automatically
        notebook_path_eda = "IDS_EDA_PART.ipynb"  # Adjust the path to your EDA notebook
        if os.path.exists(notebook_path_eda):
            notebook_html_eda = render_notebook(notebook_path_eda)
            st.components.v1.html(notebook_html_eda, height=800, scrolling=True)
        else:
            st.error(f"Notebook {notebook_path_eda} not found.")

    with st.expander("Gallery", expanded=True):
        st.subheader("EDA Visualizations Gallery")
        st.write("Here are some key visualizations from the exploratory data analysis phase:")

        # Define the directory containing the EDA images
        image_dir = "../assets/"  # Adjust this to the correct path for your images
        if os.path.exists(image_dir):
            # Load and display images in a gallery format
            image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                for image_path in image_files:
                    st.image(image_path, caption=os.path.basename(image_path), use_column_width=True)
            else:
                st.warning("No images found in the gallery directory.")
        else:
            st.error(f"Image directory {image_dir} not found.")
    
elif page == "Model Building":
        st.header("Model Building")
        st.write("Train models using algorithms such as Random Forest Regression.")
        
        # Load data for the model building (example: laptopData.csv)
        model_data_path = "../data/laptopData_table_cleaned.csv"  # Adjust the path to your dataset
        if os.path.exists(model_data_path):
            df_model = pd.read_csv(model_data_path)
            
            # Assuming 'Price' is the target variable and other columns are features
            st.write("Dataset used for Random Forest Regression:")
            st.dataframe(df_model)  # Display the dataset

            # Select features and target variable
            X = df_model.drop(columns=['Price'])  # Features (excluding the target)
            y = df_model['Price']  # Target variable

            # Define the numerical and categorical features for preprocessing
            numerical_features = ['Inches', 'Ram', 'Weight', 'Screen_Width', 'Screen_Height', 'ppi', 'CPU_Speed', 'SSD', 'HDD', 'Flash', 'Hybrid']
            numerical_transformer = StandardScaler()

            categorical_features = ['Company', 'TypeName', 'OpSys', 'Gpu_Brand', 'OS']
            categorical_transformer = OneHotEncoder(handle_unknown='ignore')

            # Combine the transformers in a column transformer
            preprocessor = ColumnTransformer(
                transformers=[ 
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ])

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Create the pipeline with preprocessing and model
            pipeline = Pipeline(steps=[ 
                ('preprocessor', preprocessor), 
                ('model', RandomForestRegressor(n_estimators=100, random_state=42)) 
            ])

            # Train the model
            pipeline.fit(X_train, y_train)

            # Predict on the test set
            y_pred = pipeline.predict(X_test)

            # Evaluate model performance
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)

            # Display evaluation metrics
            st.write(f"Mean Absolute Error (MAE): {mae}")
            st.write(f"Mean Squared Error (MSE): {mse}")
            st.write(f"Root Mean Squared Error (RMSE): {rmse}")
            st.write(f"R² Score: {r2}")

            # Visualize predictions vs actual values
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
            plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
            plt.title("Random Forest Regression: Actual vs Predicted")
            plt.xlabel("Actual Price")
            plt.ylabel("Predicted Price")
            st.pyplot(plt)

        # After training the model pipeline, save it to a file
        joblib.dump(pipeline, "trained_model_pipeline.pkl")

elif page == "Model Evaluation":
    st.header("Model Evaluation")
    st.write("""
        Here we evaluate the performance of the model using multiple metrics such as MAE, MSE, RMSE, and R² score.
        We also allow you to input custom values for laptop features to predict the price using the trained model.
    """)

    # Load the saved model pipeline
    pipeline = joblib.load("trained_model_pipeline.pkl")

    # User Input for testing the model
    st.subheader("Test the Model")

    # Allow user input for features (users can now choose or input values)
    inches = st.selectbox("Inches", options=[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0], index=6)  # Default is 15.0
    ram = st.selectbox("RAM (GB)", options=[2, 4, 6, 8, 16, 32, 64], index=3)  # Default is 8 GB
    weight = st.selectbox("Weight (kg)", options=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0], index=2)  # Default is 1.5 kg
    screen_width = st.selectbox("Screen Width (cm)", options=[20.0, 30.0, 40.0, 50.0, 60.0], index=2)  # Default is 35.0 cm
    screen_height = st.selectbox("Screen Height (cm)", options=[10.0, 15.0, 20.0, 25.0, 30.0, 40.0], index=3)  # Default is 23.0 cm
    ppi = st.selectbox("PPI", options=[100, 150, 200, 250, 300, 350, 400], index=2)  # Default is 150
    cpu_speed = st.selectbox("CPU Speed (GHz)", options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], index=3)  # Default is 2.5 GHz
    ssd = st.selectbox("SSD (GB)", options=[0, 128, 256, 512, 1024, 2048], index=2)  # Default is 256 GB
    hdd = st.selectbox("HDD (GB)", options=[0, 128, 256, 512, 1024, 2048], index=4)  # Default is 1024 GB
    flash = st.selectbox("Flash (GB)", options=[0, 32, 64, 128, 256, 512], index=0)  # Default is 0 GB
    hybrid = st.selectbox("Hybrid (GB)", options=[0, 32, 64, 128, 256, 512], index=0)  # Default is 0 GB

    # Allow the user to select categorical features (Company, TypeName, OpSys, Gpu_Brand, OS)
    company = st.selectbox("Company", options=["Dell", "HP", "Lenovo", "Asus", "Acer", "Apple"], index=0)
    typename = st.selectbox("Type Name", options=["Laptop", "Ultrabook", "Gaming", "Convertible", "Workstation"], index=0)
    opsys = st.selectbox("Operating System", options=["Windows", "MacOS", "Linux"], index=0)
    gpu_brand = st.selectbox("GPU Brand", options=["NVIDIA", "AMD", "Intel", "None"], index=0)
    os = st.selectbox("OS Version", options=["Windows 10", "Windows 11", "MacOS", "Linux"], index=0)

    # Use the trained pipeline to make predictions when the button is clicked
    if st.button("Predict Price"):
        input_data = {
            'Inches': inches,
            'Ram': ram,
            'Weight': weight,
            'Screen_Width': screen_width,
            'Screen_Height': screen_height,
            'ppi': ppi,
            'CPU_Speed': cpu_speed,
            'SSD': ssd,
            'HDD': hdd,
            'Flash': flash,
            'Hybrid': hybrid,
            'Company': company,
            'TypeName': typename,
            'OpSys': opsys,
            'Gpu_Brand': gpu_brand,
            'OS': os
        }
        user_input_df = pd.DataFrame([input_data])  # Create DataFrame from the input data
        predicted_price = pipeline.predict(user_input_df)  # Make prediction
        st.write(f"The predicted price of the laptop is: ${predicted_price[0]:,.2f}")

    # Model Evaluation Metrics
    st.subheader("Model Evaluation Metrics")

    # Example: Load the test data for evaluation
    test_data_path = "../data/laptopData_table_cleaned.csv"  # Path to your test data CSV file

    # Load the test data
    test_data = pd.read_csv(test_data_path)

    # Separate numeric and categorical columns
    numeric_cols = test_data.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = test_data.select_dtypes(exclude=['float64', 'int64']).columns

    # Fill missing values in numeric columns only
    test_data[numeric_cols] = test_data[numeric_cols].fillna(test_data[numeric_cols].mean())

    # If needed, handle missing values in categorical columns separately (e.g., by filling with the mode)
    test_data[categorical_cols] = test_data[categorical_cols].apply(lambda col: col.fillna(col.mode()[0]), axis=0)

    # Separate features and target variable
    X_test = test_data.drop(columns=["Price"])  # Features
    y_test = test_data["Price"]  # Target

    # Use the existing fitted preprocessor and model from the loaded pipeline
    y_pred = pipeline.predict(X_test)

    # Display Error Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"R-squared (R²): {r2:.2f}")

    # Plotting: Actual vs Predicted Prices
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, color='blue', alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Actual vs Predicted Prices")
    st.pyplot(fig)

    # Plotting: Residuals
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, color='green', alpha=0.5)
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_xlabel("Predicted Price")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals Plot")
    st.pyplot(fig)
        
else:
    st.write("Select a stage to view its content.")
