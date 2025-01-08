import requests
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
from dotenv import load_dotenv

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
        
        # # Load and display a specific dataset for Data Preprocessing 1
        # preprocessing_data_path_1 = "../data/laptopData_alvi.csv"  # Adjust the path to your first data preprocessing CSV file
        # if os.path.exists(preprocessing_data_path_1):
        #     df_preprocessing_1 = pd.read_csv(preprocessing_data_path_1)
        #     st.dataframe(df_preprocessing_1)  # Display the dataframe for Data Preprocessing 1
        # else:
        #     st.error(f"File {preprocessing_data_path_1} not found.")
        
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
        # preprocessing_data_path_2 = "../data/processed_laptopData.csv"  # Adjust the path to your second data preprocessing CSV file
        # if os.path.exists(preprocessing_data_path_2):
        #     df_preprocessing_2 = pd.read_csv(preprocessing_data_path_2)
        #     st.dataframe(df_preprocessing_2)  # Display the dataframe for Data Preprocessing 2
        # else:
        #     st.error(f"File {preprocessing_data_path_2} not found.")
        
        # Automatically display the WIA1007_Group_Assignment_Table.ipynb notebook for this stage
        notebook_path_2 = "WIA1007_Group_Assignment_Table.ipynb"  # Adjust the path accordingly
        if os.path.exists(notebook_path_2):
            notebook_html_2 = render_notebook(notebook_path_2)
            st.components.v1.html(notebook_html_2, height=800, scrolling=True)
        else:
            st.error(f"Notebook {notebook_path_2} not found.")
    with st.expander("Documentation", expanded=True):
        # Data Cleaning Modifications Section
        st.subheader("Data Cleaning Modifications")
        st.write("""
        Below are the key modifications made to the dataset for reporting and feature engineering:
        
        - **Commented Out One-Hot Encoding for Categorical Columns**: 
        One-hot encoding was removed to preserve the original structure of categorical columns.
        
        - **Formatted the ppi Column**: Rounded the ppi values to two decimal places.
        
        - **Converted and Formatted the Price Column**: Converted the Price column to USD and rounded the values.
        
        - **Removed Invalid Weight Entry**: A row with an unrealistic weight value was removed.
        
        - **Saved the Cleaned Data to a CSV File**: After cleaning, the dataset was saved to a new CSV file for easier handling.
        """)

        # Example of how the data was processed (this can be elaborated further)
        st.subheader("Code for Data Cleaning")
        code = '''
        # Commented out one-hot encoding for categorical columns
        # df = df.join(pd.get_dummies(df['Company'], dtype=int))
        # df = df.drop("Company", axis=1)
        # df = df.join(pd.get_dummies(df['TypeName'], dtype=int))
        # df = df.drop("TypeName", axis=1)
        # df = df.join(pd.get_dummies(df['Gpu_Brand'], dtype=int))
        # df = df.join(pd.get_dummies(df['OS'], dtype=int))

        # Formatted the ppi column
        df['ppi'] = df['ppi'].round(2)

        # Converted the Price column to USD
        df['Price'] = (df['Price'] * 0.01176).round(2)

        # Removed invalid Weight entry
        if (df['Weight'] == 0.0002).any():
            weight_row_index = df[df['Weight'] == 0.0002].index[0]
            df = df.drop(index=weight_row_index)

        # Saved the cleaned dataset to a new file
        new_file_path = '/content/drive/MyDrive/Colab Notebooks/WIA1007_Group_Assignment/laptopData_table_cleaned.csv'  
        df.to_csv(new_file_path, index=False)
        '''
        st.code(code, language='python')
        
        st.write("The cleaned dataset was saved as `laptopData_table_cleaned.csv`.")

        # Table 1: Data Properties Section
        st.subheader("Table 1: Data Properties")
        st.write("""
        This table provides the properties of the dataset including data type, minimum and maximum values, unique values, 
        and null values for each column. Additionally, the range of numerical columns and outlier ranges (if applicable) are included.
        
        Columns were cleaned by removing units, and the units were manually added back for context.
        """)


        # Table 2: Statistics Section
        st.subheader("Table 2: Statistics")
        st.write("""
        This table provides detailed statistical metrics for the dataset, including:
        - Frequency (for categorical variables)
        - Percentiles (25th, 50th, 75th)
        - Mean, Median, Mode
        - Standard Deviation, Variance
        - Skewness and Kurtosis
        """)

    
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
    
    with st.expander("Documentation", expanded=True):
        st.subheader("Documentation for EDA")
        st.write("""
        **Exploratory Data Analysis (EDA)**

        1. **Pie Chart of Percentage of Touchscreen vs. Non-Touchscreen Laptops**  
        For this part, we began the EDA by counting the number of laptops with and without touchscreens. We used the `value_counts()` function to calculate the totals for each category (touchscreen and non-touchscreen). After obtaining the totals, we created a pie chart to visually represent the percentage distribution of touchscreen versus non-touchscreen laptops. Additionally, we printed the exact counts for both categories to provide a precise numerical reference.

        2. **Percentage of Laptops by Company**  
        In the second part, we analyzed the distribution of laptops by company. Using the `value_counts()` function with the `normalize=True` parameter, we calculated the proportion of laptops for each company. These proportions were then multiplied by 100 to convert them into percentages. We visualized these percentages using a bar chart. Furthermore, a table was created to display the company names, the number of laptops, and their respective percentages.

        3. **Mean Laptop Price by Company**  
        Next, we focused on examining the mean prices of laptops from different companies. The `groupby()` function was used to calculate the mean price for each company. The results were sorted in descending order for better visualization. A bar chart was plotted to show the mean prices by company, allowing us to identify companies with the highest and lowest average prices. A table was also created to provide a detailed view of the average prices and the quantity of laptops for each company.

        4. **Percentage of Laptops by Type**  
        In this part, we aimed to determine the distribution of laptop types within the dataset. Using the `value_counts()` function with the `normalize=True` parameter, we calculated the percentage of laptops for each type. These percentages were visualized with a bar chart. A corresponding table was also created to show the type names, the number of laptops, and their respective percentages.

        5. **Mean PPI by Laptop Type**  
        We investigated the mean Pixels Per Inch (PPI) for each laptop type. Grouping the data by `TypeName` (the type name of the laptop), we calculated the mean PPI for each type. The results were visualized using a bar chart sorted in descending order. Additionally, a table was provided to display the average PPI values and the number of laptops for each type.

        6. **Laptops with IPS Panels vs. Without IPS Panels**  
        Finally, we analyzed the presence of IPS panels in laptops. This analysis was divided into two sub-parts:  
        - The count of laptops with and without IPS panels for each company.  
        - The count of laptops with and without IPS panels by type.  
        This provided insights into the distribution of IPS panels across different companies and laptop types.
        """)

    
elif page == "Model Building":
    with st.expander("Model Building", expanded=True):
        st.header("Model Building")
        st.write("Train models using algorithms such as Random Forest Regression.")

        notebook_path_3 = "Machine_Learning.ipynb"  # Adjust the path accordingly
        if os.path.exists(notebook_path_3):
            notebook_html_1 = render_notebook(notebook_path_3)
            st.components.v1.html(notebook_html_1, height=600, scrolling=True)
        else:
            st.error(f"Notebook {notebook_path_3} not found.")
        
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
    with st.expander("What is Random Forest Regressor?"):
            st.write("""
                **Random Forest Regressor** is an ensemble learning method that works by building multiple decision trees and
                averaging their predictions to improve accuracy and avoid overfitting.

                - **Ensemble Method**: Random Forest combines several decision trees to form a "forest." Each tree makes a prediction,
                  and the final prediction is the average of all trees' predictions.
                
                - **Why Random Forest?**: Random Forest is powerful because it can handle both regression and classification tasks,
                  and it is particularly good at handling large datasets with higher dimensionality.
                
                - **How it Works**: 
                  - Random Forest creates multiple decision trees using bootstrapped datasets (randomly sampled data with replacement).
                  - At each split in the tree, a random subset of features is chosen to improve the diversity of the trees.
                  - The final prediction is an average of the predictions made by each individual tree.
                
                - **Advantages**:
                  - Robust to overfitting, especially with large datasets.
                  - Handles missing values well.
                  - Can be used for both classification and regression tasks.
                  - Provides feature importance, which helps in understanding the influence of each feature on the predictions.

                - **Disadvantages**:
                  - Computationally expensive with large datasets.
                  - Less interpretable compared to a single decision tree.

                - **Tuning Hyperparameters**:
                  - `n_estimators`: The number of trees in the forest. More trees usually improve performance but at the cost of increased computation.
                  - `max_depth`: The maximum depth of each tree. Limiting the depth can prevent overfitting.
                  - `min_samples_split`: The minimum number of samples required to split a node. Larger values make the model more conservative.
                  - `min_samples_leaf`: The minimum number of samples required to be at a leaf node. Larger values prevent overfitting.

                **In summary**, Random Forest Regressor is a powerful and versatile model for regression tasks, providing good predictive accuracy without much tuning, while avoiding overfitting with its ensemble approach.
            """)

elif page == "Model Evaluation":
    with st.expander("Model Test", expanded=True):
        st.header("Model Evaluation")
        st.write("""
            Here we evaluate the performance of the model using multiple metrics such as MAE, MSE, RMSE, and R² score.
            We also allow you to input custom values for laptop features to predict the price using the trained model.
        """)

        # Load environment variables from .env file
        load_dotenv()

        # Fetch API key from environment variable
        API_KEY = os.getenv('EXCHANGE_API_KEY')
        # Load the saved model pipeline
        pipeline = joblib.load("trained_model_pipeline.pkl")
        fallback_rates = {
            "INR": 1,       # Indian Rupee (Base)
            "USD": 0.012,   # US Dollar
            "EUR": 0.011,   # Euro
            "GBP": 0.009,   # British Pound
            "AUD": 0.018,   # Australian Dollar
            "CAD": 0.016,   # Canadian Dollar
            "JPY": 1.62,    # Japanese Yen
            "CNY": 0.088,   # Chinese Yuan
            "SGD": 0.016,   # Singapore Dollar
            "AED": 0.044,   # UAE Dirham
            "SAR": 0.046,   # Saudi Riyal
            "CHF": 0.011,   # Swiss Franc
            "ZAR": 0.22,    # South African Rand
            "NZD": 0.019,   # New Zealand Dollar
            "HKD": 0.094,   # Hong Kong Dollar
            "KRW": 15.4,    # South Korean Won
            "SEK": 0.13,    # Swedish Krona
            "NOK": 0.13,    # Norwegian Krone
            "MXN": 0.21,    # Mexican Peso
            "BRL": 0.061,   # Brazilian Real
            "MYR": 0.054,   # Malaysian Ringgit
            "THB": 0.39,    # Thai Baht
            "IDR": 182.5,   # Indonesian Rupiah
            "PHP": 0.67,    # Philippine Peso
            "VND": 283.0,   # Vietnamese Dong
            "EGP": 0.37,    # Egyptian Pound
            "PKR": 3.0,     # Pakistani Rupee
            "BDT": 1.15,    # Bangladeshi Taka
            "LKR": 3.8,     # Sri Lankan Rupee
            "TWD": 0.37,    # Taiwanese Dollar
        }

        # Function to fetch exchange rates
        def fetch_exchange_rate():
            try:
                # Replace 'YOUR-API-KEY' with your actual API key from exchangerate-api.com
                url = f"https://v6.exchangerate-api.com/v6/{API_KEY}/latest/INR"  # Base currency is INR
                response = requests.get(url)
                response.raise_for_status()  # Raise an HTTPError for bad responses
                data = response.json()

                # Check if the API response is successful
                if data["result"] == "success":
                    return data["conversion_rates"]  # Return conversion rates dictionary
                else:
                    st.error("Error in API response. Using fallback rates.")
                    return fallback_rates  # Fallback rates for INR

            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching exchange rates: {e}. Using fallback rates.")
                return fallback_rates  # Fallback rates for INR

        # Fetch exchange rates
        rates = fetch_exchange_rate()

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

        # Add a dropdown for currency selection
        st.subheader("Currency Converter")
        currency = st.selectbox("Select Currency", options=list(rates.keys()), index=0)

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
            predicted_price_inr = pipeline.predict(user_input_df)[0]  # Prediction in INR

            # Convert the price to the selected currency
            conversion_rate = rates.get(currency, 1)  # Default to 1 if currency is INR
            predicted_price_converted = predicted_price_inr * conversion_rate

            # Display the price
            st.write(f"The predicted price of the laptop is: {currency} {predicted_price_converted:,.2f}")
        
else:
    st.write("Select a stage to view its content.")
    
    # Poster sharing feature
    st.subheader("Share the Poster")
    st.write("Click the button below to download or share the poster.")
    
    # Display the poster image
    st.image("../assets/Laptop Price Prediction.jpg", caption="Laptop Price Prediction Poster", use_column_width=True)

    # Provide a downloadable poster
    st.download_button(
        label="Download Poster",
        data=open("../assets/Laptop Price Prediction.jpg", "rb").read(),
        file_name="laptop_price_poster.jpg",
        mime="image/jpeg"
    )


