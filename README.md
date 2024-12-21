# Laptop Price Prediction Project

This project implements a machine learning model to predict the price of laptops based on various features such as the brand, screen size, processor, RAM, and storage. The model uses a Random Forest Regressor to estimate the price of a laptop based on the input features provided by the user.

## Project Overview

The project is divided into the following key stages:

1. **Data Collection**: A dataset of laptops with various specifications was collected from a publicly available Kaggle dataset.
2. **Data Preprocessing**: The raw data was cleaned and transformed to handle missing values, normalize numerical data, and encode categorical data.
3. **Exploratory Data Analysis (EDA)**: We performed an EDA to understand the dataset's structure, check for patterns, and visualize relationships between variables.
4. **Model Building**: A Random Forest Regressor was trained on the preprocessed data to predict the price of laptops.
5. **Model Evaluation**: The model's performance was evaluated using metrics such as MAE, MSE, RMSE, and R². The trained model was then deployed, allowing users to input custom features and predict laptop prices.

## Project Setup

### 1. Install Dependencies

To run the project, you need to install the following dependencies:

```bash
pip install -r requirements.txt
```

You can generate a `requirements.txt` file with the following content:

```plaintext
streamlit
pandas
scikit-learn
matplotlib
seaborn
joblib
nbconvert
```

### 2. Running the Application

To start the application, run the following command in the terminal:

```bash
streamlit run app.py
```

This will open a web browser where you can interact with the app and test the model. You can input different features of laptops such as screen size, RAM, weight, etc., and the app will predict the laptop's price.

## Features

- **Data Preprocessing**: Handles missing values, normalizes numerical data, and encodes categorical variables.
- **Exploratory Data Analysis (EDA)**: Visualize the dataset, understand distributions, correlations, and detect outliers.
- **Model Building**: A Random Forest Regressor is used to predict the price of laptops based on various features.
- **User Input Interface**: The application allows users to input their own laptop features through a web interface, and the trained model will predict the laptop's price.

### Key Features for Prediction:
- **Screen Size**: Size of the laptop screen in inches.
- **RAM**: Amount of RAM (in GB).
- **Weight**: Weight of the laptop (in kg).
- **Screen Resolution**: Screen dimensions in centimeters.
- **CPU Speed**: Processor speed in GHz.
- **Storage**: Amount of SSD, HDD, Flash, or Hybrid storage.
- **Brand & Model Information**: Brand, type, GPU brand, operating system.

## Model Evaluation

After training the Random Forest Regressor model, the following metrics were used to evaluate the model's performance:

- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

The model's performance can be tested using custom laptop feature values input by the user. After entering values such as screen size, RAM, and storage, the application will predict the laptop price.

## Example Usage

Once the application is running, you can input values for various features like:

- **Inches**: Select the screen size in inches.
- **RAM (GB)**: Select the RAM size.
- **Weight (kg)**: Select the weight of the laptop.
- **CPU Speed (GHz)**: Select the processor speed.
- **Storage Options**: Select the amount of SSD, HDD, and other storage.

When you click on the **"Predict Price"** button, the application will show the predicted price based on the trained model.

## Model Deployment

The trained model is saved as a `trained_model_pipeline.pkl` file, which includes both the preprocessing pipeline and the trained model. The saved model can be used to make predictions on new data without needing to retrain the model.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

