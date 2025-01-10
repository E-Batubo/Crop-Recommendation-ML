# **Crop-Recommendation-ML**
A machine learning model for recommending crops based on soil and weather conditions. This project helps farmers and agricultural experts determine the most suitable crops to grow for optimal yield.

## **Dataset**
The dataset used for this project was sourced from Kaggle, authored by [Atharva Ingle](https://www.kaggle.com/atharvaingle/crop-recommendation-dataset). It includes vital features such as soil composition, weather conditions, and other factors influencing crop growth.


## **Features**
- Predicts the best crop for cultivation based on:
  - Rainfall (mm)
  - Soil pH
  - Temperature (°C)
  - Nitrogen (N)
  - Phosphorus (P)
  - Potassium (K)
  - Relative Humidity (%)
- Implements a machine learning model trained using **scikit-learn**.
- Provides insights for improving crop yield and sustainability.

### Highlights:
- **Machine Learning Framework:** Implements a model trained using **scikit-learn**.
- **Practical Insights:** Provides actionable insights to enhance crop yield and ensure sustainability.
- **Scalable:** The approach can be adapted to similar datasets from different regions.

## **Project Structure**
The repository is organized as follows:
Crop-Recommendation-ML/ │ ├── data/ │ └── Crop_recommendation.csv # Dataset used for training and testing │ ├── notebooks/ │ └── Crop_Recommendation.ipynb # Jupyter notebook for exploration and modeling │ ├── scripts/ │ └── train_model.py # Python script for training the model │ ├── models/ │ └── crop_recommendation_model.pkl # Saved model for prediction │ └── scaler.pkl # Scaler for data standardization │ └── label_encoder.pkl # Label encoder for crop names │ ├── LICENSE # License for project usage ├── README.md # Project overview and documentation └── .gitignore # Files and folders to exclude from Git tracking


## **Getting Started**

### **Requirements**
To run the project, you need the following:
- **Python 3.8+**
- **Jupyter Notebook** (optional, for notebook-based work)
- Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `pickle`

### **Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/E-Batubo/Crop-Recommendation-ML.git
   cd Crop-Recommendation-ML
   
2. Install the required dependencies:
pip install -r requirements.txt

3. Run the Jupyter Notebook or Python scripts for training and prediction.

## **Usage**
1. **Run the Model**:
Use the Crop_Recommendation.ipynb notebook or the train_model.py script to train the model and make predictions.

2. **Make Predictions**:
Input soil and weather data (e.g., Nitrogen, rainfall, temperature) to predict the most suitable crop for cultivation.

3. **Interpret Results**:
View the recommended crop and interpret model insights.

## **Results**:
The model achieves an accuracy of **99.55%**. The feature importance chart shows that **rainfall** is the most significant factor in predicting the crop.

## **Contributing**
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a branch for your feature (git checkout -b feature-name).
3. Commit your changes (git commit -m 'Add feature-name').
4. Push to the branch (git push origin feature-name).
5. Open a pull request.
