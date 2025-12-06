🚦 Traffic Accident Severity Prediction
ML-Powered Streamlit Dashboard Using CatBoost & Exploratory Analytics

This project predicts road accident severity using machine learning and provides an interactive Streamlit dashboard for exploring accident patterns. It uses a cleaned and engineered version of the RTA Dataset and applies the CatBoost Classifier, which handles categorical data efficiently and produces explainable predictions.

The dashboard includes:
🔍 Interactive Filters
📊 Accident Analysis Charts
🤖 AI-based Severity Prediction
📈 Feature Importance Insights

📂 Project Structure
Accident_Severity_App/
│
├── app.py                # Streamlit dashboard UI
├── train_model.py        # Script that trains CatBoost model
├── accident_model.pkl    # Saved trained model
├── RTA_Dataset.csv       # Dataset used for training
├── charts/               # Auto-generated EDA charts
└── requirements.txt      # Dependencies for deployment

🚀 Features
1️⃣ Accident Severity Prediction

Predict whether an accident will lead to:

Fatal injury
Serious injury
Slight injury
Based on:
Weather
Day of Week
Driving Experience
Road Conditions
Vehicle Count & Casualties
Hour of the Day

2️⃣ Exploratory Data Analysis (EDA)

The dashboard includes powerful visualizations such as:
Severity by weather
Severity by hour
Severity by day
Accident count by vehicle involvement

3️⃣ Model Used — CatBoost Classifier

CatBoost is chosen because:
It handles categorical data automatically
Reduces preprocessing effort
Provides high accuracy
Generates feature importance for explainability

4️⃣ Streamlit Dashboard

Interactive sections include:
Filters Panel
Overview Metrics
Environment & Road Analysis
Human & Vehicle Analysis
Prediction Module

Model Insights Tab

Everything updates in real time as users apply filters.

⚙️ Installation
1. Clone the repository
git clone https://github.com/your-username/Accident_Severity_App.git
cd Accident_Severity_App

2. Install dependencies
pip install -r requirements.txt

3. Run the Streamlit app
streamlit run app.py

🧠 Model Training

The training is done using:

CatBoostClassifier

MultiClass loss

500 boosting iterations

Auto-handled categorical encoding

Hyperparameters optimized for accuracy

To retrain:

python train_model.py


This regenerates:

accident_model.pkl

EDA charts inside /charts folder

☁️ Deployment (Streamlit Cloud)

Push the entire project to GitHub

Go to https://share.streamlit.io

Select your repo

Choose app.py as the entry file

Click Deploy

You’ll get a public link like:

https://your-app-name.streamlit.app

📊 Sample Output (Screenshots)

Confusion Matrix

Feature Importance Graph

Severity Prediction Output

Dashboard Visualizations

(Add your screenshots in GitHub if needed.)

🧪 Tech Stack
Component	Technology
Programming	Python
ML Model	CatBoost Classifier
Dashboard	Streamlit
Visualization	Matplotlib, Seaborn, Plotly
Data Handling	Pandas, NumPy
🔍 Key Advantages of This System

🔥 Handles both numeric and categorical data

📉 Very low preprocessing needed

📈 High accuracy compared to traditional ML

📊 Deep accident analytics

🤖 Real-time model prediction

🧠 Explainable AI (Feature Importance)

🌐 Fully deployable as a Web App

📝 License

This project is for educational and research use.

👨‍💻 Author
CHIYAMALA DEVI
Accident Severity Prediction System 
