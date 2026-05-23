from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

app = Flask(__name__)

class InsuranceCostPredictor:
    def __init__(self):
        self.model = None
        self.encoders = {}
        self.is_trained = False
        
    def prepare_data(self, df):
        """Prepare data for training"""
        # Create copies for encoding
        data = df.copy()
        
        # Encode categorical variables
        categorical_columns = ['sex', 'smoker', 'region']
        
        for col in categorical_columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                data[col] = self.encoders[col].fit_transform(data[col])
            else:
                data[col] = self.encoders[col].transform(data[col])
        
        return data
    
    def train_model(self):
        """Train the insurance cost prediction model"""
        try:
            # Load dataset
            if not os.path.exists('medical_insurance_dataset.csv'):
                return False, "Dataset not found. Please generate dataset first."
            
            df = pd.read_csv('medical_insurance_dataset.csv')
            
            # Prepare data
            data = self.prepare_data(df)
            
            # Split features and target
            X = data.drop('charges', axis=1)
            y = data['charges']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.is_trained = True
            
            # Save model and encoders
            joblib.dump(self.model, 'insurance_model.pkl')
            joblib.dump(self.encoders, 'encoders.pkl')
            
            return True, f"Model trained successfully! MAE: ${mae:.2f}, R2 Score: {r2:.3f}"
        
        except Exception as e:
            return False, f"Error training model: {str(e)}"
    
    def load_model(self):
        """Load pre-trained model"""
        try:
            if os.path.exists('insurance_model.pkl') and os.path.exists('encoders.pkl'):
                self.model = joblib.load('insurance_model.pkl')
                self.encoders = joblib.load('encoders.pkl')
                self.is_trained = True
                return True
            return False
        except:
            return False
    
    def predict(self, age, sex, bmi, children, smoker, region):
        """Make prediction for insurance cost"""
        if not self.is_trained:
            return None, "Model not trained yet"
        
        try:
            # Create input dataframe
            input_data = pd.DataFrame({
                'age': [age],
                'sex': [sex],
                'bmi': [bmi],
                'children': [children],
                'smoker': [smoker],
                'region': [region]
            })
            
            # Encode categorical variables
            for col in ['sex', 'smoker', 'region']:
                input_data[col] = self.encoders[col].transform(input_data[col])
            
            # Make prediction
            prediction = self.model.predict(input_data)[0]
            
            return prediction, "Success"
        
        except Exception as e:
            return None, f"Error making prediction: {str(e)}"

# Initialize predictor
predictor = InsuranceCostPredictor()
predictor.load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_dataset', methods=['POST'])
def generate_dataset():
    try:
        # Import the dataset generator
        from generate_dataset import generate_medical_insurance_dataset
        
        # Generate dataset
        generate_medical_insurance_dataset(2000)
        
        return jsonify({
            'success': True,
            'message': 'Dataset generated successfully!'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error generating dataset: {str(e)}'
        })

@app.route('/train_model', methods=['POST'])
def train_model():
    success, message = predictor.train_model()
    return jsonify({
        'success': success,
        'message': message
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        age = int(data['age'])
        sex = data['sex']
        bmi = float(data['bmi'])
        children = int(data['children'])
        smoker = data['smoker']
        region = data['region']
        
        # Validate inputs
        if not (18 <= age <= 100):
            return jsonify({'success': False, 'message': 'Age must be between 18 and 100'})
        
        if not (15 <= bmi <= 50):
            return jsonify({'success': False, 'message': 'BMI must be between 15 and 50'})
        
        if not (0 <= children <= 10):
            return jsonify({'success': False, 'message': 'Children must be between 0 and 10'})
        
        prediction, message = predictor.predict(age, sex, bmi, children, smoker, region)
        
        if prediction is not None:
            return jsonify({
                'success': True,
                'prediction': round(prediction, 2),
                'message': 'Prediction successful'
            })
        else:
            return jsonify({
                'success': False,
                'message': message
            })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

@app.route('/dataset_info')
def dataset_info():
    try:
        if os.path.exists('medical_insurance_dataset.csv'):
            df = pd.read_csv('medical_insurance_dataset.csv')
            info = {
                'exists': True,
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'sample_data': df.head(5).to_dict('records'),
                'statistics': df.describe().to_dict()
            }
        else:
            info = {'exists': False}
        
        return jsonify(info)
    except Exception as e:
        return jsonify({'exists': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)