from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and preprocessors
try:
    model = joblib.load('house_price_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le_neighborhood = joblib.load('le_neighborhood.pkl')
    feature_names = joblib.load('feature_names.pkl')
    neighborhood_classes = joblib.load('neighborhood_classes.pkl')
    print("✓ Model and preprocessors loaded successfully")
    print(f"✓ Available neighborhoods: {neighborhood_classes}")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    model = None
    scaler = None
    le_neighborhood = None
    feature_names = None
    neighborhood_classes = []

@app.route('/')
def home():
    return render_template('index.html', neighborhoods=neighborhood_classes)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({
                'error': 'Model not loaded. Please check server configuration.'
            }), 500
        
        # Get data from form
        overall_qual = int(request.form['overall_qual'])
        gr_liv_area = float(request.form['gr_liv_area'])
        total_bsmt_sf = float(request.form['total_bsmt_sf'])
        garage_cars = int(request.form['garage_cars'])
        year_built = int(request.form['year_built'])
        neighborhood = request.form['neighborhood']
        
        # Validate inputs
        if overall_qual < 1 or overall_qual > 10:
            return jsonify({'error': 'Overall Quality must be between 1 and 10'}), 400
        
        if year_built < 1800 or year_built > 2025:
            return jsonify({'error': 'Year Built must be between 1800 and 2025'}), 400
        
        # Encode neighborhood
        try:
            neighborhood_encoded = le_neighborhood.transform([neighborhood])[0]
        except ValueError:
            return jsonify({'error': f'Invalid neighborhood: {neighborhood}'}), 400
        
        # Create DataFrame with exact feature order
        input_data = pd.DataFrame({
            'OverallQual': [overall_qual],
            'GrLivArea': [gr_liv_area],
            'TotalBsmtSF': [total_bsmt_sf],
            'GarageCars': [garage_cars],
            'YearBuilt': [year_built],
            'Neighborhood': [neighborhood_encoded]
        })
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Prepare response
        result = {
            'prediction': float(prediction),
            'formatted_price': f"${prediction:,.2f}",
            'inputs': {
                'Overall Quality': overall_qual,
                'Living Area': f"{gr_liv_area:,.0f} sq ft",
                'Basement Area': f"{total_bsmt_sf:,.0f} sq ft",
                'Garage Cars': garage_cars,
                'Year Built': year_built,
                'Neighborhood': neighborhood
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'neighborhoods_available': len(neighborhood_classes) if neighborhood_classes else 0
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)