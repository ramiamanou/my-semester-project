import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


model_filename = 'cnn_model_scikit.joblib'
model = joblib.load(model_filename)

# Load the scaler
scaler_filename = 'scaler.save'
scaler = joblib.load(scaler_filename)


def adjust_price_based_on_location_and_direction(predicted_price, location, direction):
    direction_multipliers = {
        "East and South": 1.10,
        "East and North": 1.05,
        "East and West": 1.00,
        "South and North": 0.95,
        "South and West": 0.95,
        "North and West": 0.85,
        "South and East": 1.10,
        "East": 1.05,
        "South": 1.05,
        "North": 0.90,
        "West": 0.95
    }

    location_multipliers = {
    'Al-Amara Al-Juwani': 18.5, 'Bab Touma': 19.5, 'Al-Qaymariyya': 21.2, 'Al-Hamidiyya': 24.1, 
    'Al-Hariqa': 24.11, 'Al-Marja': 26.1, 'Al-Amin': 11.8, 'Miathanet Al-Sham': 12.1, 
    'Shaghour Juwani': 11.76, 'Sarouja': 18.5, 'Al-Amara Al-Barani': 14.6, 'Al-Qasaa': 18.9, 
    'Al-Adawi': 18.7, 'Al-Qusour': 18.8, 'Baghdad Street': 22.4, 'Qanawat': 15.7, 
    'Al-Hijaz': 23.5, 'Al-Baramka': 24.0, 'Bab Al-Jabiyah': 17.6, 'Bab Srija': 17.92, 
    'Khalid Ibn Al-Walid Street': 16.3, 'Al-Sharbiyaat': 15.92, 'Al-Souyqa': 17.82, 
    'Al-Mujtahid': 17.85, 'Ghaws': 15.0, 'Al-Zahira Al-Jadida': 12.29, 'Al-Zahira Al-Qadima': 12.52, 
    'Al-Haqla': 13.21, 'Abu Habl': 12.4, 'Al-Qaa': 11.49, 'Bab Musalla': 15.33, 
    'Shaghour Barani': 10.89, 'Bab Sharqi': 22.24, 'Ibn Asakir': 14.12, 'Al-Nidal': 14.22, 
    'Al-Douyla': 5.8, 'Al-Zuhur': 4.12, 'Al-Tadamon': 3.85, 'Sayyida Aisha': 5.0, 
    'Al-Qadam Al-Asali': 3.9, 'Al-Qadam Sharqi': 3.4, 'Kafr Sousa Al-Balad': 21.4, 
    'Tanzim Kafr Sousa': 42.66, 'Al-Luwan': 5.2, 'Al-Mazzeh Sheikh Saad': 13.33, 
    'Mazzeh Jabal': 19.55, 'Mazzeh Eastern Villas': 24.52, 'Mazzeh Western Villas': 24.6, 
    'Mazzeh 86': 4.1, 'Moujamaa Dummar': 17.21, 'Eastern Dummar': 3.95, 'Western Dummar': 4.8, 
    'Barzeh Al-Balad': 3.35, 'Masakin Barzeh': 10.2, 'Pre-fabricated Barzeh': 12.3, 
    'Asha Al-Warour': 2.07, 'Al-Qaboun': 5.11, 'Ruken Al-Din': 22.64, 'Asad Al-Din': 21.2, 
    'Al-Faihaa': 22.55, 'Al-Salhiyeh': 24.2, 'Sheikh Muhyi Al-Din': 15.2, 
    'Al-Mazraa': 24.2, 'Al-Jisr Al-Abyad': 27.3, 'Arnos': 27.91, 'Al-Muhajireen': 21.51, 
    'Al-Rawda': 24.3, 'Abu Rummaneh': 36.4, 'Al-Malki': 34.9, 'Al-Afeef': 29.2
    }


    # Get multipliers with default values
    location_multiplier = location_multipliers.get(location, 1.0)
    direction_multiplier = direction_multipliers.get(direction, 1.0)

    # Calculate final price
    final_price = predicted_price * location_multiplier * direction_multiplier
    return final_price

def combined_model_predict(features):
    # Prepare the features for the model
    cnn_features = np.array([[features['area'], features['bedrooms'], features['bathrooms'], 
                              features['view'], features['condition'], features['grade']]])
    
    # Scale the features using the existing scaler
    cnn_features_scaled = scaler.transform(cnn_features)
    
    # Reshape the features to match the CNN input shape (samples, timesteps, features)
    cnn_features_scaled_reshaped = np.expand_dims(cnn_features_scaled, axis=2)
    
    # Predict using the CNN model
    predicted_price = model.predict(cnn_features_scaled_reshaped)
    
    # Use the first prediction and multiply by the offset
    offset = 20000  # Replace with your specific value if necessary
    predicted_price_adjusted = predicted_price[0][0] * offset
    
    # Adjust price based on location and direction
    final_price = adjust_price_based_on_location_and_direction(predicted_price_adjusted, features['location'], features['direction'])
    
    return final_price

# if __name__ == "__main__":
#     input_data = json.loads(sys.argv[1])
#     final_price = combined_model_predict(input_data)
#     print(f"{final_price:.0f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_data = json.loads(sys.argv[1])
    else:
        # Default input data for testing
        input_data = {
            "area": 180,
            "bedrooms": 2,
            "bathrooms": 2,
            "view": 5,
            "condition": 10,
            "grade": 11,
            "location": "Tanzim Kafr Sousa",
            "direction": "East and South"
        }
    final_price = combined_model_predict(input_data)
    print(f"{final_price:.0f}")