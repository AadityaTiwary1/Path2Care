from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import string
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import numpy as np

app = Flask(__name__)

# Set up NLTK data
nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download required NLTK data
required_packages = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
for package in required_packages:
    try:
        nltk.data.find(f'tokenizers/{package}' if package == 'punkt' 
                      else f'corpora/{package}' if package in ['stopwords', 'wordnet']
                      else package)
    except LookupError:
        nltk.download(package, download_dir=nltk_data_dir)

# Text preprocessing function
def preprocess_text(text):
    try:
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        text = text.lower()
        
        try:
            tokens = word_tokenize(text)
        except LookupError:
            tokens = text.split()
        
        tokens = [token for token in tokens if token not in string.punctuation and not token.isnumeric()]
        
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
        except LookupError:
            pass
        
        try:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
        except LookupError:
            pass
        
        return ' '.join(tokens)
    except Exception as e:
        print(f"Warning: Error processing text: {str(e)}")
        return ""

# Load and prepare the model
def initialize_model():
    global tfidf, rf_model
    
    # Load the dataset
    df = pd.read_csv('Symptom2Disease.csv')
    
    # Preprocess text
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(max_features=1000)
    X = tfidf.fit_transform(df['processed_text'])
    y = df['label']
    
    # Train the model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    
    print("Model initialized successfully!")

# Initialize the model when the application starts
initialize_model()

# Load skin disease model
def load_skin_disease_model():
    global skin_model, skin_class_names
    try:
        skin_model = load_model('skin_disease_model.h5')
        print(f"Model loaded successfully. Model summary:")
        print(f"Input shape: {skin_model.input_shape}")
        print(f"Output shape: {skin_model.output_shape}")
        
        # Define class names based on the training data
        skin_class_names = [
            '1. Eczema 1677',
            '10. Warts Molluscum and other Viral Infections - 2103',
            '2. Melanoma 15.75k',
            '3. Atopic Dermatitis - 1.25k',
            '4. Basal Cell Carcinoma (BCC) 3323',
            '5. Melanocytic Nevi (NV) - 7970',
            '6. Benign Keratosis-like Lesions (BKL) 2624',
            '7. Psoriasis pictures Lichen Planus and related diseases - 2k',
            '8. Seborrheic Keratoses and other Benign Tumors - 1.8k',
            '9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k'
        ]
        
        # Verify model output matches class names
        expected_classes = skin_model.output_shape[-1]
        print(f"Model expects {expected_classes} classes, we have {len(skin_class_names)} class names")
        
        if expected_classes != len(skin_class_names):
            print(f"WARNING: Model output classes ({expected_classes}) don't match class names ({len(skin_class_names)})")
        
        print("Skin disease model loaded successfully!")
    except Exception as e:
        print(f"Error loading skin disease model: {str(e)}")
        skin_model = None
        skin_class_names = []

# Initialize skin disease model
load_skin_disease_model()

# Add these imports for TomTom API integration
import requests
from math import radians, sin, cos, sqrt, atan2
import os

# TomTom API key (stored securely in the app config)
# In production, use environment variables: os.getenv('TOMTOM_API_KEY')
app.config['TOMTOM_API_KEY'] = '0TQGxzg4nBU5qP4Du9YkaQAkKJ1OvNnd'

@app.route('/')
def home():
    return render_template('index.html')

def create_tree_distribution_plot(symptoms):
    # Analyze predictions from individual trees
    processed_symptoms = preprocess_text(symptoms)
    symptoms_vector = tfidf.transform([processed_symptoms])
    
    # Get predictions from all trees
    tree_predictions = []
    for tree in rf_model.estimators_:
        pred = tree.predict(symptoms_vector)[0]
        tree_predictions.append(pred)
    
    # Calculate distribution
    unique_preds, counts = np.unique(tree_predictions, return_counts=True)
    distribution = dict(zip(unique_preds, counts))
    distribution = dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))
    
    # Create plot
    plt.figure(figsize=(10, 5))
    plt.bar(distribution.keys(), distribution.values())
    plt.title('Distribution of Predictions Across Decision Trees')
    plt.xlabel('Predicted Disease')
    plt.ylabel('Number of Trees')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()
    
    return base64.b64encode(img.getvalue()).decode()

def create_feature_importance_plot(symptoms_vector):
    # Get feature importance for this prediction
    feature_importance = pd.DataFrame({
        'feature': tfidf.get_feature_names_out(),
        'importance': rf_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Plot top 10 features
    plt.figure(figsize=(10, 5))
    plt.bar(feature_importance['feature'][:10], feature_importance['importance'][:10])
    plt.title('Top 10 Most Important Terms for Disease Prediction')
    plt.xlabel('Terms')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()
    
    return base64.b64encode(img.getvalue()).decode()

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the symptoms from the form
        symptoms = request.form['symptoms']
        
        # Preprocess and vectorize the input
        processed_symptoms = preprocess_text(symptoms)
        symptoms_vector = tfidf.transform([processed_symptoms])
        
        # Get prediction and probabilities
        prediction = rf_model.predict(symptoms_vector)[0]
        probabilities = rf_model.predict_proba(symptoms_vector)[0]
        confidence = max(probabilities) * 100
        
        # Get top 5 predictions
        top_5_indices = np.argsort(probabilities)[-5:][::-1]
        top_5_predictions = []
        chart_labels = []
        chart_values = []
        
        for idx in top_5_indices:
            disease = rf_model.classes_[idx]
            prob = probabilities[idx] * 100
            top_5_predictions.append((disease, prob))
            chart_labels.append(disease)
            chart_values.append(prob)
        
        # Generate matplotlib visualizations
        tree_dist_plot = create_tree_distribution_plot(symptoms)
        feature_imp_plot = create_feature_importance_plot(symptoms_vector)
        
        prediction_result = {
            'top_prediction': prediction,
            'confidence': confidence,
            'top_5_predictions': top_5_predictions,
            'chart_labels': chart_labels,
            'chart_values': chart_values,
            'tree_distribution_plot': tree_dist_plot,
            'feature_importance_plot': feature_imp_plot
        }
        
        return render_template('predict.html', prediction=prediction_result)
    
    return render_template('predict.html')

@app.route('/skin_diseases', methods=['GET', 'POST'])
def skin_diseases():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('skin_diseases.html', error="No image uploaded")
        
        file = request.files['image']
        if file.filename == '':
            return render_template('skin_diseases.html', error="No image selected")
        
        if file:
            try:
                # Read and preprocess the image
                img = Image.open(file.stream)
                img = img.resize((150, 150))  # Resize to 150x150 for model input
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0  # Normalize
                
                # Make prediction
                if skin_model is not None:
                    predictions = skin_model.predict(img_array)
                    predicted_class = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_class] * 100
                    
                    # Debug: Print prediction info
                    print(f"Model predictions shape: {predictions.shape}")
                    print(f"Predicted class index: {predicted_class}")
                    print(f"Confidence: {confidence:.2f}%")
                    print(f"Available class names: {len(skin_class_names)}")
                    
                    # Get top 5 predictions
                    top_5_indices = np.argsort(predictions[0])[-5:][::-1]
                    top_5_predictions = []
                    
                    for idx in top_5_indices:
                        if idx < len(skin_class_names):
                            disease = skin_class_names[idx]
                            prob = predictions[0][idx] * 100
                            top_5_predictions.append((disease, prob))
                            print(f"Top prediction {len(top_5_predictions)}: {disease} - {prob:.2f}%")
                        else:
                            print(f"Warning: Index {idx} out of range for class names")
                    
                    print(f"Total top 5 predictions: {len(top_5_predictions)}")
                    
                    # Convert image to base64 for display
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='JPEG')
                    img_str = base64.b64encode(img_buffer.getvalue()).decode()
                    
                    result = {
                        'image': img_str,
                        'predicted_class': skin_class_names[predicted_class],
                        'confidence': confidence,
                        'top_5_predictions': top_5_predictions
                    }
                    
                    return render_template('skin_diseases.html', result=result)
                else:
                    return render_template('skin_diseases.html', error="Model not loaded properly. Please check the console for details.")
                    
            except Exception as e:
                print(f"Error in skin disease prediction: {str(e)}")
                import traceback
                traceback.print_exc()
                return render_template('skin_diseases.html', error=f"Error processing image: {str(e)}")
    
    return render_template('skin_diseases.html')

@app.route('/test_model')
def test_model():
    """Debug route to test the skin disease model"""
    if skin_model is None:
        return jsonify({'error': 'Model not loaded'})
    
    try:
        # Create a dummy image (all zeros) for testing
        dummy_img = np.zeros((1, 150, 150, 3))
        predictions = skin_model.predict(dummy_img)
        
        return jsonify({
            'model_loaded': True,
            'input_shape': skin_model.input_shape,
            'output_shape': skin_model.output_shape,
            'num_classes': len(skin_class_names),
            'class_names': skin_class_names,
            'dummy_prediction_shape': predictions.shape,
            'dummy_predictions': predictions[0].tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/hospitals')
def hospitals():
    # Get unique diseases from the model
    diseases = sorted(rf_model.classes_)
    return render_template('hospital_map.html', diseases=diseases, tomtom_api_key=app.config['TOMTOM_API_KEY'])

@app.route('/geocode')
def geocode():
    location = request.args.get('location', '')
    if not location:
        return jsonify({'success': False, 'message': 'Location is required'})
    
    # Use TomTom API to geocode the location
    url = f"https://api.tomtom.com/search/2/geocode/{location}.json"
    params = {
        'key': app.config['TOMTOM_API_KEY'],
        'limit': 1
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        
        if 'results' in data and len(data['results']) > 0:
            position = data['results'][0]['position']
            return jsonify({
                'success': True,
                'location': {
                    'lat': position['lat'],
                    'lng': position['lon']
                }
            })
        else:
            return jsonify({'success': False, 'message': 'Location not found. Please try a different location or check the spelling.'})
    except requests.exceptions.RequestException as e:
        print(f"TomTom API geocoding request failed: {str(e)}")
        return jsonify({'success': False, 'message': f'TomTom API error: {str(e)}'})
    except Exception as e:
        print(f"Error processing geocoding response: {str(e)}")
        return jsonify({'success': False, 'message': f'Error processing response: {str(e)}'})

@app.route('/search_hospitals')
def search_hospitals():
    lat = request.args.get('lat', type=float)
    lng = request.args.get('lng', type=float)
    specialization = request.args.get('specialization', '')
    
    if lat is None or lng is None:
        return jsonify({'success': False, 'message': 'Latitude and longitude are required'})
    
    # Initialize hospitals data
    hospitals_df = None
    
    # Try to load hospitals data from CSV if it exists
    try:
        if os.path.exists('HospitalsInIndia.csv'):
            hospitals_df = pd.read_csv('HospitalsInIndia.csv')
            print("Loaded hospitals data from CSV")
        else:
            print("HospitalsInIndia.csv not found, using TomTom API only")
    except Exception as e:
        print(f"Error loading hospitals CSV: {str(e)}")
        hospitals_df = None
    
    # Use TomTom API to search for nearby hospitals
    url = "https://api.tomtom.com/search/2/nearbySearch/.json"
    params = {
        'key': app.config['TOMTOM_API_KEY'],
        'lat': lat,
        'lon': lng,
        'radius': 10000,  # 10km radius
        'limit': 20,
        'categorySet': '7321',  # Category for hospitals
    }
    
    nearby_hospitals = []
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        
        if 'results' in data:
            for result in data['results']:
                hospital = {
                    'name': result['poi']['name'],
                    'latitude': result['position']['lat'],
                    'longitude': result['position']['lon'],
                    'address': result.get('address', {}).get('freeformAddress', 'Address not available'),
                    'distance': result['dist'] / 1000  # Convert to kilometers
                }
                nearby_hospitals.append(hospital)
            print(f"Found {len(nearby_hospitals)} hospitals via TomTom API")
        else:
            print("No results found in TomTom API response")
            
    except requests.exceptions.RequestException as e:
        print(f"TomTom API request failed: {str(e)}")
        return jsonify({'success': False, 'message': f'TomTom API error: {str(e)}'})
    except Exception as e:
        print(f"Error processing TomTom API response: {str(e)}")
        return jsonify({'success': False, 'message': f'Error processing API response: {str(e)}'})
    
    # If we have CSV data and want to supplement with it
    if hospitals_df is not None and (not nearby_hospitals or specialization):
        print("Supplementing with CSV data...")
        # Calculate distances to hospitals in our database
        for _, row in hospitals_df.iterrows():
            # Skip hospitals without address information
            if pd.isna(row['City']) or pd.isna(row['LocalAddress']):
                continue
            
            # Use TomTom API to geocode the hospital address
            address = f"{row['LocalAddress']}, {row['City']}, {row['State']}, India"
            geocode_url = f"https://api.tomtom.com/search/2/geocode/{address}.json"
            geocode_params = {
                'key': app.config['TOMTOM_API_KEY'],
                'limit': 1
            }
            
            try:
                geocode_response = requests.get(geocode_url, params=geocode_params)
                geocode_response.raise_for_status()
                geocode_data = geocode_response.json()
                
                if 'results' in geocode_data and len(geocode_data['results']) > 0:
                    position = geocode_data['results'][0]['position']
                    hospital_lat = position['lat']
                    hospital_lng = position['lon']
                    
                    # Calculate distance using Haversine formula
                    R = 6371  # Earth radius in kilometers
                    dlat = radians(hospital_lat - lat)
                    dlng = radians(hospital_lng - lng)
                    a = sin(dlat/2)**2 + cos(radians(lat)) * cos(radians(hospital_lat)) * sin(dlng/2)**2
                    c = 2 * atan2(sqrt(a), sqrt(1-a))
                    distance = R * c
                    
                    if distance <= 10:  # Within 10km
                        hospital = {
                            'name': row['Hospital'],
                            'latitude': hospital_lat,
                            'longitude': hospital_lng,
                            'address': f"{row['LocalAddress']}, {row['City']}, {row['State']}",
                            'distance': distance
                        }
                        nearby_hospitals.append(hospital)
            except Exception as e:
                print(f"Error geocoding hospital: {str(e)}")
                continue
    
    # Sort hospitals by distance
    nearby_hospitals.sort(key=lambda x: x['distance'])
    
    if not nearby_hospitals:
        return jsonify({
            'success': False,
            'message': 'No hospitals found in the specified area. Try expanding your search radius or check your location.'
        })
    
    return jsonify({
        'success': True,
        'hospitals': nearby_hospitals[:20]  # Limit to 20 results
    })

if __name__ == '__main__':
    app.run(debug=True)
