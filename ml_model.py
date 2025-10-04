import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.svm import SVC
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import string
import os
import io
import base64

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

class DiseasePredictor:
    def __init__(self, data_path='Symptom2Disease.csv'):
        self.data_path = data_path
        self.tfidf = None
        self.rf_model = None
        self.linear_model = None
        self.bayesian_model = None
        self.svm_model = None
        self.label_mapping = None
        self.reverse_mapping = None
        
    def load_data(self):
        # Load the dataset
        df = pd.read_csv(self.data_path)
        
        # Preprocess text
        df['processed_text'] = df['text'].apply(preprocess_text)
        
        # Create label mappings for regression models
        self.label_mapping = {label: i for i, label in enumerate(df['label'].unique())}
        self.reverse_mapping = {i: label for label, i in self.label_mapping.items()}
        
        return df
    
    def train_models(self):
        # Load and preprocess data
        df = self.load_data()
        
        # Create TF-IDF vectorizer
        self.tfidf = TfidfVectorizer(max_features=1000)
        X = self.tfidf.fit_transform(df['processed_text'])
        y = df['label']
        y_numeric = df['label'].map(self.label_mapping)
        
        # Train Random Forest model
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(X, y)
        
        # Train Linear Regression model
        self.linear_model = LinearRegression()
        self.linear_model.fit(X.toarray(), y_numeric)  # Convert sparse to dense
        
        # Train Bayesian Regression model
        self.bayesian_model = BayesianRidge()
        self.bayesian_model.fit(X.toarray(), y_numeric)  # Convert sparse to dense
        
        # Train SVM model
        self.svm_model = SVC(probability=True, random_state=42)
        self.svm_model.fit(X, y)
        
        print("All models trained successfully!")
    
    def create_model_comparison_chart(self, predictions):
        """Create a bar chart comparing predictions from different models"""
        plt.figure(figsize=(10, 6))
        
        # Extract model names and their predictions
        models = list(predictions.keys())
        values = [1 if model != 'confidence_scores' else 0 for model in models]
        
        # Create bar chart
        bars = plt.bar(models, values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
        
        # Add prediction labels on top of bars
        for i, bar in enumerate(bars):
            if models[i] != 'confidence_scores':
                plt.text(i, bar.get_height() + 0.05, 
                        f"{predictions[models[i]]}", 
                        ha='center', va='bottom', rotation=0, fontsize=9)
        
        # Add confidence scores where available
        if 'confidence_scores' in predictions:
            for i, model in enumerate(models):
                if model in predictions['confidence_scores']:
                    plt.text(i, 0.5, 
                            f"{predictions['confidence_scores'][model]:.1f}%", 
                            ha='center', va='center', color='white', fontweight='bold')
        
        plt.title('Disease Predictions by Different Models')
        plt.ylabel('Prediction Made')
        plt.ylim(0, 1.5)  # Set y-axis limit
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert plot to base64 image
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode()
    
    def create_probability_chart(self, top_predictions):
        """Create a horizontal bar chart for top disease probabilities"""
        diseases = [pred[0] for pred in top_predictions]
        probabilities = [pred[1] for pred in top_predictions]
        
        # Reverse lists to show highest probability at the top
        diseases.reverse()
        probabilities.reverse()
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(diseases, probabilities, color='#3498db')
        
        # Add percentage labels
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                    f"{probabilities[i]:.1f}%", 
                    va='center')
        
        plt.title('Top Disease Probabilities')
        plt.xlabel('Probability (%)')
        plt.xlim(0, 105)  # Set x-axis limit with some padding
        plt.tight_layout()
        
        # Convert plot to base64 image
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode()
        
    def predict(self, symptoms_text):
        # Preprocess and vectorize input
        processed_text = preprocess_text(symptoms_text)
        vector = self.tfidf.transform([processed_text])
        
        # Get predictions from all models
        results = {}
        
        # Random Forest prediction
        rf_prediction = self.rf_model.predict(vector)[0]
        rf_probabilities = self.rf_model.predict_proba(vector)[0]
        rf_confidence = max(rf_probabilities) * 100
        
        # SVM prediction
        svm_prediction = self.svm_model.predict(vector)[0]
        svm_probabilities = self.svm_model.predict_proba(vector)[0]
        svm_confidence = max(svm_probabilities) * 100
        
        # Linear Regression prediction
        lr_prediction_numeric = self.linear_model.predict(vector.toarray())[0]
        lr_prediction = self.reverse_mapping[round(lr_prediction_numeric)]
        
        # Bayesian Regression prediction
        bayesian_prediction_numeric = self.bayesian_model.predict(vector.toarray())[0]
        bayesian_prediction = self.reverse_mapping[round(bayesian_prediction_numeric)]
        
        # Get top 5 predictions from Random Forest
        top_5_indices = np.argsort(rf_probabilities)[-5:][::-1]
        top_5_predictions = []
        
        for idx in top_5_indices:
            disease = self.rf_model.classes_[idx]
            prob = rf_probabilities[idx] * 100
            top_5_predictions.append((disease, prob))
        
        # Create prediction summary for visualization
        prediction_summary = {
            'Random Forest': rf_prediction,
            'SVM': svm_prediction,
            'Linear Regression': lr_prediction,
            'Bayesian Ridge': bayesian_prediction,
            'confidence_scores': {
                'Random Forest': rf_confidence,
                'SVM': svm_confidence
            }
        }
        
        # Generate charts
        model_comparison_chart = self.create_model_comparison_chart(prediction_summary)
        probability_chart = self.create_probability_chart(top_5_predictions)
        
        # Compile results
        results = {
            'random_forest': {
                'prediction': rf_prediction,
                'confidence': rf_confidence
            },
            'svm': {
                'prediction': svm_prediction,
                'confidence': svm_confidence
            },
            'linear_regression': {
                'prediction': lr_prediction
            },
            'bayesian_regression': {
                'prediction': bayesian_prediction
            },
            'top_5_predictions': top_5_predictions,
            'model_comparison_chart': model_comparison_chart,
            'probability_chart': probability_chart
        }
        
        return results
    
    def feature_importance(self, vector):
        # Get feature importance from Random Forest model
        feature_importance = self.rf_model.feature_importances_
        feature_names = self.tfidf.get_feature_names_out()
        
        # Get non-zero features from the input vector
        non_zero_indices = vector.nonzero()[1]
        
        # Create a list of (feature, importance) tuples for non-zero features
        feature_imp_list = []
        for idx in non_zero_indices:
            if idx < len(feature_names):
                feature_imp_list.append((feature_names[idx], feature_importance[idx]))
        
        # Sort by importance
        feature_imp_list.sort(key=lambda x: x[1], reverse=True)
        
        return feature_imp_list[:10]  # Return top 10 features
    
    def save_charts_to_disk(self, results, output_dir='charts'):
        """Save the generated charts to disk as PNG files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model comparison chart
        if 'model_comparison_chart' in results:
            chart_data = base64.b64decode(results['model_comparison_chart'])
            with open(f"{output_dir}/model_comparison.png", 'wb') as f:
                f.write(chart_data)
            print(f"Model comparison chart saved to {output_dir}/model_comparison.png")
        
        # Save probability chart
        if 'probability_chart' in results:
            chart_data = base64.b64decode(results['probability_chart'])
            with open(f"{output_dir}/probability_chart.png", 'wb') as f:
                f.write(chart_data)
            print(f"Probability chart saved to {output_dir}/probability_chart.png")

# Example usage
if __name__ == "__main__":
    predictor = DiseasePredictor()
    predictor.train_models()
    
    # Test prediction
    test_symptoms = "I have a fever, cough, and difficulty breathing"
    results = predictor.predict(test_symptoms)
    
    print("\nPrediction Results:")
    print(f"Random Forest Prediction: {results['random_forest']['prediction']} with {results['random_forest']['confidence']:.2f}% confidence")
    print(f"SVM Prediction: {results['svm']['prediction']} with {results['svm']['confidence']:.2f}% confidence")
    print(f"Linear Regression Prediction: {results['linear_regression']['prediction']}")
    print(f"Bayesian Regression Prediction: {results['bayesian_regression']['prediction']}")
    
    print("\nTop 5 Predictions:")
    for disease, prob in results['top_5_predictions']:
        print(f"{disease}: {prob:.2f}%")
    
    print("\nCharts generated successfully!")
    print("- Model comparison chart")
    print("- Probability chart")
    
    # Save charts to disk
    predictor.save_charts_to_disk(results)