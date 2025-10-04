# Path2Care - AI-Powered Healthcare Companion

Path2Care is a comprehensive healthcare application that combines symptom-based disease prediction with AI-powered skin disease detection using advanced machine learning models.

## Features

### 🏥 Disease Prediction
- **Symptom Analysis**: Input your symptoms in natural language
- **AI Diagnosis**: Get predictions based on machine learning models trained on extensive medical data
- **Top 5 Predictions**: View multiple possible diagnoses with confidence scores
- **Visual Analytics**: Interactive charts showing prediction distributions and feature importance

### 🖼️ Skin Disease Detection
- **Image Upload**: Upload skin condition images for AI analysis
- **CNN Model**: Advanced Convolutional Neural Network trained on 10,000+ medical images
- **Top 5 Predictions**: Get multiple possible skin conditions with confidence scores
- **10 Disease Categories**: Covers major skin conditions including:
  - Eczema and Atopic Dermatitis
  - Melanoma and Basal Cell Carcinoma
  - Melanocytic Nevi (Moles)
  - Benign Keratosis and Seborrheic Keratoses
  - Psoriasis and Lichen Planus
  - Fungal Infections (Ringworm, Candidiasis)
  - Viral Infections (Warts, Molluscum)

### 🏥 Hospital Locator
- **Geolocation**: Find hospitals near your current location
- **Specialization Filter**: Search by disease specialization
- **Interactive Map**: TomTom-powered mapping with hospital locations
- **Distance Calculation**: See hospitals within 10km radius

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Path2Care
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files are present**:
   - `skin_disease_model.h5` - Pre-trained CNN model for skin disease detection
   - `Symptom2Disease.csv` - Training data for symptom-based prediction

## Usage

### Starting the Application
```bash
python app.py
```

The application will be available at `http://localhost:5000`

### Testing the Skin Disease Model
```bash
python test_skin_model.py
```

This will verify that the model can be loaded and used properly.

## Model Architecture

### Skin Disease Detection Model
- **Input**: 150x150 RGB images
- **Architecture**: CNN with Conv2D, MaxPooling2D, Dense layers
- **Training**: 10 epochs on 3,248 training images
- **Classes**: 10 different skin disease categories
- **Accuracy**: ~61% on validation set

### Symptom Prediction Model
- **Input**: Text descriptions of symptoms
- **Architecture**: TF-IDF + Random Forest Classifier
- **Features**: 1000 most important medical terms
- **Output**: Disease predictions with confidence scores

## API Endpoints

- `/` - Home page
- `/predict` - Symptom-based disease prediction
- `/skin_diseases` - Skin disease detection via image upload
- `/hospitals` - Hospital locator with interactive map
- `/geocode` - Location geocoding service
- `/search_hospitals` - Hospital search API

## File Structure

```
Path2Care/
├── app.py                          # Main Flask application
├── skin_disease_model.h5          # Pre-trained CNN model
├── requirements.txt               # Python dependencies
├── test_skin_model.py            # Model testing script
├── templates/                     # HTML templates
│   ├── index.html                # Home page
│   ├── predict.html              # Disease prediction page
│   ├── skin_diseases.html        # Skin disease detection page
│   └── hospital_map.html         # Hospital locator page
├── static/                        # Static assets
│   └── Path2Care.jpeg            # Application logo
└── Skin_Diseases/                # Training dataset
    ├── 1. Eczema 1677/           # Eczema images
    ├── 2. Melanoma 15.75k/       # Melanoma images
    ├── 3. Atopic Dermatitis - 1.25k/ # Atopic dermatitis images
    └── ...                       # Other disease categories
```

## Dependencies

- **Flask**: Web framework
- **TensorFlow**: Deep learning framework
- **Pillow**: Image processing
- **OpenCV**: Computer vision
- **scikit-learn**: Machine learning utilities
- **pandas & numpy**: Data manipulation
- **matplotlib**: Data visualization
- **NLTK**: Natural language processing

## Important Notes

### Medical Disclaimer
⚠️ **This application is for educational and screening purposes only. It should not replace professional medical diagnosis. Always consult with a healthcare provider for proper medical evaluation and treatment.**

### Model Limitations
- The skin disease model has ~61% accuracy on validation data
- Predictions are based on visual patterns and may not capture all clinical nuances
- Image quality and lighting can affect prediction accuracy
- The model is trained on specific datasets and may not generalize to all skin conditions

### Performance Considerations
- Model loading may take a few seconds on first run
- Image processing requires sufficient memory
- Large image files may slow down prediction

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational purposes. Please ensure compliance with local regulations regarding medical software and AI applications.

## Support

For technical issues or questions about the application, please open an issue in the repository.

---

**Built with ❤️ for healthcare innovation**
