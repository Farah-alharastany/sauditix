from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
ALLOWED_EXTENSIONS = {'pkl'}
MODEL_FILENAME = "new_model.pkl"
ENCODER_FILENAME = "ordinal_encoder.pkl"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Debugging model loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Current directory: {BASE_DIR}")
print(f"Files present: {os.listdir(BASE_DIR)}")

# Load model and encoder with enhanced error handling
try:
    model_path = os.path.join(BASE_DIR, MODEL_FILENAME)
    encoder_path = os.path.join(BASE_DIR, ENCODER_FILENAME)
    print(f"Attempting to load model from: {model_path}")
    print(f"Attempting to load encoder from: {encoder_path}")
    
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    print("✅ Model and encoder loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    model = encoder = None
    if not os.path.exists(model_path):
        print(f"Model file not found at: {model_path}")
    if not os.path.exists(encoder_path):
        print(f"Encoder file not found at: {encoder_path}")

# ===== MAPPING DICTIONARIES =====
TEAM_MAPPING = {
    'Spain': 1, 'Germany': 2, 'Brazil': 3, 'Argentina': 4,
    'France': 5, 'England': 6, 'Portugal': 7, 'Italy': 8,
    'إسبانيا': 1, 'ألمانيا': 2, 'البرازيل': 3, 'الأرجنتين': 4,
    'فرنسا': 5, 'إنجلترا': 6, 'البرتغال': 7, 'إيطاليا': 8
}

STAGE_MAPPING = {
    'Group Stage': 1, 'Group%20Stage': 1, 'دور المجموعات': 1,
    'Quarter-Final': 2, 'Quarter%20Final': 2, 'ربع النهائي': 2,
    'Semi-Final': 3, 'Semi%20Final': 3, 'نصف النهائي': 3,
    'Final': 4, 'النهائي': 4
}

IMPORTANCE_MAPPING = {
    'low': 1, 'منخفض': 1,
    'medium': 2, 'متوسط': 2,
    'high': 3, 'مرتفع': 3
}

VENUE_MAPPING = {
    'King Abdullah Stadium': 1, 'King%20Abdullah%20Stadium': 1, 'ملعب الملك عبدالله': 1,
    'International Stadium': 2, 'International%20Stadium': 2, 'الملعب الدولي': 2,
    'Group Al Borj Stadium': 3, 'Group%20Al%20Borj%20Stadium': 3, 'ملعب المدينة': 3,
    'National Stadium': 4, 'National%20Stadium': 4, 'الملعب الوطني': 4
}

FLAG_MAPPING = {
    'Spain': 'es', 'Germany': 'de', 'Brazil': 'br', 'Argentina': 'ar',
    'France': 'fr', 'England': 'gb', 'Portugal': 'pt', 'Italy': 'it',
    'إسبانيا': 'es', 'ألمانيا': 'de', 'البرازيل': 'br', 'الأرجنتين': 'ar',
    'فرنسا': 'fr', 'إنجلترا': 'gb', 'البرتغال': 'pt', 'إيطاليا': 'it'
}

def safe_convert(value, convert_type):
    """Safely convert values to specified type"""
    try:
        if convert_type == int:
            return int(float(value))  # Handle both "10" and "10.0"
        elif convert_type == float:
            return float(value)
        return value
    except (ValueError, TypeError):
        return convert_type(0) if convert_type in (int, float) else value

def validate_input(data):
    """Validate and convert input data to correct types with mapping support"""
    try:
        # Get raw values
        team1 = str(data.get('team1', ''))
        team2 = str(data.get('team2', ''))
        stage = str(data.get('stage', ''))
        venue = str(data.get('venue', ''))
        importance = str(data.get('importance', ''))
        
        # Convert numeric fields
        base_price = safe_convert(data.get('base_price', 0), float)
        seat_multiplier = safe_convert(data.get('seat_multiplier', 1.0), float)
        tickets_sold = safe_convert(data.get('tickets_sold', 0), int)
        days_until = safe_convert(data.get('days_until', 0), int)
        year = safe_convert(data.get('year', 2025), int)
        
        validated = {
            'base_price': base_price,
            'seat_multiplier': seat_multiplier,
            'tickets_sold': tickets_sold,
            'days_until': days_until,
            'year': year,
            'importance': IMPORTANCE_MAPPING.get(importance.lower(), 1),
            'stage': STAGE_MAPPING.get(stage, 1),
            'venue': VENUE_MAPPING.get(venue, 1),
            'team1': TEAM_MAPPING.get(team1, 0),
            'team2': TEAM_MAPPING.get(team2, 0),
            # Keep original strings for display
            'team1_name': team1,
            'team2_name': team2,
            'stage_name': stage,
            'venue_name': venue,
            'importance_name': importance,
            'team1_flag': FLAG_MAPPING.get(team1, ''),
            'team2_flag': FLAG_MAPPING.get(team2, '')
        }
        return validated, None
    except Exception as e:
        return None, f"Invalid input data: {str(e)}"

@app.route("/")
def index():
    return render_template("index.html", price=None)


@app.route("/match")
def match():
    # Get all parameters from URL
    params = request.args.to_dict()
    
    # Validate and convert parameters
    validated_data, error = validate_input(params)
    if error:
        return render_template("error.html", message=error)
    
    # Calculate predicted price
    if model and encoder:
        try:
            prediction_data = pd.DataFrame([{
                "Base_Price_Base": validated_data['base_price'] / validated_data['seat_multiplier'],
                "Importance_Num": validated_data['importance'],
                "Stage_Num": validated_data['stage'],
                "Venue_Num": validated_data['venue'],
                "Team1_Num": validated_data['team1'],
                "Team2_Num": validated_data['team2'],
                "Days_until_match": validated_data['days_until'],
                "Tickets_Sold": validated_data['tickets_sold'],
                "Year": validated_data['year']
            }])
            predicted_price = model.predict(prediction_data)[0]
            validated_data['predicted_price'] = round(float(predicted_price), 2)
        except Exception as e:
            print(f"Prediction error in /match: {str(e)}")
            validated_data['predicted_price'] = None
    
    return render_template("match.html", match_data=validated_data)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or encoder is None:
        return jsonify({'error': 'Model not loaded properly'}), 500
        
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    try:
        data = request.get_json()
        validated_data, error = validate_input(data)
        if error:
            return jsonify({'error': error}), 400

        # Prepare prediction DataFrame
        prediction_data = pd.DataFrame([{
            "Base_Price_Base": validated_data['base_price'] / validated_data['seat_multiplier'],
            "Importance_Num": validated_data['importance'],
            "Stage_Num": validated_data['stage'],
            "Venue_Num": validated_data['venue'],
            "Team1_Num": validated_data['team1'],
            "Team2_Num": validated_data['team2'],
            "Days_until_match": validated_data['days_until'],
            "Tickets_Sold": validated_data['tickets_sold'],
            "Year": validated_data['year']
        }])

        # Make prediction
        predicted_price = model.predict(prediction_data)[0]
        return jsonify({
            'predicted_price': round(float(predicted_price), 2),
            'status': 'success',
            'match_data': {
                'team1': validated_data['team1_name'],
                'team2': validated_data['team2_name'],
                'stage': validated_data['stage_name'],
                'venue': validated_data['venue_name'],
                'importance': validated_data['importance_name'],
                'team1_flag': validated_data['team1_flag'],
                'team2_flag': validated_data['team2_flag'],
                'base_price': validated_data['base_price'],
                'seat_multiplier': validated_data['seat_multiplier'],
                'tickets_sold': validated_data['tickets_sold'],
                'days_until': validated_data['days_until'],
                'year': validated_data['year']
            }
        })

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e),
            'input_data': data if 'data' in locals() else None
        }), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7860, debug=True)