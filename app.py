from flask import Flask, request, jsonify
import numpy as np
import joblib
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from flask import Flask, request, jsonify
import io
import base64
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
gbc_model_filename = 'gradient_boosting_model.pkl'
loaded_gbc_model = joblib.load(gbc_model_filename)

# Disease mapping
disease_map = {
    0: 'AIDS', 1: 'Acne', 2: 'Alcoholic Hepatitis', 3: 'Allergy', 4: 'Arthritis',
    5: 'Bronchial Asthma', 6: 'Cervical Spondylosis', 7: 'Chickenpox', 8: 'Chronic Cholestasis',
    9: 'Common Cold', 10: 'Dengue', 11: 'Diabetes', 12: 'Dimorphic Hemmorhoids (piles)',
    13: 'Drug Reaction', 14: 'Fungal Infection', 15: 'GERD', 16: 'Gastroenteritis',
    17: 'Heart Attack', 18: 'Hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C',
    21: 'Hepatitis D', 22: 'Hepatitis E', 23: 'Hypertension', 24: 'Hyperthyroidism',
    25: 'Hypoglycemia', 26: 'Hypothyroidism', 27: 'Impetigo', 28: 'Jaundice',
    29: 'Malaria', 30: 'Migraine', 31: 'Osteoarthritis', 32: 'Paralysis (brain hemorrhage)',
    33: 'Peptic Ulcer Disease', 34: 'Pneumonia', 35: 'Psoriasis', 36: 'Tuberculosis',
    37: 'Typhoid', 38: 'Urinary Tract Infection', 39: 'Varicose Veins', 40: 'Vertigo'
}

import base64
from lime import lime_image
from skimage.io import imsave
import io
from PIL import Image

features = ['Chills', 'Acidity', 'Knee Pain', 'Vomiting', 'Diarrhoea', 'Belly Pain', 
            'Fatigue', 'Sweating', 'Indigestion', 'Headache', 'Blood in Sputum', 'Fast Heart Rate', 
            'Cramps', 'Bruising', 'Nausea', 'Spinning Movements', 'Swollen Legs', 'Skin Peeling', 
            'Unsteadiness', 'Internal Itching']

from flask import Flask, request, jsonify
import numpy as np
import joblib
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from flask import Flask, request, jsonify
import io
import base64
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
gbc_model_filename = 'gradient_boosting_model.pkl'
loaded_gbc_model = joblib.load(gbc_model_filename)

# Disease mapping
disease_map = {
    0: 'AIDS', 1: 'Acne', 2: 'Alcoholic Hepatitis', 3: 'Allergy', 4: 'Arthritis',
    5: 'Bronchial Asthma', 6: 'Cervical Spondylosis', 7: 'Chickenpox', 8: 'Chronic Cholestasis',
    9: 'Common Cold', 10: 'Dengue', 11: 'Diabetes', 12: 'Dimorphic Hemmorhoids (piles)',
    13: 'Drug Reaction', 14: 'Fungal Infection', 15: 'GERD', 16: 'Gastroenteritis',
    17: 'Heart Attack', 18: 'Hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C',
    21: 'Hepatitis D', 22: 'Hepatitis E', 23: 'Hypertension', 24: 'Hyperthyroidism',
    25: 'Hypoglycemia', 26: 'Hypothyroidism', 27: 'Impetigo', 28: 'Jaundice',
    29: 'Malaria', 30: 'Migraine', 31: 'Osteoarthritis', 32: 'Paralysis (brain hemorrhage)',
    33: 'Peptic Ulcer Disease', 34: 'Pneumonia', 35: 'Psoriasis', 36: 'Tuberculosis',
    37: 'Typhoid', 38: 'Urinary Tract Infection', 39: 'Varicose Veins', 40: 'Vertigo'
}

import base64
from lime import lime_image
from skimage.io import imsave
import io
from PIL import Image

features = ['Chills', 'Acidity', 'Knee Pain', 'Vomiting', 'Diarrhoea', 'Belly Pain', 
            'Fatigue', 'Sweating', 'Indigestion', 'Headache', 'Blood in Sputum', 'Fast Heart Rate', 
            'Cramps', 'Bruising', 'Nausea', 'Spinning Movements', 'Swollen Legs', 'Skin Peeling', 
            'Unsteadiness', 'Internal Itching']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log incoming data
        data = request.json
        print("Received data:", data)

        input_features = data.get("features")

        # Ensure features are provided
        if input_features is None:
            return jsonify({"error": "Missing 'features' in request"}), 400

        print("Features:", input_features)

        # Convert features to NumPy array
        input_array = np.array(input_features).reshape(1, -1)
        print("Input array shape:", input_array.shape)

        # Make a prediction
        prediction = loaded_gbc_model.predict(input_array)[0]
        probabilities = loaded_gbc_model.predict_proba(input_array)[0]
        predicted_disease = disease_map.get(int(prediction), "Unknown Disease")

        print("Predicted Disease:", predicted_disease)
        print("Prediction probabilities:", probabilities)

        # Prepare the LIME Explainer
        explainer = LimeTabularExplainer(
            training_data=np.random.rand(100, len(input_features)),  # Fake training data (replace with real data)
            mode="classification",
            feature_names=features,  # Ensure feature names match
            class_names=list(disease_map.values()), 
            discretize_continuous=True
        )

        # Generate LIME explanation for the prediction
        explanation = explainer.explain_instance(input_array[0], loaded_gbc_model.predict_proba)

        # Create a Matplotlib figure
        fig = explanation.as_pyplot_figure()

        # Save figure as a PNG image in memory
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format="png", bbox_inches="tight")
        img_buffer.seek(0)

        # Convert image to Base64
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')

        # Close resources
        img_buffer.close()
        plt.close(fig)  # Close the Matplotlib figure to prevent memory leaks

        print("LIME explanation image generated successfully.")

        return jsonify({
            "prediction": predicted_disease,
            "lime_explanation_image": img_base64  # Send Base64 image to frontend
        })
    
    except Exception as e:
        print("Error occurred:", str(e))
        return jsonify({"error": str(e)}), 500




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)






if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

