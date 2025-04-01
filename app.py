from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# تحميل النموذج
model = joblib.load('random_forest.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        print("📥 Received data:", data)

        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)

        return jsonify({
            'diagnosis': str(prediction[0]),
            'is_emergency': prediction[0] in ['Heart Attack', 'Stroke']
        })

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
