from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Update your model path and load the Spotify popularity model
# Ensure you have trained and saved your model as 'spotify_model.pkl'
model_path = 'music-prediction.pkl'  # Replace with your actual model path
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Define the route for popularity prediction
@app.route('/predict_popularity', methods=['POST'])
def predict_popularity():
    try:
        # Extract data from form
        energy = request.form.get('Energy', type=float)
        valence = request.form.get('Valence', type=float)
        danceability = request.form.get('Danceability', type=float)
        loudness = request.form.get('Loudness', type=float)
        acousticness = request.form.get('Acousticness', type=float)
        tempo = request.form.get('Tempo', type=float)
        speechiness = request.form.get('Speechiness', type=float)
        liveness = request.form.get('Liveness', type=float)

        # Validate inputs
        if None in [energy, valence, danceability, loudness, acousticness, tempo, speechiness, liveness]:
            return render_template('spotify.html', prediction_text='Invalid input. Please provide all fields.')

        # Create numpy array for prediction
        final_features = np.array([[energy, valence, danceability, loudness, acousticness, tempo, speechiness, liveness]])

        # Make prediction
        predicted_popularity = model.predict(final_features)[0]

        return render_template('spotify.html', prediction_text='Predicted Popularity: {:.2f}'.format(predicted_popularity))

    except Exception as e:
        return render_template('spotify.html', prediction_text=f'Error: {str(e)}')

# Define the main page route
@app.route('/')
def main_page():
    return render_template('spotify.html')

if __name__ == "__main__":
    app.run(debug=True)
