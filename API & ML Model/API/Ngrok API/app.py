from flask import Flask, request, jsonify,render_template,send_file
import numpy as np
import joblib
import pandas as pd
from scipy.stats import mode
import warnings
import traceback  # Import the traceback module
import smtplib
from email.mime.text import MIMEText
from flask_ngrok import run_with_ngrok

app = Flask(__name__)

run_with_ngrok(app)

# Loading machine learning models
model1 = joblib.load('SVM_Model.pkl')
model2 = joblib.load('NaiveB_Model.pkl')
model3 = joblib.load('RandomForest_Model.pkl')

# Loading label encoder to convert integer result to String value
le = joblib.load('labelEncoder.pkl') 



@app.route('/state')
def start():
    return "Started"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.json
        

        # Extract the "features" key containing the list of binary values
        features = data.get("features")
        input_data = np.array(features)

        if features is not None and len(features) == 131:
            # Convert the list to a NumPy array
            input_data = np.array(features)

            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

            # Make a prediction using your model
            prediction1 = model1.predict([input_data])
            prediction2 = model2.predict([input_data])
            prediction3 = model3.predict([input_data])

            prediction1 = prediction1.flatten()
            prediction2 = prediction2.flatten()
            prediction3 = prediction3.flatten()

            finalPrediction = mode([prediction1, prediction2, prediction3])[0][0]
            finalPrediction = np.asarray(finalPrediction).ravel()

            # Using the label encoder to transform the prediction back to its original label
            predicted_label = le.inverse_transform(finalPrediction)[0]
        
        
            # predicted_label_list = predicted_label.tolist()

            # You can return the result as a string
            result = predicted_label

            return jsonify({"prediction": predicted_label})
        else:
            
            print(features)
            print(input_data.shape)
            
            return jsonify({"error": "Invalid input data"}), 400

    except Exception as e:
        traceback.print_exc()
       
        return jsonify({"error": str(e)}), 500

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv('DiseaseDescription.csv')

@app.route('/get_description/<Disease>', methods=['GET'])
def get_description(Disease):
    
    # Find the precaution for the given disease in the DataFrame
    result = df[df['Disease'] == Disease]

    # Check if the disease is not found
    if result.empty:
        return jsonify({'error': 'Disease not found'}), 404

    # Get the precaution from the result
    description = result['Description'].values[0]

    # Return the precaution as JSON
    return jsonify({'disease': Disease, 'description': description})


df2 = pd.read_csv('DiseasePrecautions.csv')

# Assume 'symptom' is the column containing symptoms
disease_column = 'Disease'
# Specify the columns you want to concatenate
precaution_columns = ['Precaution_1', 'Precaution_2', 'Precaution_3','Precaution_4']

def get_precautions(disease):
    # Filter the DataFrame based on the specified symptom
    filtered_data = df2[df2[disease_column] == disease]

    if filtered_data.empty:
        return "disease not found."

    # Concatenate the values in the specified columns for each row
    concatenated_precautions = filtered_data[precaution_columns].apply(lambda row: '\n'.join(row.astype(str)), axis=1)

    return concatenated_precautions.iloc[0]  # Assuming you want the result for the first matching row

@app.route('/get_precautions/<Disease>', methods=['GET'])
def get_precautions_api(Disease):
   
    result = get_precautions(Disease)

    return jsonify({"precautions": result})



IMAGES_FOLDER="images"

@app.route('/getImage/<image_name>',methods=['GET'])
def get_image(image_name):
    image_path=f"{IMAGES_FOLDER}/{image_name}.png"

    try:
        return send_file(image_path,mimetype='image/png')
    except FileNotFoundError:
        default_image_path="images/default.png"
        return send_file(default_image_path,mimetype='image/png')



def send_email(sender_email, sender_password, recipient_email, subject, body):
    # Setup the MIME
    message = MIMEText(body)
    message['Subject'] = subject
    message['From'] = sender_email
    message['To'] = recipient_email

    # Connect to the SMTP server
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        # Start TLS for security
        server.starttls()

        # Login to the email account
        server.login(sender_email, sender_password)

        # Send the email
        server.sendmail(sender_email, recipient_email, message.as_string())

    return 'Email sent successfully'

@app.route('/send-email', methods=['POST'])
def receive_email():
    data = request.get_json()

    sender_email = 'ayanahmad2910@gmail.com'
    sender_password = 'nkzr krvu ptyx gcui'

    recipient_email = data.get('recipient_email')
    subject = data.get('subject')
    body = data.get('body')

    result = send_email(sender_email, sender_password, recipient_email, subject, body)
    
    return jsonify({'result': result})




if __name__ == '__main__':
    app.run()



