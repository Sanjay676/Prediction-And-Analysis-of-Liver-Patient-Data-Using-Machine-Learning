
# import numpy as np
# from flask import Flask, request, jsonify, render_template
# import pickle

# app = Flask(__name__)
# model = pickle.load(open('indian_liver_patient.pkl', 'rb'))

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     '''
#     For rendering results on HTML GUI
#     '''
#     try:
#         int_features = [int(x) for x in request.form.values()]
#         final_features = [np.array(int_features)]
#         prediction = model.predict(final_features)
#         output = round(prediction[0], 2)
#         return render_template('index.html', prediction_text='Predicted Value: $ {}'.format(output))
#     except Exception as e:
#         return str(e)

# @app.route('/predict_api', methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls through request
#     '''
#     try:
#         data = request.get_json(force=True)
#         prediction = model.predict([np.array(list(data.values()))])
#         output = prediction[0]
#         return jsonify(output)
#     except Exception as e:
#         return jsonify({'error': str(e)})

# if __name__ == "__main__":
#     app.run(debug=True)



"""from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
try:
    with open('indian_liver_patient.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except (EOFError, FileNotFoundError) as e:
    model = None
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            if model is None:
                raise ValueError("Model not loaded properly")
            
            # Get input data from form
            data = request.form.to_dict()
            data = {k: float(v) for k, v in data.items()}
            
            # Convert to DataFrame
            df = pd.DataFrame([data])
            
            # Make prediction
            prediction = model.predict(df)
            
            # Return result
            result = 'Liver disease' if prediction[0] == 1 else 'No liver disease'
            return render_template('result.html', result=result)
        
        except Exception as e:
            return str(e)
    return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=True)"""
    
    
from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model once during the app initialization
model = pickle.load(open('liver_analysis.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def index():
    return render_template('predict.html')

@app.route('/data_predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = request.form['age']
        gender = request.form['gender']
        tb = request.form['tb']
        db = request.form['db']
        ap = request.form['ap']
        aa1 = request.form['aa1']
        aa2 = request.form['aa2']
        tp = request.form['tp']
        a = request.form['a']
        agr = request.form['agr']
        
        data = [[float(age), float(gender), float(tb), float(db), float(ap), float(aa1), float(aa2), float(tp), float(a), float(agr)]]
        
        prediction = model.predict(data)[0]
        if prediction == 1:
            result = "You have a liver disease"
        else:
            result = "You don't have a liver disease"
        
        return render_template('result.html', prediction=result)
    
if __name__ == '__main__':
    app.run(debug=True)
