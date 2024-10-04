from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField, StringField
from wtforms.validators import DataRequired
import numpy as np

from ml.predict import predict_cancer

app = Flask(__name__)

app.config['SECRET_KEY'] = 'your_secret_key' 



class PredictionForm(FlaskForm):
    worst_area = FloatField('Worst Area', default=2019.0, validators=[DataRequired()])
    worst_concave_points = FloatField('Worst Concave Points', default=0.2654, validators=[DataRequired()])
    mean_concave_points = FloatField('Mean Concave Points', default=0.14710, validators=[DataRequired()])
    worst_radius = FloatField('Worst Radius', default=25.380, validators=[DataRequired()])
    mean_concavity = FloatField('Mean Concavity', default=0.30010, validators=[DataRequired()])
    submit = SubmitField('Predict')


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        pass
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        pass

    return render_template('login.html')

@app.route('/logout')
def logout():
    #remove from session
    return redirect(url_for('login'))

@app.route('/user_input', methods=['GET', 'POST'])
@app.route('/user_input', methods=['GET', 'POST'])
def user_input():
    form = PredictionForm()
    
    if form.validate_on_submit():
        # Gather features from form input
        features = np.array([[
            form.worst_area.data,
            form.worst_concave_points.data,
            form.mean_concave_points.data,
            form.worst_radius.data,
            form.mean_concavity.data
        ]])
        
        # Make prediction
        prediction = predict_cancer(features)[0]
        
        if prediction == 1:
            cancer_type = 'Malignant'
            explanation = "Malignant tumors are cancerous. They can grow quickly and spread to other parts of the body. Early detection and treatment are crucial."
        else:
            cancer_type = 'Benign'
            explanation = "Benign tumors are not cancerous and usually grow slowly. They don't spread to other parts of the body and are less harmful, but monitoring is still important."
        
        # Redirect to result page, passing the prediction type and explanation
        return redirect(url_for('result', prediction=cancer_type, explanation=explanation))
    
    # Render the input form
    return render_template('user_input.html', form=form, prediction="NO PREDICTION")


@app.route('/result')
def result():
    # Get the prediction and explanation from the query parameters
    prediction = request.args.get('prediction', 'No prediction available')
    explanation = request.args.get('explanation', '')
    
    # Render the result page with the prediction and explanation
    return render_template('result.html', prediction=prediction, explanation=explanation)

@app.route('/history')
def history():
    return render_template('history.html')


if __name__ == '__main__':
    app.run(debug=True)
