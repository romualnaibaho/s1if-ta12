from flask import Flask, render_template, request, redirect
import pickle
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)

# initialize the database
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgres://lpumvkzddhzjql:b91763d9b7b11c8a1aa95676f9d0428c78e617b58cdf4dfed5718fae2616ec0b@ec2-52-206-182-219.compute-1.amazonaws.com:5432/d6fr5uojn85vd4'

db = SQLAlchemy(app)

with open('svm_bp_model', 'rb') as model:
  svm_model = pickle.load(model)

with open('knn_bp_model', 'rb') as model:
  knn_model = pickle.load(model)

class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    last_achievement = db.Column(db.String(50), nullable=False) 
    sick_leaves = db.Column(db.String(50), nullable=False) 
    age = db.Column(db.String(50), nullable=False) 
    job_rotation = db.Column(db.String(50), nullable=False) 
    year_graduated = db.Column(db.String(50), nullable=False) 
    job_duration_from_training = db.Column(db.String(50), nullable=False) 
    branch_rotation = db.Column(db.String(50), nullable=False) 
    assign_of_other_position = db.Column(db.String(50), nullable=False) 
    annual_leave = db.Column(db.String(50), nullable=False) 
    gpa = db.Column(db.String(50), nullable=False) 
    job_duration_in_current_branch = db.Column(db.String(50), nullable=False) 
    gender = db.Column(db.String(50), nullable=False) 
    number_of_dependences = db.Column(db.String(50), nullable=False) 
    job_duration_in_current_person_level = db.Column(db.String(50), nullable=False) 
    education_level = db.Column(db.String(50), nullable=False) 
    employee_type = db.Column(db.String(50), nullable=False) 
    job_duration_in_current_job_level = db.Column(db.String(50), nullable=False) 
    achievement_above_100_during3quartal = db.Column(db.String(50), nullable=False)
    best_performance = db.Column(db.String(50), nullable=False)
    date_added = db.Column(db.DateTime, default=datetime.utcnow)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/history')
def history():
    datas = Result.query.order_by(Result.date_added)

    return render_template('history.html', datas=datas)

@app.route('/ta-12/admin/delete/<int:id>')
def delete(id):
    data_to_delete = Result.query.get_or_404(id)
    db.session.delete(data_to_delete)
    db.session.commit()

    datas = Result.query.order_by(Result.date_added)

    return redirect(url_for('history'))


@app.route('/predict', methods=['POST'])
def predict():
    if(request.method == "POST"):
        data = {
            'Last_achievement_%' : [request.form["last_achievement_percent"]],
            'sick_leaves' : [request.form["sick_leaves"]],
            'age' : [request.form["age"]],
            'job_rotation' : [request.form["job_rotation"]],
            'year_graduated' : [request.form["year_graduated"]],
            'job_duration_from_training' : [request.form["job_duration_from_training"]],
            'branch_rotation' : [request.form["branch_rotation"]],
            'assign_of_otherposition' : [request.form["assign_of_other_position"]],
            'annual leave' : [request.form["annual_leave"]],
            'GPA' : [request.form["gpa"]],
            'job_duration_in_current_branch' : [request.form["job_duration_in_current_branch"]],
            'gender' : [request.form["gender"]],
            'number_of_dependences' : [request.form["number_of_dependences"]],
            'job_duration_in_current_person_level' : [request.form["job_duration_in_current_person_level"]],
            'Education_level' : [request.form["education_level"]],
            'Employee_type' : [request.form["employee_type"]],
            'job_duration_in_current_job_level' : [request.form["job_duration_in_current_job_level"]],
            'Achievement_above_100%_during3quartal' : [request.form["achievement_above_100_percent"]]
        }

        new_data = pd.DataFrame(data)
        predicted_result = []
        model = request.form["model_classifier"]

        if (model == "1"):
            predicted_result = knn_model.predict(new_data)
        else:
            predicted_result = svm_model.predict(new_data)
        
        db_result = Result(
            last_achievement=data['Last_achievement_%'][0],
            sick_leaves = data['sick_leaves'][0],
            age = data['age'][0],
            job_rotation = data['job_rotation'][0],
            year_graduated = data['year_graduated'][0],
            job_duration_from_training = data['job_duration_from_training'][0],
            branch_rotation = data['branch_rotation'][0],
            assign_of_other_position = data['assign_of_otherposition'][0],
            annual_leave = data['annual leave'][0],
            gpa = data['GPA'][0],
            job_duration_in_current_branch = data['job_duration_in_current_branch'][0],
            gender = data['gender'][0],
            number_of_dependences = data['number_of_dependences'][0],
            job_duration_in_current_person_level = data['job_duration_in_current_person_level'][0],
            education_level = data['Education_level'][0],
            employee_type = data['Employee_type'][0],
            job_duration_in_current_job_level = data['job_duration_in_current_job_level'][0],
            achievement_above_100_during3quartal = data['Achievement_above_100%_during3quartal'][0],
            best_performance = str(predicted_result[0])
            )
        db.session.add(db_result)
        db.session.commit()

    return render_template('result.html', data = data, result = predicted_result[0], model = model)

if __name__ == '__main__':
    app.port = 5000
    app.run(debug=True)
