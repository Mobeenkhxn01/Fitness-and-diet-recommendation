import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import random

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('gym-recommendation.csv')

# Ensure 'Sex' column is treated as a string
df['Sex'] = df['Sex'].astype(str)

# Create LabelEncoder and fit on the entire 'Sex' column
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# Preprocess the dataset
df['BMI'] = df['Weight'] / (df['Height'] ** 2)
X = df[['Sex', 'Age', 'Weight', 'Height', 'BMI']].copy()

# Target variables
y = df[['Level', 'Fitness Goal', 'Fitness Type', 'Exercises', 'Diet']]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Random Forest model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train_scaled, y_train)

# Function to clean up exercises (remove 'and')
def clean_exercises(exercise_string):
    exercises = []
    for part in exercise_string.split(' or '):
        part_exercises = part.replace(' and ', ', ').split(',')
        exercises.extend([ex.strip().capitalize() for ex in part_exercises if ex.strip()])
    
    return exercises

# Function to parse diet into structured format with subpoints
def parse_diet(diet_string):
    items = diet_string.split(';')
    structured_diet = []
    for item in items:
        if '(' in item:
            main_item, subitems = item.split('(')
            subitems = subitems.rstrip(')').replace(' and ', ', ').replace('or',', ').split(',')
            structured_diet.append({
                'main_item': main_item.strip().capitalize(),
                'subitems': [sub.strip().capitalize() for sub in subitems if sub.strip()]
            })
        else:
            structured_diet.append({'main_item': item.strip().capitalize(), 'subitems': []})
    return structured_diet

# Function to create a weekly planner with random subitems from diet
def create_weekly_plan(exercises, diet):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    planner = {}
    for day in days:
        day_exercises = np.random.choice(exercises, size=2, replace=False)  # Random 2 exercises
        day_diet = [random.choice(d['subitems']) for d in diet]  # Random subitems from each diet category
        planner[day] = {
            'exercise': day_exercises,
            'diet': day_diet
        }
    return planner

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    sex = request.form['sex']  # 'male' or 'female'
    age = int(request.form['age'])
    weight = float(request.form['weight'])
    height = float(request.form['height'])
    bmi = weight / (height ** 2)

    # Encode the 'sex' input using the same label encoder
    sex_encoded = le.transform([sex])[0]  # Convert 'male' or 'female' to the encoded value

    # Create the input data for prediction
    user_data = np.array([[sex_encoded, age, weight, height, bmi]])

    print(sex)
    print(sex_encoded)

    # Scale the input data
    user_data_scaled = scaler.transform(user_data)

    # Predict using the trained Random Forest model
    prediction = rf.predict(user_data_scaled)
    
    level = prediction[0][0]
    goal = prediction[0][1]
    fitness_type = prediction[0][2]
    
    # Clean and split exercises and diet
    exercises = clean_exercises(prediction[0][3])
    diet = parse_diet(prediction[0][4])

    # Define a threshold for how close the BMI needs to be
    bmi_threshold = 0.5  # Example threshold

# Fetch recommendations where BMI is within the threshold range
    filtered_recommendation = df.loc[
        (df['Sex'] == sex) & 
        (df['Age'] == age) & 
        (df['BMI'] >= (bmi - bmi_threshold)) | 
        (df['BMI'] <= (bmi + bmi_threshold)), 
        'Recommendation'
    ]
# Check if we found any recommendations
    if not filtered_recommendation.empty:
        recommendation = filtered_recommendation.values[0]  # Get the first matching recommendation
    else:
        recommendation = "No specific recommendation available."


    # Create weekly planner
    weekly_planner = create_weekly_plan(exercises, diet)

    return render_template('result.html', bmi=round(bmi, 2), level=level, goal=goal,
        fitness_type=fitness_type, exercises=exercises, diet=diet, planner=weekly_planner, Recommendation=recommendation)

if __name__ == '__main__':
    app.run(debug=True)
