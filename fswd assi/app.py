from flask import Flask, render_template
import pandas as pd
import pickle

app = Flask(__name__, template_folder="templates")

with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    df=pd.read_csv('AQI_Data.csv')
    my_prediction=loaded_model.predict(df.iloc[:,:-1].values)
    my_prediction=my_prediction.tolist()
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)
