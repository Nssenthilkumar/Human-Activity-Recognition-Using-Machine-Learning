from flask import Flask, render_template, request
import joblib
#from sklearn.externals import joblib
import pickle


app = Flask(__name__)

knn_model = joblib.load('svm_model.pkl')

@app.route("/")

@app.route("/accuracy", methods = ["POST"])
def display_accuracy():
    with open('svm_accuracy.txt', 'r') as f:
        accuracy = f.read()
    return render_template('index.html', accuracy=accuracy)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000)
