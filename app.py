from flask import Flask, render_template, request
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load the Boston dataset
boston = fetch_openml(data_id=531)
X = boston.data
y = boston.target

# Train the model
model = LinearRegression()
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    price = model.predict([features])[0]
    return render_template('index.html', price=price)

if __name__ == '__main__':
    app.run(debug=True)