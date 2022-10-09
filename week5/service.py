import pickle
from flask import Flask
from flask import request
from flask import jsonify

with open("dv.bin", "rb") as f_in:
    dv = pickle.load(f_in)
with open("model1.bin", "rb") as f_in:
    model = pickle.load(f_in)

app = Flask('predict')


@app.route('/predict', methods=['POST'])
def predict_client():
    client = request.get_json()
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0,1]
    result = {
        'probability': y_pred
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)