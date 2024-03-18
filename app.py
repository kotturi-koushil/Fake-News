from flask import Flask, request, render_template
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.pipeline import Pipeline


model = joblib.load("twit.pkl")


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        text = request.form["text1"]
        ans = model.predict([text])
        if ans[0] == 0:
            return render_template("result.html", content="Fake")
        else:
            return render_template("result.html", content="True")
    return render_template("index.html")


if __name__ == "__main__":
    app.run()
