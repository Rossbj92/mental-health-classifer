
from flask import Flask, request, render_template
import reddit_predictor_api as pred

app = Flask(__name__)  # create instance of Flask class

@app.route('/', methods=["POST", "GET"])
def home():
    predictions = pred.predict(request.form.get('feels'))
    text = request.form.get('feels')
    return render_template('elements.html',
                           text=text,
                           prediction=predictions
                          )

if __name__ == '__main__':
    app.run(debug = False)



