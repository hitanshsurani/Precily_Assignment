from flask import Flask, render_template, request , redirect , url_for
import tensorflow as tf
import tensorflow_hub as hub
from numpy import dot
from numpy.linalg import norm

app = Flask(__name__,template_folder='template')

# Load the ML model
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)


@app.route('/')
def index():
    similarity_score = request.args.get('similarity_score')
    return render_template('index.html', similarity_score=similarity_score)



@app.route('/predict', methods=['POST'])
def predict():
    text1 = request.form['text1']
    text2 = request.form['text2']
    print("Text 1:", text1)
    print("Text 2:", text2)
    msg = [text1, text2]
    msg_embeddings = model(msg)
    a = tf.make_ndarray(tf.make_tensor_proto(msg_embeddings))
    cos_sim = dot(a[0], a[1]) / (norm(a[0]) * norm(a[1]))

    # Make predictions using the ML model
    similarity_score = cos_sim

    return redirect(url_for('index', similarity_score=similarity_score))


if __name__ == '__main__':
    app.run(debug=True)