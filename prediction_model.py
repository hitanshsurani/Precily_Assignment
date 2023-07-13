import tensorflow as tf
import tensorflow_hub as hub
from numpy import dot
from numpy.linalg import norm

def predict(text1, text2):
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)

    def embed(input):
        return model(input)

    msg = [text1, text2]
    msg_embeddings = embed(msg)
    a = tf.make_ndarray(tf.make_tensor_proto(msg_embeddings))
    cos_sim = dot(a[0], a[1]) / (norm(a[0]) * norm(a[1]))
    return cos_sim
