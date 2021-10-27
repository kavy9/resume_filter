from flask import Flask ,request ,jsonify
from flask import jsonify


# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# import tensorflow as tf

# from tensorflow import keras
# import re
# from nltk.corpus import stopwords

import nltk

# from tensorflow.keras.preprocessing.text import one_hot
# from nltk.stem.porter import PorterStemmer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from nltk.corpus import stopwords
# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
# from nltk.corpus import stopwords
# import numpy as np

app=Flask(__name__)

nltk.download('stopwords')
ps=PorterStemmer()
model = keras.models.load_model('model_text_analysis')

from sklearn.externals import joblib
label_encoder = joblib.load('text_analysis_label_encoder.pkl')

voc_size=10000
sent_length=35

def predict_emotion(stri):
    review = re.sub('[^a-zA-Z]', ' ', stri)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    onehot_repr = [one_hot(review,voc_size)] 
    embed = pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
    predicti = model.predict(embed)
    return label_encoder.classes_[np.argmax(predicti)]


@app.route("/",methods=['POST'])
def helloWorld():
    data=request.json
    # print(data[0])
    # dict_data=dict(data[0])
    # print(dict_data)
    predictioni=predict_emotion(data[0]['data'])
    return jsonify({"data":predictioni})

#https://stackoverflow.com/questions/59823283/could-not-load-dynamic-library-cudart64-101-dll-on-tensorflow-cpu-only-install