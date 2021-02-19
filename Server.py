import numpy as np
from flask import Flask, request, jsonify ,render_template,g
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tools.eval_measures import rmse
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import warnings
warnings.filterwarnings("ignore")
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from tensorflow import keras
from pandas.tseries.offsets import DateOffset

import csv


import PyPDF2
import os
import nltk
import re
import math
import string
import gensim
import spacy
import sklearn

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import wordnet 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from DocSim import DocSim

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


app = Flask(__name__,static_url_path='', 
            static_folder='static',
            template_folder='templates')
model = keras.models.load_model("model.h5")
transformer = joblib.load("data_transformer.joblib")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/NewData')
def NewData():
    return render_template('NewData.html')

@app.route('/Analysis')
def Analysis():
    return render_template('Analysis.html')

@app.route('/TextAn')
def TextAn():
    return render_template('TextAn.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("x")
    division = request.form["area"]
    df= LoadPastDataset(division)
    dft= transformer.transform(df)
    batch = dft[-4:].reshape((1,4,1))
    pred_list = []
    for i in range(4):  
        pred_list.append(model.predict(batch)[0])
        batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)
    add_dates = [df.index[-1] + DateOffset(weeks=x) for x in range(0,5) ]
    future_dates = pd.DataFrame(index=add_dates[1:],columns=df.columns)
    df_predict = pd.DataFrame(transformer.inverse_transform(pred_list),
                          index=future_dates[-4:].index, columns=['Prediction'])
    df_proj = pd.concat([df,df_predict], axis=1)
    df_proj.index.name ='Month'
    df_proj.reset_index(inplace=True)
    df_proj['Month'].to_list()
    df_proj['Month']=df_proj['Month'].astype(str)
    dateList = df_proj['Month'].to_list()
    Prediction = df_predict["Prediction"].tolist()
    P1 = Prediction[0]
    P2 = Prediction[1]
    P3 = Prediction[2]
    P4 = Prediction[3]
    Data = df["Total"].tolist()
    print(Prediction)
    print(P1)
    print(dateList[-13:])
    return render_template('index.html', P1=round(P1),P2=round(P2),P3=round(P3),P4=round(P4),Data=Data[-9:],dateList=dateList[-13:],d=division)

from firebase import firebase

@app.route('/EnterNewData', methods=['POST'])
def EnterNewData():
    date = request.form["D"]
    division = request.form["Division"]
    police =  request.form["Police"]
    category = request.form["Category"]
    status = request.form["Status"]

    FBConn = firebase.FirebaseApplication('https://ccminner-252fe.firebaseio.com/', None)
    data_to_upload = {
        'date' : date,
        'division' : division,
        'police' : police,
        'category' : category,
        'status' : status
    }
    result = FBConn.post('MyTestData',data_to_upload)
    return render_template('NewData.html', Done=1)

def LoadPastDataset(area):
    df = pd.read_csv(r'Datasets/ForecastingData/{0}.csv'.format(area))
    df =  df.drop(['Pending','Obstruction_to_police_officers'], axis=1)
    df.Month = pd.to_datetime(df.Month)
    df = df.set_index('Month')
    return df


##preprocessing
def strip_data(s):
    stripped = s.lower()
    replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    stripped=stripped.translate(replace_punctuation)
    stripped = re.sub(r'\s\d\s',' ',stripped)
    stripped = re.sub(r'\s\w\s',' ',stripped)
    stripped = re.sub(r'\s\w\s',' ',stripped)
    stripped = re.sub('\\n+',' ',stripped)
    stripped = re.sub('\s+',' ',stripped)
    stripped = stripped.strip()
    return stripped

def stop_words(sent):
    stop_words = set(stopwords.words('english')) 
    stop_words.update(['page','vs'])
    word_tokens = word_tokenize(sent) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    return filtered_sentence

##lemmantization-spacy
def spacy_lemmantize(sent):
    spacy_nlp = spacy.load('en_core_web_sm')
    word_tokens = spacy_nlp(sent)
    spacy_lemmantized = []
    for token in word_tokens:     
        spacy_lemmantized.append(token.text)  
    return spacy_lemmantized


@app.route('/caseAn', methods=['POST'])
def caseAn():
    input_text = request.form["1"]
    data = ""
    dataset = []
    doc_names = []
    dataset.append(input_text)
    for i in os.listdir('./Datasets/CaseData/data/'):
        if i.endswith(".pdf"):
        #print(i)
            doc_names.append(i)
            name = './Datasets/CaseData/data/' + i
            file = open(name, 'rb')
            pdfReader = PyPDF2.PdfFileReader(file)
            for x in range(pdfReader.numPages):
                text = pdfReader.getPage(x).extractText()
                data = data + text
            dataset.append(data)
    print(doc_names)
    text_sents = [strip_data(i) for i in dataset]

    text_sents_stop = []
    for sent in text_sents:
        t = stop_words(sent)
        text=' '.join(t)
        #print(text)
        text_sents_stop.append(text)
    text_sents_stem =[]
    for sent in text_sents_stop:
        t = spacy_lemmantize(sent)
        text=' '.join(t)
        text_sents_stem.append(text)

    ##Word Embedding Model Implementation
    doc = text_sents_stem.pop(0)
    print(doc)
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(text_sents_stem)

    queryTFIDF = TfidfVectorizer().fit(text_sents_stem)
    queryTFIDF = queryTFIDF.transform([doc])

    cosine_similarities = cosine_similarity(queryTFIDF, tfidf).flatten()
    related_indices = cosine_similarities.argsort()[:-11:-1]
    print(related_indices)

    output=[]
    for id in related_indices:
        output.append(doc_names[id])
    return render_template('TextAn.html',output=output)

if __name__ == '__main__':
    app.run(debug=True)