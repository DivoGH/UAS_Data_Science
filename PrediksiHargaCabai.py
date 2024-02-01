import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
from PIL import Image

img = Image.open('upb.png')
df = pd.read_csv('HB.csv')

st.image(img, width=260)
st.title('Prediksi Harga Beras Premium')

X = df[['Tahun']]
y = df['Cabai Merah Keriting']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

ip_1 = st.number_input ("Masukan Tanggal YYYYMMDD (Ex : 12 Jan 2024 = 20240112)", 0)

est = model.predict([[ip_1]])

if st.button ("Cek Prediksi Harga") :
    est = model.predict([[ip_1]])
    st.success(f"Prediksi Harga Beras Premium = IDR {est}")
    
    
