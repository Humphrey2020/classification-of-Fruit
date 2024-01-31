import streamlit  as st 
import numpy as np
import joblib as jb
import pandas as pd

st.title('Detector of Apple')  
st.sidebar.header('INPUT VALUES') 

model=jb.load('knn.py')

def input_values():
     Size=st.sidebar.slider('Size',-8,7,1) 
     Weight=st.sidebar.slider('Weight',-7,5,1) 
     Sweetness=st.sidebar.slider( 'Sweetness',-7,7,1) 
     Crunchiness =st.sidebar.slider('Crunchiness',-7,5,1) 
     Juiciness=st.sidebar.slider('Juiciness',-7,5,1) 
     Ripeness=st.sidebar.slider('Ripeness',-7,5,1) 
     Acidity=st.sidebar.slider('Acidity',-7,5,1) 

     data={'Size':Size ,'Weight':Weight,'Sweetness':Sweetness, 'Crunchiness':Crunchiness,'Juiciness':Juiciness,
          'Ripeness':Ripeness ,'Acidity':Acidity } 
     
     features=pd.DataFrame(data,index=[0])
     return features
x_values=input_values() 
st.write(x_values)
predictions_knn=model.predict(x_values) 
st.write(predictions_knn)
prob_knn=model.predict_proba(x_values) 
st.write(prob_knn)