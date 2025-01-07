import streamlit as st
import pickle
import numpy as np
import pandas as pd

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

with open('rf_model.pkl','rb') as file:
    model = pickle.load(file)

st.title('Deployment of ML model..')
st.subheader('Testing of deployment')

st.divider()

tab1, tab2, tab3 = st.tabs(['Single prediction','Bulk prediction', 'Selection of Model'])

with tab1:
    st.subheader('House information:')
    
    size = st.text_input('House size (in sqft)')
    
    lot_size = st.text_input('Lot size (in sqft)')
    
    bath = st.number_input('No. of bathroom', 1, 4)
    
    bed = st.number_input('No. of bedroom', 1, 5)
    
    zip = st.selectbox("Zip code",[98144, 98106, 98107, 98199])
    
    startPredict = st.button('Estimate Price')
    
    if startPredict==True:
    
        size = int(size)
        lot_size = int(size)
        zip = int(zip)
    
        data = np.array([bed, bath, size, lot_size, zip])
        # st.write(data)
        
        data = scaler.transform(data.reshape(1,-1))
        # st.write(data)
    
        result = model.predict(data)
        st.write(f"Predicted price: {result}")

    st.divider()

with tab2:

    file = st.file_uploader('Upload testing data')
    if file is not None:
        test_data = pd.read_csv(file)
        st.dataframe(test_data)
    
        test_data = scaler.transform(test_data)
    
        result = model.predict(test_data)
        st.write("Predicted results")
        st.write(result)

    st.divider()

with tab3:
    select_model = st.selectbox('Choose model',['RF','DT','DNN'])
    
    if select_model == 'RF':
        with open('rf_model.pkl','rb') as file:
            model_1 = pickle.load(file)
        
        
    elif select_model == 'DT':
        with open('dt_model.pkl','rb') as file:
            model = pickle.load(file)

