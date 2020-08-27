import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image
#Custom modules
#Module to predict values
import predict as pt

#Module to get values from keys
import get_value as gv

#Module to retrain model with new data
import model_latest as ml

def main():
    
    img=Image.open('Absentism_pic.png')
    st.image(img,width=700)
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Predicting Absent Hours for Employees </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    
    #Option to user whether bulk upload data or Single Sample check
    st.subheader('Select a Data Input Options')
    Data_Option=st.selectbox('',['--Select an option--','Upload csv file for bulk data','Input single data via UI','Re-train Model'])
                             
    if Data_Option =='Upload csv file for bulk data':
       #Data input via CSV                      
        st.subheader('Instructions')
        st.set_option('deprecation.showfileUploaderEncoding', False)
        st.write('Please upload a .CSV file')
        st.write('Please maintain the below columns in the exact same sequence in your file .')
        st.write('R_code,Transportation expense,Office Distance,Age,Body mass index')
        st.write('As of now we are not handling empty cells')
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            
            try:
                uploaded_data=pd.read_csv(uploaded_file)
                if st.button('Predict Absent Hours'):
                    pred_hours=pt.predict(uploaded_data)
                    #st.write(type(pred_hours))
                    Pred_hours=pd.Series(pred_hours,name='Predicted_Absent_hours')
                    FinalData=pd.concat([uploaded_data,Pred_hours],axis=1)
                    st.subheader('Predictions')
                    st.write(FinalData)
            except:
                st.write('Please follow the instructions !!')
                 
    elif Data_Option =='Input single data via UI':

        #Input for Reason for Absence of the Employee
        reason=st.selectbox("Select Reason for Absense ", tuple(gv.reason_label.keys()))
        st.write('You selected the reason as :',reason)
        v_reason=gv.get_value(reason,gv.reason_label)
        st.write('Reason Code : ',v_reason)

        #Input for Office travel expenses

        Travel_Expense=st.number_input('Travel Expenses to Office in Dollars ',1.0,1000.0)
        st.write('Travel Expenses to Office as :',Travel_Expense)

        #Input for Employees Distance from Office
        Distance=st.number_input('Distance from Residence to Work in kms',1.0,100.0)
        st.write('You Selected Distance as :',Distance)


        #Input Employees Age
        Age=st.number_input('Enter Employee Age in Years',0.0, 100.0)
        st.write('You entered Employee Age as ', Age ,'years')


        #Input for Employess Body Mass Index
        BMI=st.number_input('Enter a BMI index',0.0,100000.0)
        st.write('You selected Body Mass Index as :',BMI)
        pred_hours=''

        #creating a dataframe to give input to the code
        columns=['R_code','Transportation expense','Office Distance','Age','Body mass index']
        df_test=pd.DataFrame({'R_code':[v_reason],'Transportation expense':[Travel_Expense],'Office Distance':[Distance],'Age':[Age], 'Body mass index':[BMI]},columns=columns)

        if st.button('Predict Absent Hours'):

            pred_hours=pt.predict(df_test)
            st.success('Prediction {}'.format(pred_hours[0]))

    elif Data_Option =='Re-train Model':
        st.subheader('Instructions')
        st.set_option('deprecation.showfileUploaderEncoding', False)
        st.write('Please upload a training file in .CSV format')
        st.write('Please maintain the below columns in the exact same sequence in your file .')
        st.write('R_code,Transportation expense,Office Distance,Age,Body mass index','Absnt_hr_estimate')
        st.write('As of now we are not handling empty cells')
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")

        if uploaded_file is not None:

            uploaded_data = pd.read_csv(uploaded_file)
            if st.button('Re-Train Model'):
                ReTrain_status = ml.re_train(uploaded_data)

                st.success(ReTrain_status)


    else :
        st.write('')

if __name__=='__main__':
    main()
    


