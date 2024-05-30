import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#Dataset used: https://archive.ics.uci.edu/dataset/936/national+poll+on+healthy+aging+(npha)
#  py -m streamlit run ai_streamlit_project.py

st.set_page_config(page_title="AI Mental Health inspector for elders")
title = st.title("Mental health evaluation for elders")
st.write("This website lets you fill a table with information about an adult between the age of 50 and 80 and will evaluate/predict their mental health state based on that information. This is not to be taken seriously and is only for experimental purposes. The dataset includes multiple features for elderly people, with the mental health feature being the class. We decided to omit the race label as we didn't find it relevant in this case.")


st.image("Elders.jpg", use_column_width=True)

with st.expander("Click here to see value guides"):
    st.write('''
Number of Doctors Visited: The total count of different doctors the patient has seen
\n1: 0-1 doctors
\n 2: 2-3 doctors
\n 3: 4 or more doctors
\n
‎ 
\nAge: The patient's age group
\n 1: 50-64
\n 2: 65-80
\n
‎
\nPhysical Health: A self-assessment of the patient's physical well-being
\n -1: Refused 
\n 1: Excellent 
\n 2: Very Good 
\n 3: Good
\n 4: Fair
\n 5: Poor
\n
‎
\nMental Health: A self-evaluation of the patient's mental or psychological health. This is the label being predicted, so it cannot be filled.
\n -1: Refused 
\n 1: Excellent 
\n 2: Very Good 
\n 3: Good
\n 4: Fair
\n 5: Poor
\n
‎
\nDental Health: A self-assessment of the patient's oral or dental health
\n -1: Refused 
\n 1: Excellent 
\n 2: Very Good 
\n 3: Good
\n 4: Fair
\n 5: Poor
\n
‎
\nEmployment: The patient's employment status or work-related information
\n -1: Refused 6
\n 1: Working full-time
\n 2: Working part-time
\n 3: Retired
\n 4: Not working at this time
\n
‎
\nStress Keeps Patient from Sleeping: Whether stress affects the patient's ability to sleep
\n 0: No
\n 1: Yes
\n
‎
\nMedication Keeps Patient from Sleeping: Whether medication impacts the patient's sleep
\n 0: No
\n 1: Yes
\n
‎
\nPain Keeps Patient from Sleeping: Whether physical pain disturbs the patient's sleep 
\n 0: No
\n 1: Yes
\n
‎
\nBathroom Needs Keeps Patient from Sleeping: Whether the need to use the bathroom affects the patient's sleep
\n 0: No
\n 1: Yes
\n
‎
\nUnknown Keeps Patient from Sleeping: Unidentified factors affecting the patient's sleep 
 \n0: No
 \n1: Yes
\n
‎
\nTrouble Sleeping: General issues or difficulties the patient faces with sleeping
 \n0: No
 \n1: Yes
\n
‎
\nPrescription Sleep Medication: Information about any sleep medication prescribed to the patient
 \n-1: Refused
 \n1: Use regularly
 \n2: Use occasionally
 \n3: Do not use
\n
‎
\nGender: The gender identity of the patient
 \n-2: Not asked
 \n-1: REFUSED
 \n1: Male
 \n2: Female
\n
    ''')
#machine learning
data = pd.read_csv("health.csv")

x = data.drop(['Mental Health','Race'], axis=1)
y = data['Mental Health']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
dt = DecisionTreeClassifier(criterion = 'entropy')
dt.fit(x_train, y_train)



xInput = x.iloc[0:1]
xInput.reset_index(drop=True,inplace=True)
xInput[:].values[:] = 1

xInput = st.data_editor(xInput)

predict = st.button("Evaluate")

if predict:
    y_pred = dt.predict(xInput)
    "Mental health:"
    ["Excellent","Very Good","Good","Fair","Poor"][y_pred[0]]

