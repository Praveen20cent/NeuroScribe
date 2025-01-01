from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



# Page Configuraion
st.set_page_config(
    page_title ='EEG Alphabet Predictor App',
    layout = 'wide',
    initial_sidebar_state = 'expanded',
)

# Title of the Application
st.title('EEG Alphabet Predictor')

# Load Dataset
data_file = st.file_uploader("Upload Your CSV file",type=['csv'])

if data_file:
    df = pd.read_csv(data_file)
    st.info("EEG Dataset Uploaded")

else:
    # Default Dataset if user did not upload any csv file.
    df = pd.read_csv('train.csv')
    st.info("Using default Dataset")

# Dataset Display and filtering rows which have target values in {0,1,2}
# corresponding to {'A','B'.'C'}.
st.subheader("Dataset Preview")
filtered_dataset = df[df['# Letter'].isin([0,1])]
#filtered_dataset = df[df['# Letter'].isin(range(26))]
st.write(filtered_dataset)

# Features and target seperated
st.sidebar.subheader('Input Features')
feature_columns = [col for col in df.columns if col not in ['Line', '# Letter']]
X = df[feature_columns]
y = df['# Letter']

# Model building.
rf = RandomForestClassifier(max_depth = 10,max_features=5,n_estimators=300,random_state=42)
x_train,x_test,y_train,y_test = train_test_split (X,y,test_size=0.2,random_state=42)
rf.fit(x_train,y_train)


# Selecting top 6 features.
importance = rf.feature_importances_

# Displaying the importance of each feature.
feature_importance_df = pd.DataFrame({
    'Feature':feature_columns,
    'Importance':importance
}).sort_values(by = 'Importance',ascending=False)
top_features = feature_importance_df.head(10)
top_features_names = top_features['Feature'].tolist()


# Data Splitting.
# Setting a random_state makes sure the data split is the same every time the code is running.
x_train,x_test,y_train,y_test = train_test_split (X[top_features_names],y,test_size=0.3,random_state=45)

# Train the model with only top features.
rf.fit(x_train,y_train)

st.subheader('Displaying importance of each feature')
# Displaying importance of each feature in the form of bar chart.
st.bar_chart(feature_importance_df.set_index('Feature'))

# Input Widges which only consist of Top features as input.
input_features = pd.DataFrame([{feature:st.sidebar.slider(
    feature,
    float(filtered_dataset[feature].min()),
    float(filtered_dataset[feature].max()),
    float(filtered_dataset[feature].mean()))
    for feature in top_features_names}],columns=top_features_names)
st.subheader('Input Features')
st.write(input_features)


# Prediction part
number_to_alphabet = {0: 'A', 1: 'B'}
prediction = rf.predict(input_features)[0]
pred_Alpha = number_to_alphabet.get(prediction)
st.subheader("Prediction")
st.metric(label='Predicted Alphabet', value=pred_Alpha)

# Displaying Accuracy
st.subheader("Accuracy:")
y_pred = rf.predict(x_test)
st.write(accuracy_score(y_pred,y_test))


# Confusion Matrix
st.subheader("Confusion Matrix")
fig, axis = plt.subplots(figsize = (5,5))
cm = confusion_matrix(y_test, y_pred,labels=[0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["A","B"])
disp.plot(cmap = 'Greens',ax=axis)
plt.title("Confusion matrix")
st.pyplot(fig)



