"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import numpy as np # required for processing functions

# Vectorizer
# news_vectorizer = open("resources/tfidfvect.pkl","rb")
# tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Vectorizer modified
news_vectorizer = open("resources/countvec_randfr_1.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# Processing cleaning functions
# Convert all string entries to lower case.
def to_lower(df):
    """
    Changes all string values to lower case in dataframe column.
    
    
    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame to clean

    Returns
    -------
    df : pandas DataFrame
            The cleaned DataFrame
    """
    
    df["message"] = df["message"].str.lower()
    return df

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Clean the input with processing functions
			tt_df = pd.DataFrame(np.array([tweet_text]), columns=["message"])
			st.write(tt_df[['message']])
			clean_tt_df = to_lower(tt_df)
			st.write(clean_tt_df[['message']])
			vect_text = tweet_cv.transform(clean_tt_df["message"]).toarray()

			# Transforming user input with vectorizer (original)
			# vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			# predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			# prediction = predictor.predict(vect_text)

			# Model modified
			predictor = joblib.load(open(os.path.join("resources/model_randfr_1.pkl"),"rb")) # load trained random_forest model
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))
	
	# Creating sidebar with selection box -
	# you can create multiple pages this way
	members = ["Leon", "Saveshnee", "Kiren", "Kwanda", "Thato", "Rinkie"]
	choose = st.sidebar.selectbox("Choose Team Member", members)
	
	# Building out the "Information" page
	if choose == "Leon":
		st.info("Quick Intro")
		# You can read a markdown file from supporting resources folder
		st.markdown(
			"To tell you more about myself: I was born in Mthatha, my parents were teachers there and then we moved to the Western Province and we live close to the ocean.  \n"
			"I attended the high school and went on to Stellenbosch University where I studied Mechanical Engineering. During my Varsity years.  \n"
			"I was in Eendrag Mens Residence and played too much rugby and enjoyed going out with friends to the pub Die Akker  \n"
			"I worked for a few years and then joined the Explore DS Academy. Which is really fun and I am learning a lot.  \n"
			"I enjoy surfing, running, biking and golfing. That’s my basic intro."
		)

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
