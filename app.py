import numpy as np
from PIL import Image
import streamlit as st
import pickle

# Load model
lg = pickle.load(open('placement.pkl', 'rb'))

# Load image
img = Image.open('jobplacement.jpg')
st.image(img, width=700)

st.title('Job Placement Prediction Model')

# Model expects EXACTLY 14 features
EXPECTED_FEATURES = 14

input_text = st.text_input(
    f"Enter {EXPECTED_FEATURES} features separated by commas"
)

if input_text:
    input_list = input_text.split(',')

    if len(input_list) != EXPECTED_FEATURES:
        st.error(
            f"❌ Model expects {EXPECTED_FEATURES} features, "
            f"but you entered {len(input_list)}"
        )
    else:
        try:
            np_df = np.array(input_list, dtype=float).reshape(1, -1)
            prediction = lg.predict(np_df)

            if prediction[0] == 1:
                st.success('🎉 Congratulations! You are likely to get placed.')
            else:
                st.warning('❌ Sorry, you are not likely to get placed.')

        except ValueError:
            st.error("⚠️ Please enter valid numeric values only.")
