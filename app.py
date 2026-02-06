import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import pickle


lg = pickle.load(open('placement.pkl', 'rb'))
img=Image.open('jobplacement.jpg')
st.image(img,width=700)
st.title('Job Placement Prediction Model') 