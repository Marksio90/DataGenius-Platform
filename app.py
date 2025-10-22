import streamlit as st
from frontend.ui_components import sidebar

st.set_page_config(page_title="TMIV – Advanced ML Platform", layout="wide")
sidebar()

st.title("TMIV – Advanced ML Platform v2.0 Pro")
st.success("Szkielet projektu gotowy. Dodamy funkcje krok po kroku.")
st.write("➡️ Zacznij od wrzucenia danych w zakładce **Analiza Danych** (wkrótce).")
