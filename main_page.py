import streamlit as st
from PIL import Image
st.set_page_config(
    page_title= "Welcome",
    page_icon="🌺",
)

st.write("# Welcome to ProfiNetics! ")

st.sidebar.success("Select the feature to experience.")

logo = Image.open('logo_fin.png')
st.image(logo, caption="Logo of ProfiNetics")
st.markdown(
    """
    **ProfiNetics is a portfolio optimization tool currently in development.**
ProfiNetics has the capability to assess and compare various figures and perform factor analysis based on historical data. The tool utilizes advanced
optimizers to allocate the customer’s budget such as Efficient Frontier, SemiVariance, Mean-Variance (MeanVar), Monte Carlo Value at Risk (McVaR),
Minimum Variance (MIN Var), and Hierarchical Risk Parity (HRP),
mean, and risk moment to manage portfolios. Furthermore, ProfiNetics calculates the estimated future returns of each state-of-the-art model by using
AutoML, a cutting-edge technology that automates machine learning tasks.
The combination of these advanced methods results in a tool that is capable of providing high-quality investment advice to investors and has the potential
to significantly impact the portfolio optimization market.
    """
)