search_com = st.selectbox("Please choose the company youwant to know more." , filtered)



# Ticker information
tickerData = yf.Ticker(search_com)
string_logo = '<img src=%s>' % tickerData.info['logo_url']
st.markdown(string_logo, unsafe_allow_html=True)

string_name = tickerData.info['longName']
st.header('**%s**' % string_name)

string_summary = tickerData.info['longBusinessSummary']
st.info(string_summary)
#st_display_info(search_com)