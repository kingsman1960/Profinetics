import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
from pandas_datareader import data as web
import base64
import sua
import matplotlib.pyplot as plt

st.set_page_config(page_title= "ProfiNetics", page_icon = "💸")

st.title('Portfolio Optimization App')


START = st.date_input('Start', value = pd.to_datetime('2017-06-01', format='%Y-%m-%d'))
TODAY = st.date_input('Today', value = pd.to_datetime('today', format='%Y-%m-%d'))

stocks = ('consumer', 'nasdaq_techtele','nyse_techtele', 'nasdaq_nyse_energy', 'nasdaq_nyse_financial', 'nasdaq_nyse_health', 'nasdaq_nyse_industrial', 'nasdaq_nyse_property', 'nasdaq_nyse_utility', 'nyse_consumer_discre')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

ticker = pd.read_csv("./pages/dataset/" + selected_stock + ".csv")['Symbol']



@st.cache_data(experimental_allow_widgets=True)
def load_data(ticker):
	
	yf.pdr_override()
	stock_list = ticker.to_list()
	#data = pd.DataFrame()
	#ticker = ticker.str.split(';').str.join(',')
	data = web.get_data_yahoo(stock_list, START, TODAY)['Adj Close']
    #data.reset_index(inplace=True)
	return data

	
data_load_state = st.text('Loading data...')
data = load_data(ticker)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.dataframe(data)



    # Use the Streamlit theme.
    # This is the default. So you can also omit the theme argument.
#st.subheader('Historical Adjusted Price from listed stocks')
#st.line_chart(data)

# Plot raw data
#def plot_raw_data():
	#rel = data.reset_index(inplace=True)
	#fig = go.Figure()
	#fig.add_trace(go.Scatter(x=data.iloc[:, 1], y=data['SONY'], name="Adj Close"))
	#fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	#fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	#st.plotly_chart(fig)
	
#plot_raw_data()

budget = st.number_input('Budget to invest:', min_value = 2000)
sel_optimizer = st.selectbox(
    'How would you like to optimize the portfolio?',
    ('EF', 'HRP', 'SemiVar','mCVaR', "MINVAR","MEANVAR"))
sel_returns = st.selectbox(
    'Which return model would you like to choose?',
    ('mean_historical_return', 'ema_historical_return', 'capm_return'))


#@st.cache(suppress_st_warning=True)
def EF_Cal(data):
	from sua import EfficientFrontier
	from sua import risk_models
	from sua import expected_returns
	

# Calculate the expected annual returns and the annulized sample covariance matrix of the daily asset  returns

	mu = expected_returns.mean_historical_return(data)
	S = risk_models.CovarianceShrinkage(data).ledoit_wolf()
	#from sua import objective_functions, base_optimizer
	ef = EfficientFrontier(mu, S)
	#ef.add_objective(objective_functions.L2_reg, gamma=2)
	weights = ef.max_sharpe()
	#weights = ef.nonconvex_objective(
       #objective_functions.sharpe_ratio,
       #objective_args=(ef.expected_returns, ef.cov_matrix),
       #weights_sum_to_one=True,)
	cleaned_weights = ef.clean_weights()
	st.write(cleaned_weights)
	st.write(ef.portfolio_performance(verbose=True))

	from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

	portfolio_val = budget
	latest_prices = get_latest_prices(data)
	weights = cleaned_weights
	da = DiscreteAllocation(weights, latest_prices, total_portfolio_value = portfolio_val)
	allocation, leftover = da.lp_portfolio(reinvest=True ,verbose = True)
	st.write('Discrete allocation:', allocation)
	st.write('Funds remaining: $', leftover)

def EF_EMA_Cal(data):
	from sua import EfficientFrontier
	from sua import risk_models
	from sua import expected_returns
	

# Calculate the expected annual returns and the annulized sample covariance matrix of the daily asset  returns

	mu = expected_returns.ema_historical_return(data)
	S = risk_models.CovarianceShrinkage(data).ledoit_wolf()
	#from sua import objective_functions, base_optimizer
	ef = EfficientFrontier(mu, S)
	#ef.add_objective(objective_functions.L2_reg, gamma=2)
	weights = ef.max_sharpe()
	#weights = ef.nonconvex_objective(
       #objective_functions.sharpe_ratio,
       #objective_args=(ef.expected_returns, ef.cov_matrix),
       #weights_sum_to_one=True,)
	cleaned_weights = ef.clean_weights()
	st.write(cleaned_weights)
	st.write(ef.portfolio_performance(verbose=True))

	from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

	portfolio_val = budget
	latest_prices = get_latest_prices(data)
	weights = cleaned_weights
	da = DiscreteAllocation(weights, latest_prices, total_portfolio_value = portfolio_val)
	allocation, leftover = da.lp_portfolio(reinvest=True ,verbose = True)
	st.write('Discrete allocation:', allocation)
	st.write('Funds remaining: $', leftover)

def EF_CAPM_Cal(data):
	from sua import EfficientFrontier
	from sua import risk_models
	from sua import expected_returns
	

# Calculate the expected annual returns and the annulized sample covariance matrix of the daily asset  returns

	mu = expected_returns.capm_return(data)
	S = risk_models.CovarianceShrinkage(data).ledoit_wolf()
	#from sua import objective_functions, base_optimizer
	ef = EfficientFrontier(mu, S)
	#ef.add_objective(objective_functions.L2_reg, gamma=2)
	weights = ef.max_sharpe()
	#weights = ef.nonconvex_objective(
       #objective_functions.sharpe_ratio,
       #objective_args=(ef.expected_returns, ef.cov_matrix),
       #weights_sum_to_one=True,)
	cleaned_weights = ef.clean_weights()
	st.write(cleaned_weights)
	st.write(ef.portfolio_performance(verbose=True))

	from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

	portfolio_val = budget
	latest_prices = get_latest_prices(data)
	weights = cleaned_weights
	da = DiscreteAllocation(weights, latest_prices, total_portfolio_value = portfolio_val)
	allocation, leftover = da.lp_portfolio(reinvest=True ,verbose = True)
	st.write('Discrete allocation:', allocation)
	st.write('Funds remaining: $', leftover)

def HRP_Cal(data):
	from sua import HRPOpt
	from sua import expected_returns

	daily_returns = expected_returns.returns_from_prices(data).dropna()
	
	# run the optimization algorithm to get the weights:
	hrp = HRPOpt(daily_returns)
	hrp_weights = hrp.optimize()

	# performance of the portfolio and the weights:
	hrp.portfolio_performance(verbose=True)
	hrp_weights = dict(hrp_weights)
	st.write(hrp_weights)

	from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
	latest_prices = get_latest_prices(data)
	da_hrp = DiscreteAllocation(hrp_weights, latest_prices, total_portfolio_value=budget)

	allocation, leftover = da_hrp.lp_portfolio(reinvest=True ,verbose = True)
	st.write("Discrete allocation (HRP):", allocation)
	st.write("Funds remaining (HRP): ${:.2f}".format(leftover))

def MINVAR_Cal(data):
	from sua import expected_returns
	from sua import risk_models
	from sua import EfficientFrontier
	from sua import objective_functions

	mu = expected_returns.capm_return(data)
	S = risk_models.CovarianceShrinkage(data).ledoit_wolf()
	ef = EfficientFrontier(mu, S)
	ef.add_objective(objective_functions.L2_reg, gamma=2)
	ef.min_volatility()
	cleaned_weights = ef.clean_weights()
	st.write(cleaned_weights)
	st.write(ef.portfolio_performance(verbose=True))

	from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

	portfolio_val = budget
	latest_prices = get_latest_prices(data)
	weights = cleaned_weights
	da = DiscreteAllocation(weights, latest_prices, total_portfolio_value = portfolio_val)
	allocation, leftover = da.lp_portfolio(reinvest=True ,verbose = True)
	st.write('Discrete allocation:', allocation)
	st.write('Funds remaining: $', leftover)

def MEANVAR_Cal(data, vol_max=0.15):
	from sua import expected_returns
	from sua import risk_models
	from sua import EfficientFrontier
	from sua import objective_functions

	mu = expected_returns.capm_return(data)
	S = risk_models.CovarianceShrinkage(data).ledoit_wolf()
	ef = EfficientFrontier(mu, S)
	ef.add_objective(objective_functions.L2_reg, gamma=2)
	ef.efficient_risk(vol_max)
	cleaned_weights = ef.clean_weights()
	st.write(cleaned_weights)
	st.write(ef.portfolio_performance(verbose=True))

	from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

	portfolio_val = budget
	latest_prices = get_latest_prices(data)
	weights = cleaned_weights
	da = DiscreteAllocation(weights, latest_prices, total_portfolio_value = portfolio_val)
	allocation, leftover = da.lp_portfolio(reinvest=True ,verbose = True)
	st.write('Discrete allocation:', allocation)
	st.write('Funds remaining: $', leftover)

def mCVaR_Cal(data):
	from pypfopt.efficient_frontier import EfficientCVaR
	from sua import expected_returns
	import numpy as np

	mu= expected_returns.mean_historical_return(data)
	S = data.cov()
	ef_cvar = EfficientCVaR(mu, S)
	cvar_weights = ef_cvar.min_cvar()

	cleaned_weights = ef_cvar.clean_weights()
	st.write(dict(cleaned_weights))
	round(np.mean(ef_cvar.expected_returns) *100,2)

	np.seterr(invalid= 'ignore')
	from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
	latest_prices = get_latest_prices(data)
	da_cvar = DiscreteAllocation(cvar_weights, latest_prices, total_portfolio_value=budget)
	allocation, leftover = da_cvar.lp_portfolio(reinvest=True ,verbose = True)
	st.write("Discrete allocation (CVAR):", allocation)
	st.write("Funds remaining (CVAR): ${:.2f}".format(leftover))


def mCVaR_EMA_Cal(data):
	from pypfopt.efficient_frontier import EfficientCVaR
	from sua import expected_returns
	import numpy as np

	mu= expected_returns.ema_historical_return(data)
	S = data.cov()
	ef_cvar = EfficientCVaR(mu, S)
	cvar_weights = ef_cvar.min_cvar()

	cleaned_weights = ef_cvar.clean_weights()
	st.write(dict(cleaned_weights))
	round(np.mean(ef_cvar.expected_returns) *100,2)

	np.seterr(invalid= 'ignore')
	from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
	latest_prices = get_latest_prices(data)
	da_cvar = DiscreteAllocation(cvar_weights, latest_prices, total_portfolio_value=budget)
	allocation, leftover = da_cvar.lp_portfolio(reinvest=True ,verbose = True)
	st.write("Discrete allocation (CVAR):", allocation)
	st.write("Funds remaining (CVAR): ${:.2f}".format(leftover))


def mCVaR_CAPM_Cal(data):
	from pypfopt.efficient_frontier import EfficientCVaR
	from sua import expected_returns
	import numpy as np

	mu= expected_returns.capm_return(data)
	S = data.cov()
	ef_cvar = EfficientCVaR(mu, S)
	cvar_weights = ef_cvar.min_cvar()

	cleaned_weights = ef_cvar.clean_weights()
	st.write(dict(cleaned_weights))
	round(np.mean(ef_cvar.expected_returns) *100,2)

	np.seterr(invalid= 'ignore')
	from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
	latest_prices = get_latest_prices(data)
	da_cvar = DiscreteAllocation(cvar_weights, latest_prices, total_portfolio_value=budget)
	allocation, leftover = da_cvar.lp_portfolio(reinvest=True ,verbose = True)
	st.write("Discrete allocation (CVAR):", allocation)
	st.write("Funds remaining (CVAR): ${:.2f}".format(leftover))



def SemiVar_Cal(data):
	from pypfopt import EfficientSemivariance
	#from sua import EfficientFrontier
	#from sua import risk_models
	from sua import expected_returns

	mu = expected_returns.mean_historical_return(data)
	historical_returns = expected_returns.returns_from_prices(data)

	es = EfficientSemivariance(mu, historical_returns)
	es.efficient_return(0.20)

	# We can use the same helper methods as before
	weights = es.clean_weights()
	st.write(weights)
	st.write(es.portfolio_performance(verbose=True))

	from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

	portfolio_val = budget
	latest_prices = get_latest_prices(data)
	#weights = cleaned_weights
	da = DiscreteAllocation(weights, latest_prices, total_portfolio_value = portfolio_val)
	allocation, leftover = da.lp_portfolio(reinvest=True ,verbose = True)
	st.write('Discrete allocation (Semi-Var):', allocation)
	st.write('Funds remaining (Semi-Var): $', leftover)

def SemiVar_EMA_Cal(data):
	from pypfopt import EfficientSemivariance
	#from sua import EfficientFrontier
	#from sua import risk_models
	from sua import expected_returns

	mu = expected_returns.ema_historical_return(data)
	historical_returns = expected_returns.returns_from_prices(data)

	es = EfficientSemivariance(mu, historical_returns)
	es.efficient_return(0.20)

	# We can use the same helper methods as before
	weights = es.clean_weights()
	st.write(weights)
	st.write(es.portfolio_performance(verbose=True))

	from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

	portfolio_val = budget
	latest_prices = get_latest_prices(data)
	#weights = cleaned_weights
	da = DiscreteAllocation(weights, latest_prices, total_portfolio_value = portfolio_val)
	allocation, leftover = da.lp_portfolio(reinvest=True ,verbose = True)
	st.write('Discrete allocation (Semi-Var):', allocation)
	st.write('Funds remaining (Semi-Var): $', leftover)


def SemiVar_CAPM_Cal(data):
	from pypfopt import EfficientSemivariance
	#from sua import EfficientFrontier
	#from sua import risk_models
	from sua import expected_returns


	mu = expected_returns.capm_return(data)
	historical_returns = expected_returns.returns_from_prices(data)

	es = EfficientSemivariance(mu, historical_returns)
	es.efficient_return(0.20)

	# We can use the same helper methods as before
	weights = es.clean_weights()
	st.write(weights)
	st.write(es.portfolio_performance(verbose=True))

	from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

	portfolio_val = budget
	latest_prices = get_latest_prices(data)
	#weights = cleaned_weights
	da = DiscreteAllocation(weights, latest_prices, total_portfolio_value = portfolio_val)
	allocation, leftover = da.lp_portfolio(reinvest=True ,verbose = True)
	st.write('Discrete allocation (Semi-Var):', allocation)
	st.write('Funds remaining (Semi-Var): $', leftover)


if sel_optimizer == "EF" and sel_returns == "mean_historical_return":
	st.info('Efficient Frontier object (inheriting from Base Convex Optimizer) contains multiple optimization methods that can be called (corresponding to different objective functions) with various parameters.', icon="ℹ️")
	st.info('Calculate annualised mean (daily) historical return from input (daily) asset prices with using default geometric mean (CAGR).', icon="ℹ️")
	EF_Cal(data)

elif sel_optimizer == "EF" and sel_returns == "ema_historical_return":
	st.info('Efficient Frontier object (inheriting from Base Convex Optimizer) contains multiple optimization methods that can be called (corresponding to different objective functions) with various parameters.', icon="ℹ️")
	st.info('Calculate the exponentially-weighted mean of (daily) historical returns, giving higher weight to more recent data.', icon="ℹ️")
	EF_EMA_Cal(data)

elif sel_optimizer == "EF" and sel_returns == "capm_return":
	st.info('Efficient Frontier object (inheriting from Base Convex Optimizer) contains multiple optimization methods that can be called (corresponding to different objective functions) with various parameters.', icon="ℹ️")
	st.info('Compute a return estimate using the Capital Asset Pricing Model. Under the CAPM, asset returns are equal to market returns plus a eta term encoding the relative risk of the asset.', icon="ℹ️")
	EF_CAPM_Cal(data)


elif sel_optimizer == "mCVaR" and sel_returns == "mean_historical_return":
	st.info('Calculate annualised mean (daily) historical return from input (daily) asset prices with using default geometric mean (CAGR).', icon="ℹ️")
	mCVaR_Cal(data)

elif sel_optimizer == "mCVaR" and sel_returns == "ema_historical_return":
	st.info('Calculate the exponentially-weighted mean of (daily) historical returns, giving higher weight to more recent data.', icon="ℹ️")
	mCVaR_EMA_Cal(data)
elif sel_optimizer == "mCVaR" and sel_returns == "capm_return":
	st.info('Compute a return estimate using the Capital Asset Pricing Model. Under the CAPM, asset returns are equal to market returns plus a eta term encoding the relative risk of the asset.', icon="ℹ️")
	mCVaR_CAPM_Cal(data)

elif sel_optimizer == "SemiVar" and sel_returns == "mean_historical_return":
	st.info('Calculate annualised mean (daily) historical return from input (daily) asset prices with using default geometric mean (CAGR).', icon="ℹ️")
	SemiVar_Cal(data)

elif sel_optimizer == "SemiVar" and sel_returns == "ema_historical_return":
	st.info('Calculate the exponentially-weighted mean of (daily) historical returns, giving higher weight to more recent data.', icon="ℹ️")
	SemiVar_EMA_Cal(data)

elif sel_optimizer == "SemiVar" and sel_returns == "capm_return":
	st.info('Compute a return estimate using the Capital Asset Pricing Model. Under the CAPM, asset returns are equal to market returns plus a eta term encoding the relative risk of the asset.', icon="ℹ️")
	st.warning('In some sectors. failure might occurs due to the target_return must be lower than the largest expected return', icon="⚠️")
	SemiVar_CAPM_Cal(data)

elif sel_optimizer == "HRP":
	HRP_Cal(data)
	st.info('Hierarchical Risk Parity is a novel portfolio optimization method developed by Marcos Lopez de Prado.', icon="ℹ️")

elif sel_optimizer == "MINVAR":
	MINVAR_Cal(data)
	st.info('A minimum variance portfolio is an investing method that helps you maximize returns and minimize risk.', icon="ℹ️")

elif sel_optimizer == "MEANVAR":
	MEANVAR_Cal(data)
	st.info('Maximise return for a target risk. The resulting portfolio will have a volatility less than the target (but not guaranteed to be equal).', icon="ℹ️")


st.subheader('Portfolio Result Estimation')

st.info('Please select EF in case of SemiVAR, mCVAR', icon="ℹ️")
filtered = st.multiselect("Please choose the filtered stocks from previous stage" , ticker)
sel_optimizer2 = st.selectbox(
    'How would you like to optimize the portfolio?',
    ('EF', 'HRP',"MEANVAR", "MINVAR"))
#The HRP method works by finding subclusters of similar assets based on returns and constructing a hierarchy from these clusters to generate weights for each asset.
#HRP does not require inverting of a covariance matrix, which is a measure of how stock returns move in the same direction.
#HRP is not as sensitive to outliers.

#want to know more about the company?

import yfinance as yf


def st_display_pdf(pdf_file):
	with open(pdf_file,"rb") as f:
		base64_pdf = base64.b64encode(f.read()).decode('latin-1')
	pdf_display = f'<embed src = "data:application/pdf;base64,{base64_pdf}" width = "800" height = "1000" type= "application/pdf">'
	st.markdown(pdf_display, unsafe_allow_html=True)
	
#@st.cache (allow_output_mutation=True)
def Start():
	from sua import sua, get_report_st, Start

	portfolio = Start(
    	start_date = START,
    	end_date= TODAY,
    	portfolio= filtered,
    	optimizer = sel_optimizer2,
    	#rebalance = "1y",  # rebalance every year
    	risk_manager = {"Stop Loss" : -0.2} # Stop the investment when the drawdown becomes superior to -20%
	)
	#quantity = sua.orderbook
	#portfolio=Start.reshape(6,5)
	 #For showing in JupyterNotebook
	get_report_st(portfolio) #if you want to st.write wirh pdf form, plz use this.
	#optimize_portfolio(portfolio)
	#st.plotly_chart(sel1, theme="streamlit", use_container_width=True)
	
	#quantity
Start()
#st_display_pdf('./report.pdf')






st.subheader('Portfolio Prediction')

# Importing prediction models, metrics and logging
import datetime as dt
from prophet import Prophet
from darts import *
from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values
from darts.metrics import mape, mase
import warnings
import logging

import datetime as dt
START_PRE = st.date_input('Start date of getting historical data:', value = dt.datetime(2017,6, 1))
END_PRE = st.date_input('End date of getting historical data:', value = dt.datetime(2022,12,30))
n_days = st.number_input('Days to predict:', min_value = 2)

def prediction2():
	global START_PRE, END_PRE, n_days
	from sua import prediction_st
	import datetime as dt

	start_date = dt.datetime.strftime(START_PRE, "%Y-%m-%d")
	end_date = dt.datetime.strftime(END_PRE, '%Y-%m-%d')
	#n_days = pd.to_datetime(n_days)
	#n_days = dt.datetime.strptime(str(n_days), '%d')
	#test = datetime.strptime(str(n_days), '%d').date()

  
	prediction_st(
    	  portfolio=filtered, #stocks you want to predict
    	  start_date = start_date,#date from which it will take data to predict
    	  end_date= end_date,
     	  #weights = [0.3, 0.2, 0.3, 0.2,0.2], #allocate 30% to TSLA and 20% to AAPL...(equal weighting  by default)
    	  prediction_days=n_days #number of days you want to predict
	)
	

prediction2()

