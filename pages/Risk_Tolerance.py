import streamlit as st


st.set_page_config(page_title= "Risk Tolerance Evaluator", page_icon = "🔰")

st.title('Risk Tolerance Evaluator')

# Define the questions and choices
questions = [
    "How comfortable are you with the idea of losing some or all of your investment in exchange for potentially higher returns?",
    "What is your investment time horizon?",
    "What is your preferred investment style?",
    "How would you react if your investment portfolio lost 20% of its value in a short period of time?",
]

choices = [
    ["Not Comfortable at all", "Somewhat Comfortable", "Neutral", "Comfortable", "Very Comfortable"],
    ["Short-term (1-3 years)", "Medium-term (3-5 years)", "Long-term (5+ years)"],
    ["Conservative (Lower Risk, Lower Return)", "Moderate (Balanced Risk and Return)", "Aggressive (Higher Risk, Higher Return)"],
    ["Panic and consider selling", "Hold on and monitor closely", "Buy more while prices are low"],
]

# Function to calculate risk score
def calculate_risk_score(answers):
    score = 0
    for i, answer in enumerate(answers):
        if i == 0:  # Question 1: Risk Tolerance
            if answer == "Not Comfortable at all":
                score += 1
            elif answer == "Somewhat Comfortable":
                score += 2
            elif answer == "Neutral":
                score += 3
            elif answer == "Comfortable":
                score += 4
            elif answer == "Very Comfortable":
                score += 5
        elif i == 3:  # Question 4: Reaction to Loss
            if answer == "Panic and consider selling":
                score += 1
            elif answer == "Hold on and monitor closely":
                score += 3
            elif answer == "Buy more while prices are low":
                score += 5
    return score

# Function to evaluate risk profile
def evaluate_risk_profile(score):
    if score <= 6:
        return "Risk Averse / Low Risk & Stable Return"
    elif 6 < score <= 10:
        return "Moderate Risk Taker / Medium Risk & Medium Return"
    else:
        return "Risk Seeker (Gambler) / High Risk & High Return"

def match_type(type):
    if type == "Risk Seeker (Gambler) / High Risk & High Return":
        return "Recommending you to use Mean Variance Optimization or Sharpe Ratio Optimization"
    elif type == "Moderate Risk Taker / Medium Risk & Medium Return":
        return "Recommending you to use Efficient Frontier or Minimum Variance Optimization"
    else:
        return "Recommending you to use Hierichial Risk Parity (HRP)"

# Function to display questions and get user responses
def risk_tolerance_questionnaire():
    st.title("Investment Risk Tolerance Questionnaire")
    
    answers = []
    
    for i, question in enumerate(questions):
        st.write(f"**Question {i+1}:** {question}")
        answer = st.radio("", choices[i])
        answers.append(answer)
        st.write("---")
    
    risk_score = calculate_risk_score(answers)
    risk_profile = evaluate_risk_profile(risk_score)
    optimize_method = match_type(risk_profile)
    
    st.write(f"### Risk Score: {risk_score}")
    st.write(f"### Risk Profile: {risk_profile}")
    st.write(f"### Recommeded Opinion: {optimize_method}")

# Run the function
risk_tolerance_questionnaire()