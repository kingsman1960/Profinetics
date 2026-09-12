import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QComboBox, QPushButton, QDateEdit, QLineEdit, QTextEdit, 
                             QTableWidget, QTableWidgetItem, QHeaderView, QSplitter)
from PyQt5.QtCore import Qt, QDate
from PyQt5.QtGui import QFont, QColor
import pandas as pd
import yfinance as yf
from pandas_datareader import data as web
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

class ModernPortfolioApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ProfiNetics - Portfolio Optimization")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QMainWindow { background-color: #1C1C1C; color: #FFFFFF; }
            QLabel { color: #FFFFFF; font-size: 12px; }
            QComboBox, QDateEdit, QLineEdit { 
                background-color: #2C2C2C; 
                color: #FFFFFF; 
                border: 1px solid #3C3C3C;
                padding: 5px;
                font-size: 12px;
            }
            QPushButton { 
                background-color: #0066CC; 
                color: #FFFFFF; 
                border: none; 
                padding: 8px 15px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #0077EE; }
            QTextEdit, QTableWidget { 
                background-color: #2C2C2C; 
                color: #FFFFFF;
                font-size: 12px;
            }
        """)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.setup_ui()

    def setup_ui(self):
        # Left panel for inputs
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Date inputs
        date_widget = QWidget()
        date_layout = QHBoxLayout(date_widget)
        self.start_date = QDateEdit(QDate(2017, 6, 1))
        self.end_date = QDateEdit(QDate.currentDate())
        date_layout.addWidget(QLabel("Start:"))
        date_layout.addWidget(self.start_date)
        date_layout.addWidget(QLabel("End:"))
        date_layout.addWidget(self.end_date)
        left_layout.addWidget(date_widget)

        # Stock dataset selection
        self.stock_combo = QComboBox()
        stocks = ('consumer', 'nasdaq_techtele', 'nyse_techtele', 'nasdaq_nyse_energy', 'nasdaq_nyse_financial', 'nasdaq_nyse_health', 'nasdaq_nyse_industrial', 'nasdaq_nyse_property', 'nasdaq_nyse_utility', 'nyse_consumer_discre')
        self.stock_combo.addItems(stocks)
        left_layout.addWidget(QLabel("Dataset:"))
        left_layout.addWidget(self.stock_combo)

        # Budget input
        self.budget_input = QLineEdit()
        self.budget_input.setPlaceholderText("Enter budget")
        left_layout.addWidget(QLabel("Budget:"))
        left_layout.addWidget(self.budget_input)

        # Optimizer selection
        self.optimizer_combo = QComboBox()
        optimizers = ('EF', 'HRP', 'SemiVar', 'mCVaR', "MINVAR", "MEANVAR")
        self.optimizer_combo.addItems(optimizers)
        left_layout.addWidget(QLabel("Optimizer:"))
        left_layout.addWidget(self.optimizer_combo)

        # Returns model selection
        self.returns_combo = QComboBox()
        returns_models = ('mean_historical_return', 'ema_historical_return', 'capm_return')
        self.returns_combo.addItems(returns_models)
        left_layout.addWidget(QLabel("Returns Model:"))
        left_layout.addWidget(self.returns_combo)

        # Calculate button
        self.calculate_button = QPushButton("Optimize Portfolio")
        self.calculate_button.clicked.connect(self.calculate_optimization)
        left_layout.addWidget(self.calculate_button)

        left_layout.addStretch()

        # Right panel for results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Asset", "Weight", "Allocation"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right_layout.addWidget(QLabel("Portfolio Allocation:"))
        right_layout.addWidget(self.results_table)

        # Performance metrics
        self.performance_text = QTextEdit()
        self.performance_text.setReadOnly(True)
        right_layout.addWidget(QLabel("Performance Metrics:"))
        right_layout.addWidget(self.performance_text)

        # Add panels to main layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])
        self.main_layout.addWidget(splitter)

    def calculate_optimization(self):
        start_date = self.start_date.date().toPyDate()
        end_date = self.end_date.date().toPyDate()
        selected_stock = self.stock_combo.currentText()
        budget = float(self.budget_input.text())
        optimizer = self.optimizer_combo.currentText()
        returns_model = self.returns_combo.currentText()

        # Load data
        ticker = pd.read_csv(f"./pages/dataset/{selected_stock}.csv")['Symbol']
        yf.pdr_override()
        stock_list = ticker.to_list()
        data = web.get_data_yahoo(stock_list, start_date, end_date)['Adj Close']

        # Perform optimization (example using EF and mean_historical_return)
        if optimizer == "EF" and returns_model == "mean_historical_return":
            mu = expected_returns.mean_historical_return(data)
            S = risk_models.CovarianceShrinkage(data).ledoit_wolf()
            ef = EfficientFrontier(mu, S)
            weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights()
            performance = ef.portfolio_performance(verbose=True)

            # Discrete allocation
            latest_prices = get_latest_prices(data)
            da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=budget)
            allocation, leftover = da.lp_portfolio()

            # Display results in table
            self.results_table.setRowCount(len(allocation))
            for i, (asset, amount) in enumerate(allocation.items()):
                self.results_table.setItem(i, 0, QTableWidgetItem(asset))
                self.results_table.setItem(i, 1, QTableWidgetItem(f"{cleaned_weights[asset]:.2%}"))
                self.results_table.setItem(i, 2, QTableWidgetItem(str(amount)))

            # Display performance metrics
            performance_text = f"Expected annual return: {performance[0]:.2%}\n"
            performance_text += f"Annual volatility: {performance[1]:.2%}\n"
            performance_text += f"Sharpe Ratio: {performance[2]:.2f}\n"
            performance_text += f"Funds remaining: ${leftover:.2f}"
            self.performance_text.setText(performance_text)
        else:
            self.performance_text.setText("This combination is not implemented in the example.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModernPortfolioApp()
    window.show()
    sys.exit(app.exec_())