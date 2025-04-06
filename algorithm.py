import numpy as np
from statsmodels.tsa.api import VAR, ARIMA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Custom trading Algorithm
class Algorithm():

    ########################################################
    # INITIALISATION
    ########################################################
    def __init__(self, positions):
        self.data = {}
        self.positionLimits = {}
        self.day = 0
        self.positions = positions
        self.var_model = None
        self.scaler = StandardScaler()
        self.lookback = 1.02      # You can experiment with lookback length here.
        self.threshold = 0.0001       # ARIMA threshold: try 0.0005 - 0.0007 if needed.
        self.lag_order = 1
        self.var_instruments = ['Fried Chicken', 'Raw Chicken', 'Secret Spices']
        self.totalDailyBudget = 600000
        self.Direction = 1           # This will be updated by the regression model.

    def get_current_price(self, instrument):
        return self.data[instrument][-1]

    ########################################################
    # MAIN POSITION CALCULATION
    ########################################################
    def get_positions(self):
        # Get current positions and limits
        currentPositions = self.positions
        positionLimits = self.positionLimits

        # Initialise desired positions for all instruments
        desiredPositions = {instrument: 0 for instrument in positionLimits.keys()}

        # Decide positions for each asset
        desiredPositions["UQ Dollar"] = self.get_uq_dollar_position(currentPositions["UQ Dollar"], positionLimits["UQ Dollar"])
        desiredPositions['Dawg Food'] = self.get_dwgFood_position()
        desiredPositions["Quantum Universal Algorithmic Currency Koin"] = self.get_quack_position(currentPositions["Quantum Universal Algorithmic Currency Koin"], positionLimits["Quantum Universal Algorithmic Currency Koin"])
        desiredPositions["Goober Eats"] = self.get_goober_eats_position()
        
        # Apply Purple Elixir strategy (as originally given)
        self.get_prplelixr_position(desiredPositions, positionLimits)
        
        # Fintech Token using ARIMA with tuned orders
        self.apply_arima_model("Fintech Token", positionLimits, desiredPositions, p=2, d=1, q=1)
        
        # Regression Model for Fried Chicken (tuned below)
        self.apply_regression_model(positionLimits, desiredPositions)
        
        # ARIMA for Secret Spices and Raw Chicken (default orders)
        self.apply_arima_model("Secret Spices", positionLimits, desiredPositions)
        self.apply_arima_model("Raw Chicken", positionLimits, desiredPositions)
        
        # ARIMA for Rare Watch with tuned orders
        self.apply_arima_model("Rare Watch", positionLimits, desiredPositions, p=4, d=1, q=1)
        
        # Scale positions to meet budget constraints
        desiredPositions = self.scale_positions(desiredPositions, currentPositions)
        return desiredPositions

    ########################################################
    # HELPER METHODS
    ########################################################
    
    def apply_regression_model(self, positionLimits, desiredPositions):
        # Calculate the regression signal for Fried Chicken based on current prices.
        FC = self.get_current_price("Fried Chicken")
        RC = self.get_current_price("Raw Chicken")
        SS = self.get_current_price("Secret Spices")
        
        # The following coefficients and threshold were tuned historically.
        reg_signal = -129.04616223805 * FC + 21.6126341844 * RC + SS
        threshold_value = -194.2325350693  # Fine-tune this threshold if needed.
        
        if reg_signal > threshold_value:
            desiredPositions["Fried Chicken"] = positionLimits["Fried Chicken"]
            desiredPositions["Secret Spices"] = -positionLimits["Secret Spices"]
            desiredPositions["Raw Chicken"] = -positionLimits["Raw Chicken"]
            self.Direction = -1
        else:
            desiredPositions["Fried Chicken"] = -positionLimits["Fried Chicken"]
            desiredPositions["Secret Spices"] = positionLimits["Secret Spices"]
            desiredPositions["Raw Chicken"] = positionLimits["Raw Chicken"]
            self.Direction = 1

    # ARIMA model for trend-based instruments
    def apply_arima_model(self, instrument, positionLimits, desiredPositions, p=1, d=1, q=1):
        if self.day >= self.lookback:
            data = np.array(self.data[instrument])
            model = ARIMA(data, order=(p, d, q))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)[0]
            current_price = self.get_current_price(instrument)
            price_diff = forecast - current_price
            if abs(price_diff) > self.threshold * current_price:
                position = positionLimits[instrument] if price_diff > 0 else -positionLimits[instrument]
                desiredPositions[instrument] = position

    def get_prplelixr_position(self, desiredPositions, positionLimits):
        # Purple Elixir strategy based on EMAs and a regression equation.
        elixr_df = pd.DataFrame(self.data["Purple Elixir"])
        quack_df = pd.DataFrame(self.data["Quantum Universal Algorithmic Currency Koin"])

        elixr_df['EMA'] = elixr_df[0].ewm(span=5, adjust=False).mean()
        quack_df['EMA'] = quack_df[0].ewm(span=5, adjust=False).mean()
        elixr_df['EMA25'] = elixr_df[0].ewm(span=25, adjust=False).mean()
        elixr_df['Cross'] = elixr_df['EMA'] - elixr_df['EMA25']

        price_drink = self.data['Purple Elixir'][-1]
        price_quack = self.data['Quantum Universal Algorithmic Currency Koin'][-1]
        ema_drink = elixr_df['EMA'].iloc[-1]
        ema_quack = quack_df['EMA'].iloc[-1]
        cross_signal = elixr_df['Cross'].iloc[-1]

        # The theoretical price is computed from tuned coefficients.
        theo = ema_drink - 0.025 * ema_quack + 0.055 * np.sign(cross_signal) * (abs(cross_signal)**(1/4))
        if price_drink > theo:
            desiredPositions["Purple Elixir"] = -positionLimits["Purple Elixir"]
        else:
            desiredPositions["Purple Elixir"] = positionLimits["Purple Elixir"]

    def get_goober_eats_position(self):
        goober_df = pd.DataFrame(self.data["Goober Eats"])
        goober_df['EMA'] = goober_df[0].ewm(span=13, adjust=False).mean()
        price = self.data['Goober Eats'][-1]
        ema = goober_df['EMA'].iloc[-1]
        limit = self.positionLimits["Goober Eats"]
        if price > ema:
            desiredPositions = -limit
        else:
            desiredPositions = limit
        return desiredPositions

    def get_uq_dollar_position(self, currentPosition, limit):
        avg = sum(self.data["UQ Dollar"][-4:]) / 4
        price = self.get_current_price("UQ Dollar")
        diff = avg - price
        # Fine-tune these delta multipliers if needed.
        if diff > 0.24:
            delta = limit * 2
        elif diff < -0.24:
            delta = -2 * limit
        else:
            delta = 0

        if currentPosition + delta > limit:
            desiredPosition = limit
        elif currentPosition + delta < -limit:
            desiredPosition = -limit
        else:
            desiredPosition = currentPosition + delta

        return desiredPosition

    def get_quack_position(self, currentPosition, limit):
        avg = sum(self.data["Quantum Universal Algorithmic Currency Koin"][-10:]) / 10
        price = self.get_current_price("Quantum Universal Algorithmic Currency Koin")
        if price < 2.2:
            desiredPosition = limit
        elif price > 2.45:
            desiredPosition = -limit
        else:
            desiredPosition = currentPosition

        return desiredPosition
    
    def get_dwgFood_position(self):
        dwgFood_df = pd.DataFrame(self.data["Dawg Food"])
        dwgFood_df['EMA5'] = dwgFood_df[0].ewm(span=2, adjust=False).mean()
        price = self.data['Dawg Food'][-1]
        ema = dwgFood_df['EMA5'].iloc[-1]
        if price > ema:
            desiredPosition = -self.positionLimits["Dawg Food"]
        else:
            desiredPosition = self.positionLimits["Dawg Food"]
        return desiredPosition

    def get_token_position(self, currentPosition, limit):
        step = 35
        if self.day < 10:
            return currentPosition

        first_half = self.data["Fintech Token"][-10:-5]
        second_half = self.data["Fintech Token"][-5:]
        first_grad = self.calculate_gradient(first_half)
        second_grad = self.calculate_gradient(second_half)
        lim = 18

        if abs(first_grad) < lim and second_grad > lim:
            delta = step
        elif abs(first_grad) < lim and second_grad < -lim:
            delta = -step
        else:
            delta = 0

        if currentPosition + delta > limit:
            desiredPosition = limit
        elif currentPosition + delta < -limit:
            desiredPosition = -limit
        else:
            desiredPosition = currentPosition + delta

        return desiredPosition

    ########################################################
    # POSITION SCALING AND BUDGET ADJUSTMENT
    ########################################################

    def scale_positions(self, desiredPositions, currentPositions):
        total_pos_value, prices_current, pos_values = self.calc_current_total_trade_val(desiredPositions, currentPositions)
        if total_pos_value > self.totalDailyBudget:
            reduction_val = total_pos_value - self.totalDailyBudget
            reduction_Tokens = int(reduction_val / prices_current["Rare Watch"])
            if pos_values["Rare Watch"] > 0:
                desiredPositions["Rare Watch"] -= min(reduction_Tokens, desiredPositions["Rare Watch"])
            else:
                desiredPositions["Rare Watch"] += min(reduction_Tokens, desiredPositions["Rare Watch"])
            total_pos_value, prices_current, pos_values = self.calc_current_total_trade_val(desiredPositions, currentPositions)
            if total_pos_value <= self.totalDailyBudget:
                return desiredPositions
            for inst in ['Fintech Token', 'Quantum Universal Algorithmic Currency Koin', 'UQ Dollar', 
                         'Raw Chicken', 'Secret Spices', 'Fried Chicken', 'Goober Eats', 'Purple Elixir', 'Dawg Food']:
                reduction_val = total_pos_value - self.totalDailyBudget
                reduction_inst = int(reduction_val / prices_current[inst]) + 1
                if pos_values[inst] > 0:
                    desiredPositions[inst] -= min(reduction_inst, desiredPositions[inst])
                else:
                    desiredPositions[inst] += min(reduction_inst, -desiredPositions[inst])
                total_pos_value, prices_current, pos_values = self.calc_current_total_trade_val(desiredPositions, currentPositions)
                if total_pos_value <= self.totalDailyBudget:
                    return desiredPositions
        return desiredPositions
                
    def adjust_positions_for_budget(self, desiredPositions, positionLimits):
        prices = {inst: self.get_current_price(inst) for inst in desiredPositions}
        total_value = sum(abs(desiredPositions[inst] * prices[inst]) for inst in desiredPositions)
        if total_value > self.totalDailyBudget:
            scaling_factor = self.totalDailyBudget / total_value
            for inst in desiredPositions:
                desiredPositions[inst] = int(desiredPositions[inst] * scaling_factor)
            total_value = sum(abs(desiredPositions[inst] * prices[inst]) for inst in desiredPositions)
        if total_value < self.totalDailyBudget and "Secret Spices" in prices:
            extra_budget = self.totalDailyBudget - total_value
            additional_units = int(extra_budget / prices["Secret Spices"])
            current_direction = getattr(self, 'Direction', 1) or 1
            new_position = desiredPositions.get("Secret Spices", 0) + current_direction * additional_units
            limit = positionLimits.get("Secret Spices", new_position)
            desiredPositions["Secret Spices"] = max(-limit, min(limit, new_position))
            total_value = sum(abs(desiredPositions[inst] * prices[inst]) for inst in desiredPositions)
        maxInventoryValue = 599997
        if total_value > maxInventoryValue:
            excess_value = total_value - maxInventoryValue
            instruments = ['Fintech Token', 'Quantum Universal Algorithmic Currency Koin', 'UQ Dollar', 
                           'Raw Chicken', 'Secret Spices', 'Fried Chicken', 'Goober Eats', 'Purple Elixir', 'Dawg Food']
            for instrument in instruments:
                if instrument in desiredPositions and desiredPositions[instrument] != 0:
                    instrument_value = abs(desiredPositions[instrument] * prices[instrument])
                    if instrument_value <= excess_value:
                        excess_value -= instrument_value
                        desiredPositions[instrument] = 0
                    else:
                        reduction_units = int(excess_value / prices[instrument])
                        desiredPositions[instrument] -= int(np.sign(desiredPositions[instrument]) * reduction_units)
                        excess_value = 0
                        break
        for inst in desiredPositions:
            desiredPositions[inst] = int(desiredPositions.get(inst, 0))
        return desiredPositions

    def calc_current_total_trade_val(self, desiredPositions, currentPositions):
        prices_current = {inst: self.get_current_price(inst) for inst in desiredPositions}
        pos_values = {inst: abs(desiredPositions[inst] * prices_current[inst]) for inst in desiredPositions}
        total_pos_value = sum(pos_values.values())
        return total_pos_value, prices_current, pos_values

    ########################################################
    # SUPPORTIVE METHODS
    ########################################################

    def linear_extrapolation(self, values):
        if len(values) < 5:
            return np.nan 
        x = np.arange(5)
        y = values[-6:-1]
        coeffs = np.polyfit(x, y, 1)  # Linear fit (degree 1)
        extrapolated_value = np.polyval(coeffs, 5)  # Extrapolate to the next point
        return extrapolated_value
    
    def calculate_gradient(self, values):
        x = np.arange(5) 
        y = np.array(values)
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return m

    

#     Total PNL ($): 2591295.18
# ##################################################
# Fintech Token Returns ($): 60813.72
# Purple Elixir Returns ($): 896500.00
# Quantum Universal Algorithmic Currency Koin Returns ($): 96400.00
# Dawg Food Returns ($): 344632.00
# UQ Dollar Returns ($): 100646.00
# Fried Chicken Returns ($): 226200.00
# Secret Spices Returns ($): 88186.00
# Goober Eats Returns ($): 734250.00
# Raw Chicken Returns ($): 19050.00
# Rare Watch Returns ($): 24617.46
##################################################



## v2
# Total PNL ($): 2597644.85
# ##################################################
# Fintech Token Returns ($): 63405.03
# Purple Elixir Returns ($): 896500.00
# Quantum Universal Algorithmic Currency Koin Returns ($): 96400.00
# Dawg Food Returns ($): 344632.00
# UQ Dollar Returns ($): 100646.00
# Fried Chicken Returns ($): 226200.00
# Secret Spices Returns ($): 88074.00
# Goober Eats Returns ($): 734250.00
# Raw Chicken Returns ($): 18150.00
# Rare Watch Returns ($): 29387.82
# ##################################################