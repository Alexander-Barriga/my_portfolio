import pandas as pd  
import matplotlib.pyplot as plt      

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

EDA_PLOT_DIR = '/Users/alexanderbarriga/EV_project/plots/EDA/'

class RegressionTrainer:
    def __init__(self, df, model):
        self.df = df
        self.model = model
        self.model_name = model.__str__().strip("()")
    
    def train(self):
        x_feats = ['AccelSec', 'TopSpeed_KmH', 'Range_Km', 'Efficiency_WhKm', 'FastCharge_KmH']
        X = self.df[x_feats]
        y = self.df['PriceEuro']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("Mean Squared Error:", mse)
        print("Mean Absolute Error:", mae)
        print("R-squared:", r2)

        self.plot_predictions(y_test, y_pred)
        self.plot_residuals(y_test, y_pred)

        if "Forest" in self.model_name:
            self.plot_feature_importance(x_feats)
        else:
            self.plot_coefficients(X.columns)

    def plot_predictions(self, y_test, y_pred):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, color='blue')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')
        plt.xlabel('Actual Price (Euro)')
        plt.ylabel('Predicted Price (Euro)')
        plt.title('Predicted vs. Actual Prices')
        plt.grid(True)
        plt.savefig(EDA_PLOT_DIR + 'Predicted vs. Actual Prices_{}.png'.format(self.model_name))
        plt.show()

    def plot_residuals(self, y_test, y_pred):
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, residuals, color='green')
        plt.xlabel('Actual Price (Euro)')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.axhline(y=0, color='red', linestyle='--')
        plt.grid(True)
        plt.savefig(EDA_PLOT_DIR + 'Residuals_Plot_{}.png'.format(self.model_name))
        plt.show()

    def plot_feature_importance(self, feature_names):
        importances = self.model.feature_importances_
        indices = list(range(len(importances)))

        indices_sorted = sorted(indices, key=lambda i: importances[i], reverse=True)

        feature_names_sorted = [feature_names[i] for i in indices_sorted]

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices_sorted], align='center')
        plt.xticks(range(len(importances)), feature_names_sorted, rotation=45, ha='right')
        plt.xlabel('Feature')
        plt.ylabel('Feature Importance')
        plt.title('Feature Importance Plot')
        plt.tight_layout()
        plt.savefig(EDA_PLOT_DIR + 'Feature Importance Plot_{}.png'.format(self.model_name))
        plt.show()

    def plot_coefficients(self, feature_names):
        coefficients = pd.Series(self.model.coef_, index=feature_names)
        coefficients.plot(kind='bar')
        plt.xlabel('Features')
        plt.ylabel('Coefficient Value')
        plt.title('Feature Importance Plot')
        plt.savefig(EDA_PLOT_DIR + 'Feature Importance Plot_{}.png'.format(self.model_name))
        plt.show()

