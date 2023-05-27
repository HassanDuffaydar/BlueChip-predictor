import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tkinter as tk


def scrape_stock_data(symbol):
    url = f"https://finance.yahoo.com/quote/{symbol}/history"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    table = soup.find("table")
    rows = table.tbody.find_all("tr")

    data = []
    for row in rows:
        cols = row.find_all("td")
        if len(cols) == 7:  # Exclude dividend rows
            date = cols[0].text
            close_price = cols[4].text.replace(",", "")
            data.append([date, float(close_price)])

    return data


def preprocess_data(data):
    df = pd.DataFrame(data, columns=["Date", "Close Price"])
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(by="Date", inplace=True)
    df.set_index("Date", inplace=True)

    return df


def train_model(df):
    X = df.index.values.astype("int64").reshape(-1, 1)
    y = df["Close Price"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
    print(f"Train RMSE: {rmse_train:.2f}")

    y_pred_test = model.predict(X_test)
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
    print(f"Test RMSE: {rmse_test:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, y, label="Actual")
    plt.plot(df.index[:-len(X_test)], y_pred_train, label="Train Predictions")
    plt.plot(df.index[-len(X_test):], y_pred_test, label="Test Predictions")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title("Predicted vs. Actual Stock Prices")
    plt.legend()
    plt.show()

    return model


def predict(model, future_dates):
    future_dates = future_dates.astype("int64").reshape(-1, 1)
    predictions = model.predict(future_dates)
    return predictions


def handle_prediction():
    symbol = entry.get()

    data = scrape_stock_data(symbol)
    df = preprocess_data(data)
    model = train_model(df)

    future_dates = pd.date_range(start=df.index[-1], periods=5, freq="D")
    predictions = predict(model, future_dates)

    result_text.delete(1.0, tk.END)
    for date, prediction in zip(future_dates, predictions):
        result_text.insert(tk.END, f"Date: {date.date()}, Predicted Close Price: {prediction:.2f}\n")


window = tk.Tk()
window.title("Stock Prediction")

label = tk.Label(window, text="Enter Stock Symbol:")
label.pack()

entry = tk.Entry(window)
entry.pack()

button = tk.Button(window, text="Predict", command=handle_prediction)
button.pack()

result_text = tk.Text(window, height=10, width=50)
result_text.pack()

window.mainloop()
