from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from bot import get_data
import time

start = time.perf_counter()

syms = ['NVDA']
for sym in syms:
    data = get_data(sym)
    signals  = ['SMAsignal', 'MACDsignal', 'EMAsignal', 'RSIsignal', 'BBsignal', 'Return']
    features = ['SMA_Fast', 'SMA_Slow', 'EMA', 'UpperBand', 'LowerBand']
    scalar = StandardScaler()
    X = data[signals]
    X_scaled = scalar.fit_transform(X)
    y = data['Target']
    gmm = GaussianMixture(n_components=2, random_state=50)
    data['Cluster'] = gmm.fit_predict(X_scaled)


    plt.scatter(data['Return'], data['Close'], c=data['Cluster'], cmap='viridis', marker='o')   
    plt.title('GMM Clustering of Technical Indicators')
    plt.xlabel('Return')
    plt.ylabel('Close')
    plt.show()
    print(data['Cluster'].value_counts())
    print(data['Target'].value_counts())

    data['Signal'] = 0
    data.loc[data['Cluster'] == 0, 'Signal'] = 1
    data.loc[data['Cluster'] == 1, 'Signal'] = 0


    data['sTrue'] = data['Signal'] == data['Target']
    data['Entry'] = data.Signal.diff()
    print(data['sTrue'].value_counts())
    print(data['Entry'].value_counts())
    data.index = pd.to_datetime(data.index)
    print(data)


    # Filter for the last 3 months of data
    three_months_ago = data.index.max() - pd.DateOffset(months=3)
    recent_data = data[data.index >= three_months_ago]

    # Plot the Return Price per Day for the last 3 months
    plt.figure(figsize=(14, 7))
    plt.plot(recent_data.index, recent_data['Close'], label=sym, color='blue')


    # Plotting arrows for Entry signals
    for i in range(1, len(data)):
        if data['Entry'].iloc[i] == 1:
            plt.annotate('', xy=(data.index[i], data['Close'].iloc[i]), xytext=(data.index[i], data['Close'].iloc[i] - 0.02),
                        arrowprops=dict(facecolor='green', shrink=0.05, headwidth=10, headlength=15, width=2))
        elif data['Entry'].iloc[i] == -1:
            plt.annotate('', xy=(data.index[i], data['Close'].iloc[i]), xytext=(data.index[i], data['Close'].iloc[i] + 0.02),
                        arrowprops=dict(facecolor='red', shrink=0.05, headwidth=10, headlength=15, width=2))


    plt.title('Return Price per Day with Entry Signals')
    plt.xlabel('Date')
    plt.ylabel('Close Price')  
    plt.legend()
    plt.grid(True)
    plt.show()

X = data[signals]
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

rf = RandomForestClassifier(random_state=50)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(X_test)

print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

end = time.perf_counter()
total = end - start
print(f'Total time: {total}')

