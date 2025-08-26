# Ex.No: 01A PLOT A TIME SERIES DATA
###  Date: 19-08-2025
## Name: K KESAVA SAI
##Register Number: 212223230105
# AIM:
To Develop a python program to Plot a time series data (population/ market price of a commodity
/temperature.
# ALGORITHM:
1. Import the required packages like pandas and matplot
2. Read the dataset using the pandas
3. Calculate the mean for the respective column.
4. Plot the data according to need and can be altered monthly, or yearly.
5. Display the graph.
# PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load dataset
data = pd.read_csv("Clean_Dataset.csv")

# Choose period (weekly-like since we only have 49 days_left values)
period = 7  

# Get unique airlines
airlines = data["airline"].unique()

for airline in airlines:
    # Filter data per airline
    df_airline = data[data["airline"] == airline]
    
    # Group by days_left (average price for each day before departure)
    data_grouped = df_airline.groupby("days_left")["price"].mean().reset_index()
    data_grouped = data_grouped.sort_values("days_left")
    data_grouped.set_index("days_left", inplace=True)

    # Create transformations
    data_grouped["price_diff"] = data_grouped["price"].diff()

    # Seasonal decomposition on raw prices
    if len(data_grouped) >= 2 * period:  # check enough data
        result = seasonal_decompose(data_grouped["price"], model="additive", period=period)
        data_grouped["price_sea_diff"] = result.resid
    else:
        data_grouped["price_sea_diff"] = np.nan

    # Log transform and differencing
    data_grouped["price_log"] = np.log(data_grouped["price"])
    data_grouped["price_log_diff"] = data_grouped["price_log"].diff()

    if len(data_grouped.dropna()) >= 2 * period:
        result_log = seasonal_decompose(data_grouped["price_log_diff"].dropna(), model="additive", period=period)
        resid_log = result_log.resid
        resid_log.index = data_grouped["price_log_diff"].dropna().index
        data_grouped["price_log_sea_diff"] = resid_log
    else:
        data_grouped["price_log_sea_diff"] = np.nan

    # Plot for this airline
    plt.figure(figsize=(16, 18))
    plt.suptitle(f"Airline: {airline}", fontsize=18)

    plt.subplot(6, 1, 1)
    plt.plot(data_grouped["price"], label="Original Price")
    plt.legend(loc="best")
    plt.title("Average Price vs Days Left")
    plt.xlabel("Days Left")
    plt.ylabel("Price")

    plt.subplot(6, 1, 2)
    plt.plot(data_grouped["price_diff"], label="Regular Difference")
    plt.legend(loc="best")
    plt.title("Regular Differencing")
    plt.xlabel("Days Left")
    plt.ylabel("Diff(Price)")

    plt.subplot(6, 1, 3)
    plt.plot(data_grouped["price_sea_diff"], label="Seasonal Adjustment")
    plt.legend(loc="best")
    plt.title("Seasonal Adjustment")
    plt.xlabel("Days Left")
    plt.ylabel("Resid")

    plt.subplot(6, 1, 4)
    plt.plot(data_grouped["price_log"], label="Log Transformation")
    plt.legend(loc="best")
    plt.title("Log Transformation")
    plt.xlabel("Days Left")
    plt.ylabel("Log(Price)")

    plt.subplot(6, 1, 5)
    plt.plot(data_grouped["price_log_diff"], label="Log Differencing")
    plt.legend(loc="best")
    plt.title("Log Transformation + Differencing")
    plt.xlabel("Days Left")
    plt.ylabel("Diff(Log(Price))")

    plt.subplot(6, 1, 6)
    plt.plot(data_grouped["price_log_sea_diff"], label="Log + Diff + Seasonal Adjustment")
    plt.legend(loc="best")
    plt.title("Log Transformation + Differencing + Seasonal Adjustment")
    plt.xlabel("Days Left")
    plt.ylabel("Resid")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

```


# OUTPUT:
<img width="1816" height="718" alt="image" src="https://github.com/user-attachments/assets/74b02b69-1dcd-4cbd-8be7-8f8ccf709eec" />
<img width="1835" height="560" alt="image" src="https://github.com/user-attachments/assets/6b2c9da1-ef2e-45a9-bd3a-884ae28a509c" />
<img width="1856" height="555" alt="image" src="https://github.com/user-attachments/assets/1c0a53b2-58f2-4b73-ab2e-ddd1c0d1204e" />





# RESULT:
Thus we have created the python code for plotting the time series of given data.
