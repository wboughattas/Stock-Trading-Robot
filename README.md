# Stock Trading Robot
The app extracts live trade data of selected stocks from [Finnhub's](https://finnhub.io/docs/api/) databases using their 
free websocket api. A per-minute ohlcv quote, corresponding to each unique stock, is instantly calculated. 
The trade and quote data are instantly being ingested into local Influxdb server. 

The project can be scaled to incorporate historical data. The goal is to use both traditional quantitative finance 
methodologies and machine learning algorithms to predict stock movements. 

## Run Locally

Clone the project
```bash
  git clone https://github.com/wboughattas/Stock-Trading-Robot.git
```

Go to the project directory
```bash
  cd Stock-Trading-Robot
```

Create and activate a virtual environment and execute the following command to install dependencies
```bash
  pip install -r requirements.txt
```

Install [InfluxDB2.0](https://docs.influxdata.com/influxdb/v2.0/install/) and execute the following command to start 
influxDB2.0 local server (Windows)
```bash
  influxd
```

Add the following environment variables to your .env file in your venv folder
```python
# to connect to Finnhub websocket api
finnhub_token = '<API-token>'

# to connect to an already-set-up local SQL server
influxdb_token = '<InfluxDB-bucket-token>'
org = '<organization-name>'
```

Select the appropriate stock symbols in 
[conf/livedata.py](https://github.com/wboughattas/Stock-Trading-Robot/blob/main/conf/livedata.py). 
Your selections must exist in [Finnhub's](https://finnhub.io/docs/api/) databases
```python
finnhub = {
    'BINANCE': ['BTCUSDT', 'ETHUSDT'],
    'COINBASE': ['ETH-EUR', 'DOGE-USDT']
}
```

Execute the following command to run [get_livedata.py](https://github.com/wboughattas/Stock-Trading-Robot/blob/main/get_livedata.py) to start ingesting livedata to local Influx database
```bash
venv/Scripts/python.exe get_livedata.py
```

## Git workflow
You can see the main version as the public release like app 1.0.
DEV is the branch with the version of the next update,
so when you merge DEV into main, it becomes app 1.1.
Also, to be even safer before merging a feature into DEV, 
you should merge dev into your feature branch, 
resolve the conflicts, and then merge your feature into DEV.
This keeps the DEV version clean.

- main is untouched until DEV is pushed to main
- To work on a feature (e.g. feature/establish-fill-trade-db), create a branch within DEV called: feature/establish-fill-trade-db. Once your feature is fully working and ready to be pushed to DEV, create a pull request for anyone else to verify your code
