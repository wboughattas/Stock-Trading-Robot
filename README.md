# Stock Trading Robot

## Run Locally

Clone the project
```bash
  git clone https://github.com/wboughattas/Stock-Trading-Robot.git
```

Go to the project directory
```bash
  cd .../Stock-Trading-Robot
```

Create a venv and install dependencies
```bash
  pip install -r requirements.txt
```

Install [InfluxDB2.0](https://docs.influxdata.com/influxdb/v2.0/install/) and start influxDB2.0 local server (Windows)
```bash
  influxd
```

Add the following environment variables to your .env file in your venv folder:
 
```python
# to connect to Finnhub websocket api
finnhub_token = '<API-token>'

# to connect to an already-set-up local SQL server
influxdb_token = '<InfluxDB-bucket-token>'
org = '<organization-name>'
```

## Git workflow
"You can see the main version as the public release like app 1.0.
DEV is the branch with the version of the next update,
so when you merge DEV into main, it becomes app 1.1.
Also, to be even safer before merging a feature into dev, 
you should merge dev into your feature branch, 
resolve the conflicts, and then merge your feature into dev.
This keeps the dev version clean"
-Charley

- main is untouched until DEV is pushed to main
- To work on a feature (e.g. feature/establish-fill-trade-db), create a branch within DEV called: feature/establish-fill-trade-db. Once your feature is fully working and ready to be pushed to DEV, create a pull request for anyone else to verify your code
