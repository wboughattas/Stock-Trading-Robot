# Stock Trading Robot
Stock trading software that automates trading decisions.
Get live Github flags from [shields io](https://shields.io/)

## Table of Contents  
<!--ts-->
- [Run Locally](#Run-Locally)
- [Skills Summary](#Skills-Summary)
  - [Contributors](#Contributors)
  - [Skills Required](#Skills-Required)
- [Roadmap](#Roadmap)
  - [Git workflow](#Git-workflow)
    - [DEV](#DEV)
      - [feature/establish-fill-trade-db](#featureestablish-fill-trade-db)
      - [feature/integrate-gradle](#featureintegrate-gradle)
      - [feature/bloomberg-api](#featurebloomberg-api)
      - [feature/nltk](#featurenltk)
      - [feature/ML-predict](#featureML-predict)
      - [feature/ML-decision-making](#featureML-decision-making)
      - [feature/alpaca-api-buy-sell](#featurealpaca-api-buy-sell)
      - [feature/system-hardware](#featuresystem-hardware)
      - [feature/ui](#featureui)
    - [TEST](#TEST)
    - [PROD](#PROD)
<!--te-->
https://github.com/wboughattas/Stock-Trading-Robot#run-locally

## Run Locally

Clone the project
```bash
  git clone https://github.com/wboughattas/Stock-Trading-Robot.git
```

Go to the project directory
```bash
  cd .../Stock-Trading-Robot
```

Install dependencies
```bash
  pip install -r requirements.txt
```

Install [InfluxDB2.0](https://docs.influxdata.com/influxdb/v2.0/install/) and start influxDB2.0 local server
```bash
  influxd
```

Add the following environment variables to your .env file in your venv folder:
 
```python
# to connect to Finnhub websocket api
socket = 'wss://ws.finnhub.io?token=<API-token>'

# to connect to an already-set-up local SQL server
token = '<InfluxDB-bucket-token>'
org = '<organization-name>'
bucket = '<dataset-name>'
```

## Skills Summary
### Contributors

| Names                                                                      | Skills (in-scope of this project)         |Hyperlink      |
| -----------------------                                                    |:-------------:                            |:-------------:|
| [@Alessandro Morsella](https://github.com/Alessmorsella)                   |                                           |               |
| [@Charles-Etienne Désormeaux](https://github.com/CharlesEtienneDesormeaux) |                                           |               |
| [@Humam Hawara](https://www.github.com/Humamhwr)                           |                                           |               |
| [@Wasim Boughattas](https://github.com/wboughattas)                        |SQL-ML-Pandas/torch                        |X              |

### Skills Required
- SQL: SQL integration with Python/Java and automated DML/DLL
- Pandas/torch: Quick and efficient data manipulation to set up ML models
- ML: Predict future prices
- NLTK: Natural language processing
- AI: Automate buy/sell decision-making
- MAS (Multi-agent systems): multi-agent system consisting of multiple decision-making agents
- System Hardware: Backup (RAID), arduino, Fail system, automated system check
- UI: Simple report on earnings/losses with more indicators 

## Roadmap
### Git workflow
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

* [ ] The code in the main branch needs to be edited to incorporate all the above
* [ ] Edit README's structure and add new features

### DEV
#### feature/establish-fill-trade-db

Each stock has 3 datasets (sorted by time desc) with data from 2010: 
1. live/historical stock Trades
2. live/historical intraday indicators
3. live/historical order book.

To create and populate the stock trades dataset:
* [x] Connect python to a finnhub's websocketAPI and print every trade
* [x] parse live trades into intraday stock indicators (Open, close, low, high)
* [ ] Ingest stock indicators to local InfluxDB server recurrently
* [ ] Lookup for historical stock trades 
* [ ] (maybe) replace finnhub with the exchange api if historical stock trades exist

To create and populate the stock indicators:
* [ ] Connect python to an alpha-vintage websocketAPI to get historical data (stock indicators)
* [ ] Ingest the missing stock indicators values from 2010 to sqlDB (test for missing values every 5-minutes for each stock)

To create and populate the order book dataset:
* [x] Connect python to a crypto EXCHANGE's websocketAPI
* [ ] Find the time until the trade is executed for each trade 
* [ ] Ingest order book trades with the time attribute to local InfluxDB server recurrently
* [ ] Lookup for historical order book trades 

#### feature/integrate-gradle
* [ ] replace pip with gradle
* [ ] update gradle with already existing requirements
* [ ] .

#### feature/bloomberg-api
* [ ] give the bloomberg api (test version => free) a look
* [ ] integrate the bloomberg api in java
* [ ] update gradle
* [ ] .

#### feature/nltk
* [ ] set up connection with the server and add columns with random numbers as default (for testing only) to sqlDB
* [ ] .
* [ ] .
* [ ] .

#### feature/ML-predict
* [ ] set up connection with the server and add columns with random numbers as default (for testing only) to sqlDB
* [ ] .
* [ ] .
* [ ] .

#### feature/ML-decision-making
* [ ] set up connection with the server and set-up random decision-making strategy (for testing only)
* [ ] .
* [ ] .
* [ ] .

#### feature/alpaca-api-buy-sell
* [ ] set up a test env and use the outputs from feature/ML-decision-making to buy/sell stocks
* [ ] .
* [ ] .
* [ ] .

#### feature/system-hardware
* [ ] Backup (RAID)
* [ ] arduino server
* [ ] Fail system
* [ ] automated system check
* [ ] .

#### feature/ui
* [ ] Simple report on earnings/losses with more indicators
* [ ] .
* [ ] .
* [ ] .

### TEST

### PROD
