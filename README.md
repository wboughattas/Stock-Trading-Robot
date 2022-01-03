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
      - [feature](#feature)
        - [establish-fill-trade-db](#establish-fill-trade-db)
        - [collect-system-stat](#collect-system-stat)
        - [integrate-gradle](#integrate-gradle)
        - [logging](#logging)
        - [bloomberg-api](#bloomberg-api)
        - [nltk](#nltk)
        - [ml-predict](#ml-predict)
        - [ml-decision-making](#ml-decision-making)
        - [multi-agent-systems](#multi-agent-systems)
        - [api-buy-sell](#api-buy-sell)
        - [system-hardware](#system-hardware)
        - [ui](#ui)
    - [TEST](#TEST)
    - [PROD](#PROD)
<!--te-->

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
| [@Wasim Boughattas](https://github.com/wboughattas)                        |SQL/Flux-ML-Pandas/torch                   |X              |

### Skills Required
- SQL/Flux: InfluxDB integration with Python/Java and automated DML/DLL
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
#### feature
##### establish-fill-trade-db

Each stock has 3 datasets (sorted by time desc) with data from 2010: 
1. live/historical stock Trades

| time | _stop |
|:------:|:-----:|
| 2016-06-13T17:43:50.1004002Z       |       |

2. live/historical intraday indicators

| _start | _stop | _time | _measurement | _field | _value | _time | _measurement | 
|:------:|:-----:|:-----:|:------------:|:------:|:------:|:-----:|:------------:|
|        |       |       |              |        |        |       |              |

3. live/historical order book

| _start | _stop | _time | _measurement | _field | _value | _time | _measurement | 
|:------:|:-----:|:-----:|:------------:|:------:|:------:|:-----:|:------------:|
|        |       |       |              |        |        |       |              |

Querying from tables outputs a table or a stream of tables with this structure:

| _start | _stop | _time | _measurement | _field | _value | _time | _measurement | 
|:------:|:-----:|:-----:|:------------:|:------:|:------:|:-----:|:------------:|
|        |       |       |              |        |        |       |              |

* _start: Query range start time (defined by range())
* _stop: Query range stop time (defined by range())
* _time: Data timestamp
* _measurement: Measurement name
* _field: Field key
* _value: Field value
* Tag columns: A column for each tag where the column label is the tag key and the column value is the tag value


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

##### collect-system-stat
* [ ] Prometheus/Telegraf Integrations into DB
* [ ] "monitoring"/"tasks" observation and integration into DB
* [ ] .

##### integrate-gradle
* [ ] replace pip with gradle
* [ ] update gradle with already existing requirements
* [ ] .

##### logging
* [ ] Add a logging feature (efficient/useful/not memory-heavy) 
* [ ] https://realpython.com/python-logging/
* [ ] .

##### bloomberg-api
* [ ] give the bloomberg api (test version => free) a look
* [ ] integrate the bloomberg api in java
* [ ] update gradle
* [ ] .

##### nltk
* [ ] set up connection with the server and add columns with random numbers as default (for testing only) to sqlDB
* [ ] .
* [ ] .
* [ ] .

##### ml-predict
* [ ] set up connection with the server and add columns with random numbers as default (for testing only) to sqlDB
* [ ] .
* [ ] .
* [ ] .

##### ml-decision-making
* [ ] set up connection with the server and set-up random decision-making strategy (for testing only)
* [ ] .
* [ ] .
* [ ] .

##### multi-agent-systems
* [ ] .
* [ ] .
* [ ] .
* [ ] .

##### api-buy-sell
* [ ] set up a test env and use the outputs from feature/ML-decision-making to buy/sell stocks
* [ ] .
* [ ] .
* [ ] .

##### system-hardware
* [ ] Backup (RAID)
* [ ] arduino server
* [ ] Fail system
* [ ] automated system check
* [ ] .

##### ui
* [ ] Simple report on earnings/losses with more indicators
* [ ] .
* [ ] .
* [ ] .

### TEST

### PROD
