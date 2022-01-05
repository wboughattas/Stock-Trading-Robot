import os

import finnhub

from util.config import get_env

if __name__ == '__main__':
    get_env()
    print(finnhub.Client(api_key=os.getenv('finnhub_token')).crypto_symbols('KUCOIN'))
