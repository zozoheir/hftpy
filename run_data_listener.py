import argparse

from cryptofeed import FeedHandler
from cryptofeed.defines import *
from cryptofeed.exchanges import EXCHANGE_MAP

from hftpy.quant.custom_order_book import CustomGateIoOrderBook
from hftpy.quant.online_transforms import ExponentialMovingAverage


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--exchange',
                    type=str,
                    required=True)
parser.add_argument('-s', '--symbols',
                    nargs='+',
                    required=True,
                    help="Names or Number of influencers")
parser.add_argument('-ki', '--key_id',
                    type=str,
                    required=True)
parser.add_argument('-ks', '--key_secret',
                    type=str,
                    required=True)

args = parser.parse_args()

EXCHANGE = args.exchange
KEY_ID = args.key_id
KEY_SECRET = args.key_secret

assert EXCHANGE in EXCHANGE_MAP.keys(), "Exchange not supported, check exchange argument"

SYMBOLS = args.symbols

# TODO Get alpha, transforms and standardization params from config file
# TODO Create FeatureTransforms class to pool database requests and avoid multiple connections
# TODO Create Standardizer class for online Z score standardization of features

CUSTOM_OB_MAP = {
    'GATEIO': CustomGateIoOrderBook
}

live_order_books = {
    sym: CUSTOM_OB_MAP[EXCHANGE]() for sym in EXCHANGE_MAP[EXCHANGE].symbols()
}

live_mid_emas = {
    sym: ExponentialMovingAverage(input_feature_name='mid',
                                  alpha=0.5,
                                  init_value=None,
                                  # TODO Initialize value from latest value stored in DB, else use warmup
                                  required_n_warmups=100) for sym in SYMBOLS
}


async def order_book_callback(data,
                              receipt_timestamp):
    ob_dict = data.to_dict()
    ob_dict['receipt_timestamp'] = receipt_timestamp
    live_order_books[data.symbol].update(ob_dict)
    live_mid_emas[data.symbol].update(live_order_books[data.symbol].mid)

    print("--------------")
    print(f"receipt timestamp: {receipt_timestamp} ")
    print(f"symbol | {data.symbol} ")
    print(f"ob weighted mid | {live_order_books[data.symbol].getTOBWMID()} ")
    print(f"mid | {live_order_books[data.symbol].mid}")
    print(f"ema | {live_mid_emas[data.symbol].value}")
    print(f"updates_since_ema_init | {live_mid_emas[data.symbol].updates_since_init}")


def main():
    f = FeedHandler(
        config="cryptofeed_config.yaml"
    )
    f.add_feed(EXCHANGE_MAP[EXCHANGE](symbols=SYMBOLS,
                                      channels=[L2_BOOK],
                                      callbacks={L2_BOOK: order_book_callback}))
    f.run()


if __name__ == '__main__':
    main()
