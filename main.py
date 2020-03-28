
from oanda_api import oanda_rest_api


class oanda_rest_json:
    def __init__(self):
        self.a = []

    def printdump(self,rate):
        import json
        print(json.dumps(rate, indent=2))

def rest_action(account):
    action = oanda_rest_json()
    oanda_rest_api(account, instrument="USD_JPY", action=action.printdump, count=100, granularity="M1")

def get_account():
    from oanda_api.account import create_account
    account = create_account('oanda.conf')
    return account

def create_ex_csv(account):
    from oanda_action.create_csv import create_csv
    create_csv(account,instrument="USD_JPY",from_date="2020-01-01T00:00:00")

def main():
    rest_action(get_account())
    create_ex_csv(get_account())




if __name__ == '__main__':
    main()
