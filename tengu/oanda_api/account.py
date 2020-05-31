class Account:
    id = ""
    token = ""

def create_account(config_file):
    import configparser
    config = configparser.ConfigParser()
    config.read(config_file)

    from tengu import oanda_api
    account = oanda_api.Account()
    account.id = config['OANDA']['accountID']
    account.token = config['OANDA']['access_token']

    return account