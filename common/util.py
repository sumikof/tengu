def oanda_dataframe(file_name):
    import pandas as pd
    df = pd.read_csv(file_name)
    df['date'] = pd.to_datetime(df['date'])
    #df = df[df['date'].dt.minute % 10 == 0]

    return df
