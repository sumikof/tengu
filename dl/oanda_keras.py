from keras.callbacks import EarlyStopping

from dl import util

class oanda_keras:
    def __init__(self,length_of_sequences = 100,in_out_neurons = 1,hidden_neurons = 300,batch_size=600, epochs=5, validation_split=0.05):
        self.length_of_sequences = length_of_sequences
        self.in_out_neurons = in_out_neurons
        self.hidden_neurons = hidden_neurons
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split

    def odl(self,X_train,y_train,X_test,y_test):
        from keras.models import Sequential
        from keras.layers import Dense, Activation
        from keras.layers.recurrent import LSTM


        model = Sequential()

        model.add(LSTM(self.hidden_neurons, batch_input_shape=(None, self.length_of_sequences, self.in_out_neurons), return_sequences=False))
        model.add(Dense(self.in_out_neurons))
        model.add(Activation('linear'))

        model.compile(loss="mean_squared_error", optimizer="adam")

        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        history = model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, validation_split=self.validation_split, callbacks=[early_stopping])

        predicted = model.predict(X_test)
        return X_test,y_test,predicted,history


    def oadna_make_batchdata(self,df):
        """
            ans = df.drop(0, axis=0)[['high']].values
            np = df.drop(df.tail(1).index)[['high']].values

            print(np)
            print(ans)
        """
        (X_train, y_train), (X_test, y_test) = util.train_test_split(df,test_size=0.1, n_prev=self.length_of_sequences)
        return self.odl(X_train,y_train,X_test,y_test)