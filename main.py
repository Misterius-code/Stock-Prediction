import matplotlib
import numpy
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout
from flask import Flask, render_template, request


def preprocessking(my_data):

    X = my_data[['Open', 'High', 'Low', 'Close']].tail(-1)
    Y = my_data[['Open', 'High', 'Low', 'Close']].head(-1)
    X = X.tail(60)
    Y = Y.tail(60)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    #test_Date= new_data.reset_index()


    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\nSlope:", model.coef_)
    print('Intercept:', model.intercept_)
    print('Mean absolute error: {:.2f}'.format(mae))
    print('Root mean squared error: ', numpy.sqrt(mae))
    print('R2 score: ', r2)


    # Next Day prediction
    nextDay = my_data.tail(2)
    nextDayPred = model.predict(nextDay[['Open', 'High', 'Low', 'Close']].head(1))
    # Rounding it to 2 decimal places for website
    nextDayPredRound = numpy.round(nextDayPred, 2)
    print("Next Day Prediction:", nextDayPred)



    #Compare stock vs predicted stock value
    allStock = model.predict(my_data[['Open', 'High', 'Low', 'Close']])



    #Week Prediction
    seven = []
    weekPrediction = []
    seven.append(my_data[['Open', 'High', 'Low', 'Close']].tail(7).head(1))
    for x in range(5):
        y_pred = model.predict(seven[x])
        weekPrediction.append(y_pred[:, 3])
        seven.append(y_pred)
    print("Week Prediction:", weekPrediction)

    plotvisualization(allStock[:, 3], my_data,weekPrediction,"Linear")
    return model.coef_, model.intercept_, mae, numpy.sqrt(mae), r2,nextDayPredRound, numpy.round(weekPrediction,2)




def gatherdata(ticker):
    # Gather data from 1/1/2017 till today
    start = datetime.datetime(2017, 1, 1)
    # Download a stock data from Yahoo
    end = datetime.date.today()
    stock = yf.download(ticker, start, end)
    #Return all the values from stock
    return stock

def lstmmodel(stock):
    # scale data

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform( stock['Close'].values.reshape(-1, 1))
    prediciton_days =60
    x_train, y_train = [], []
    for i in range(prediciton_days+1, len(scaled_data)):
        x_train.append(scaled_data[i - prediciton_days:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = numpy.array(x_train), numpy.array(y_train)
    x_train = numpy.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Creating model
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    # Fit model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=10, epochs=15)



    # Testing
    scaled_data = scaler.fit_transform(stock['Close'].values.reshape(-1, 1))

    x_test = []
    for i in range(prediciton_days, len(scaled_data)):
        x_test.append(scaled_data[i - prediciton_days:i, 0])

    x_test = numpy.array(x_test)
    x_test1 = numpy.array(x_test)
    x_test = numpy.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


    predictions = model.predict(x_test)
    allStock = scaler.inverse_transform(predictions)

    rms = numpy.sqrt(numpy.mean(numpy.power((numpy.array(x_test1) - numpy.array(predictions)), 2)))

    # Prediction for next day
    NextDay = numpy.array([x_test[x_test.shape[0]-1]])
    predictions = model.predict(NextDay)
    NextDayPrediction = scaler.inverse_transform(predictions)



    weekPred = numpy.array([x_test[x_test.shape[0] - 5]])
    WeekPredArray =[]
    for x in range(5):
        predictions = model.predict(weekPred)
        predInverse = scaler.inverse_transform(predictions)
        WeekPredArray.append(predInverse[0])
        weekPred = numpy.delete(weekPred[0], [0])
        weekPred = numpy.append(weekPred, predictions)
        weekPred = weekPred.reshape(-1, 1)
        weekPred = numpy.array([weekPred])



    plotvisualization(allStock, stock.tail(x_test.shape[0]), WeekPredArray, 'LSTM')
    return numpy.round(NextDayPrediction, 2), numpy.round(WeekPredArray, 2),rms

def datainfo(data):
    print("Data Head :\n {0} \n".format(data.head()))
    print("Data Info :\n {0} \n".format(data.info()))
    print("Data Describe :\n {0} \n".format(data.describe()))
    print("Data Columns :\n {0} \n".format(data.columns))
    print("Check if there are empty values :\n {0} \n".format(data.isna().values.any()))

def datavisualization(stock_data):
    # Box plot
    stock_data.plot(kind="box", subplots=True, layout=(2, 6), sharex=False)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(16, 4)
    plt.subplots_adjust(left=0.05, right=0.999, top=0.8, bottom=0.05)
    plt.savefig('static\\plots\\boxplot.png', dpi=100)
    plt.show()

    # density plot
    stock_data.plot(kind="density", subplots=True, layout=(3, 3), sharex=False)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(16, 4)
    plt.subplots_adjust(left=0.05, right=0.999, top=0.9, bottom=0.01)
    plt.savefig('static\\plots\\densityplot.png', dpi=100)
    plt.show()

def plotvisualization(allStock, stock_data, weekPred, whichModel):

#    plt.close(plt)


    #Plot to compare stock vs predicted stock value
    plt.title('Prediction Of Stock Data')
    plt.plot(stock_data.index, stock_data['Close'], label='Real value')
    plt.plot(stock_data.index, allStock, label='Predicted value')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(16, 4)
    plt.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.12)
    plt.savefig(f'static\\plots\\{whichModel}plot.png', dpi=100)
    plt.show()

    # Plot to compare one week stock vs predicted stock value
    plt.title('Week Prediction Of Stock Data')
    plt.plot(stock_data.tail(10).index, stock_data['Close'].tail(10), label='Real value')
    plt.plot(stock_data.tail(5).index, weekPred, label='Predicted value')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(16, 4)
    plt.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.12)
    plt.savefig(f'static\\plots\\{whichModel}week.png', dpi=100)
    plt.show()


    plt.title('Week Prediction Of Stock Data Knowing The Values')
    plt.plot(stock_data.tail(10).index, stock_data['Close'].tail(10), label='Real value')
    plt.plot(stock_data.tail(5).index, allStock[allStock.shape[0]-5:allStock.shape[0]], label='Predicted value')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(16, 4)
    plt.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.12)
    plt.savefig(f'static\\plots\\{whichModel}weekv2.png', dpi=100)
    plt.show()







app = Flask(__name__ )


@app.route('/')
def home():

    return render_template('index.html')

@app.route('/ticker', methods=['POST'])
def ticker():
    if request.method == 'POST':
        #Take desired ticker from search bar
        ticker= request.form['search']
        #Collection of ticker data such as Price , open ,close
        dataCollection = gatherdata(ticker)
        datainfo(dataCollection)
        # Create charts for given stock
        datavisualization(dataCollection)
        # Preprocess data by removing unnecessary information
        lrModel=preprocessking(dataCollection)
        lstModel=lstmmodel(dataCollection)
        ystrDate=dataCollection.reset_index()
        data=dataCollection.skew()
        print("Summon Yesterday Date:",dataCollection.iloc[dataCollection.shape[0]-1])

        print(data.shape)


    return render_template('ticker.html' ,
        ticker=ticker,
        dataCollection=dataCollection,
        dtype=dataCollection.dtypes,
        ystrClose = numpy.round(dataCollection.tail(5).values, 2),
        lrModel=lrModel,
        weekDate=ystrDate['Date'].tail(5).dt.strftime('%d/%m').values,
        lstModel=lstModel,

                           )





if __name__ == '__main__':

    app.run()









