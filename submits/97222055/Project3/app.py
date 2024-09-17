import numpy as np
import pandas as pd
from flask import Flask, request
from statsmodels.tsa.ar_model import AutoReg
from utils.common import response_message, read_json_time_series
from utils.interpolation_methods import linear_interpolation
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler
import khayyam as kh
from datetime import date, datetime


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def isup():
    return response_message('API is active')


@app.route('/service1', methods=['GET', 'POST'])
def interpolation():
    req = request.get_json()
    data = pd.DataFrame(req['data'])
    config = pd.Series(req['config'])


    if config['type'] == 'shamsi':
        dates = []
        for d in data.time:
            x = list(d.split('/'))
            temp = kh.JalaliDatetime(x[0], x[1], x[2]).todatetime()
            temp = pd.to_datetime(temp)
            dates.append(temp)
        data.time = dates
    else:
        data.time = pd.to_datetime(data.time, infer_datetime_format=True)


    if config['time'] == 'daily':
        data.time = pd.to_datetime(data.time.dt.strftime('%Y-%m-%d'), infer_datetime_format=True)
    else:
        data.time = pd.to_datetime(data.time.dt.strftime('%Y-%m'), infer_datetime_format=True)


    start = data.time[0]
    end = data.time[-1]
    if config['time'] == 'daily':
        index = pd.date_range(start=start, end=end, freq=eval('pd.offsets.Day(1)'))
    else:
        index = pd.date_range(start=start, end=end, freq=eval('pd.offsets.MonthBegin(1)'))


    data.index = data.time
    for i in index:
        if i not in data.time.to_list():
            data.loc[i] = None

    data = data.sort_index()
    out = data.drop('time', axis=1)

    if config.interpolation == 'linear':
        out.vol = out.vol.interpolate()
    elif config.interpolation == 'polynomial':
        out.vol = out.vol.interpolate(method='polynomial', order=2)

    out = out.reset_index().to_json()
    out = {'data': out}

    return response_message(out)

@app.route('/service2', methods=['GET', 'POST'])
def interpolation2():
    req = request.get_json()
    data = pd.DataFrame(req['data'])
    config = pd.Series(req['config'])

    if config['type'] == 'miladi':
        data.time = pd.to_datetime(data.time, infer_datetime_format=True)
    else:
        pass

    if config['time'] == 'daily':
        data.time = pd.to_datetime(data.time.dt.strftime('%Y-%m-%d'), infer_datetime_format=True)
    else:
        data.time = pd.to_datetime(data.time.dt.strftime('%Y-%m'), infer_datetime_format=True)

    start = data.time[0]
    end = data.time[-1]
    if config['time'] == 'daily':
        index = pd.date_range(start=start, end=end, freq=eval('pd.offsets.Day(1)'))
    else:
        index = pd.date_range(start=start, end=end, freq=eval('pd.offsets.MonthBegin(1)'))

    data.index = data.time
    for i in index:
        if i not in data.time.to_list():
            data.loc[i] = None

    data = data.sort_index()

    if config.interpolation == 'linear':
        data.vol = data.vol.interpolate()
    elif config.interpolation == 'polynomial':
        data.vol = data.vol.interpolate(method='polynomial', order=2)

    data.time = data.index

    arr = []
    for i in range(len(data)):
        date_time_obj = datetime.strptime(str(data['time'].iloc[i]), '%Y-%m-%d %H:%M:%S')
        arr.append(str(date_time_obj).split(' ')[0])

    for i in range(len(arr)):
        temp = arr[i].split('-')
        if temp[1][0] == '0' and temp[2][0] == '0':
            arr[i] = str(kh.JalaliDate(date(int(temp[0]), int(temp[1][1]), int(temp[2][1]))))
        elif temp[1][0] == '0' and temp[2][0] != '0':
            arr[i] = str(kh.JalaliDate(date(int(temp[0]), int(temp[1][1]), int(temp[2]))))
        elif temp[1][0] != '0' and temp[2][0] == '0':
            arr[i] = str(kh.JalaliDate(date(int(temp[0]), int(temp[1]), int(temp[2][1]))))
        else:
            arr[i] = str(kh.JalaliDate(date(int(temp[0]), int(temp[1]), int(temp[2]))))

    data['time'] = arr

    data.set_index('time', inplace=True)
    out = data.reset_index().to_json()
    out = {'data': out}



    return response_message(out)

@app.route('/service3', methods=['GET', 'POST'])
def interpolation3():
    req = request.get_json()
    data = pd.DataFrame(req['data'])
    config = pd.Series(req['config'])

    ts_detector = False
    if config.time_series:
        ts_detector = True
        data = data.set_index(data.time).drop('time', axis=1)
    else:
        data = data.set_index(data.id).drop('id', axis=1)
    feature = data.feature.copy()


    df = data.copy()
    t = (df < np.quantile(df, 0.1)) | (df > np.quantile(df, 0.9))
    outliers1 = [x for x in t['feature']]


    up = data.mean() + 3 * data.std()
    low = data.mean() - 3 * data.std()
    mask = (data > up) | (data < low)
    t = data[mask]
    outliers2 = ['True' if x is True else "False" for x in t['feature']]

    # model = ARIMA(feature, order=(1, 1, 0))
    # model_fit = model.fit()
    model = AutoReg(feature, lags=len(feature) // 4)
    model_fit = model.fit()
    predictions = model_fit.predict(start=0, end=len(feature))
    diff = []
    outliers2 = ['false' for i in range(len(feature))]
    for i in range(1, len(feature)):
        diff.append(abs(predictions[i] - feature[i]))

    diff = [0 if i is 0 else diff[i - 1] for i in range(len(diff) + 1)]
    outliers3 = (diff > np.quantile(diff, 0.9))

    if config.time_series:
        data['method1'] = outliers1
        data['method3'] = outliers3
    else:
        data['method1'] = outliers1
        data['method2'] = outliers2
    out = data.reset_index().to_json()
    out = {'data': out}
    return response_message(out)


@app.route('/service4', methods=['GET', 'POST'])
def interpolation4():
    req = request.get_json()
    data = pd.DataFrame(req['data'])
    config = pd.Series(req['config'])

    if config.method == 'SMOTE':
        min_num = data['class'].value_counts().values[-1]
        if min_num <= 1:
            data = data.append(data[data['class'] == config.minor_class], ignore_index=True)
        oversample = SMOTE(k_neighbors=min(min_num, 6))
        X, y = oversample.fit_resample(data.drop(['class', 'id'], axis=1), data['class'])

    elif config.method == 'Oversampling':
        ros = RandomOverSampler(random_state=1)
        X, y = ros.fit_resample(data.drop(['class', 'id'], axis=1), data['class'])

    elif config.method == 'UnderSampling':
        rus = RandomUnderSampler(random_state=1)
        X, y = rus.fit_resample(data.drop(['class', 'id'], axis=1), data['class'])

    elif config.method == 'Clustercentroids':
        cc = ClusterCentroids()
        X, y = cc.fit_resample(data.drop(['class', 'id'], axis=1), data['class'])


    X['class'] = y
    X['id'] = np.array(range(len(X))) + 1
    out = pd.DataFrame(X, columns=data.columns)
    out = out.reset_index().to_json()
    out = {'data': out}
    return response_message(out)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
