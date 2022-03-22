from flask import Flask, request, render_template
import pandas as pd
import pickle


app=Flask(__name__)


def final_prediction(data_points):
    columns = ['totals.sessionQualityDim', 'trafficSource.source', 'visitNumber',
               'channelGrouping', 'totals.timeOnSite', 'trafficSource.keyword',
               'geoNetwork.metro', 'totals.pageviews', 'geoNetwork.city',
               'device.deviceCategory', 'trafficSource.medium', 'device.browser',
               'device.operatingSystem', 'device.isMobile', 'geoNetwork.networkDomain',
               'visitStartTime', 'totals.hits', 'trafficSource.referralPath',
               'fullVisitorId', 'geoNetwork.country',
               'totals.transactions', 'date', 'geoNetwork.continent',
               'geoNetwork.subContinent', 'geoNetwork.region',
               ]

    # create a dataframe from given data points
    data_df = pd.DataFrame(data=data_points, columns=columns)

    # preprocess the boolean feature
    data_df['device.isMobile'] = data_df['device.isMobile'].astype(bool)
    print("Boolean feature preprocessing done..!")

    # preprocess the numerical features
    numerical_cols = ["totals.hits", "totals.pageviews", "visitNumber", "visitStartTime", 'totals.timeOnSite',
                      'totals.transactions']

    for col in numerical_cols:
        data_df[col].fillna(0, inplace=True)
        data_df[col] = data_df[col].astype('float')

    # preprocess the categorical features
    categorical_cols = ["channelGrouping",
                        "device.browser", "device.deviceCategory", "device.operatingSystem",
                        "geoNetwork.city", "geoNetwork.continent", "geoNetwork.country", "geoNetwork.metro",
                        "geoNetwork.networkDomain", "geoNetwork.region", "geoNetwork.subContinent",
                        "trafficSource.keyword", "trafficSource.medium",
                        "trafficSource.referralPath", "trafficSource.source",
                        "totals.sessionQualityDim"]

    for col in categorical_cols:
        with open('label_encoders/' + col + '.pkl','rb') as file:
            le = pickle.load(file)
        data_df[col] = le.transform(list(data_df[col].values.astype('str')))

    # Featurization of data point:

    # max date and min date used for featurization purpose
    data_maxdate = max(data_df['date'])
    data_mindate = min(data_df['date'])

    # Additional features after time series featurization
    data_df = data_df.groupby('fullVisitorId').agg({
        'geoNetwork.networkDomain': [('networkDomain', lambda x: x.dropna().max())],  # max value of network domain
        'geoNetwork.city': [('city', lambda x: x.dropna().max())],  # max value of city
        'device.operatingSystem': [('operatingSystem', lambda x: x.dropna().max())],  # max value of Operating System
        'geoNetwork.metro': [('metro', lambda x: x.dropna().max())],  # max value of metro
        'geoNetwork.region': [('region', lambda x: x.dropna().max())],  # max vaue of region
        'channelGrouping': [('channelGrouping', lambda x: x.dropna().max())],  # max value of channel grouping
        'trafficSource.referralPath': [('referralPath', lambda x: x.dropna().max())],  # max value of referral path
        'geoNetwork.country': [('country', lambda x: x.dropna().max())],  # max value of country
        'trafficSource.source': [('source', lambda x: x.dropna().max())],  # max value of source
        'trafficSource.medium': [('medium', lambda x: x.dropna().max())],  # max value of medium
        'trafficSource.keyword': [('keyword', lambda x: x.dropna().max())],  # max value of keyboard
        'device.browser': [('browser', lambda x: x.dropna().max())],  # max value of browser
        'device.deviceCategory': [('deviceCategory', lambda x: x.dropna().max())],  # max of device category
        'geoNetwork.continent': [('continent', lambda x: x.dropna().max())],  # max of continent value
        'geoNetwork.subContinent': [('subcontinent', lambda x: x.dropna().max())],  # max of sub_continent value
        'totals.timeOnSite': [('timeOnSite_sum', lambda x: x.dropna().sum()),  # total timeonsite of user
                              ('timeOnSite_min', lambda x: x.dropna().min()),  # min timeonsite
                              ('timeOnSite_max', lambda x: x.dropna().max()),  # max timeonsite
                              ('timeOnSite_mean', lambda x: x.dropna().mean())],  # mean timeonsite
        'totals.pageviews': [('pageviews_sum', lambda x: x.dropna().sum()),  # total of page views
                             ('pageviews_min', lambda x: x.dropna().min()),  # min of page views
                             ('pageviews_max', lambda x: x.dropna().max()),  # max of page views
                             ('pageviews_mean', lambda x: x.dropna().mean())],  # mean of page views
        'totals.hits': [('hits_sum', lambda x: x.dropna().sum()),  # total of hits
                        ('hits_min', lambda x: x.dropna().min()),  # min of hits
                        ('hits_max', lambda x: x.dropna().max()),  # max of hits
                        ('hits_mean', lambda x: x.dropna().mean())],  # mean of hits
        'visitStartTime': [('visitStartTime_counts', lambda x: x.dropna().count())],  # Count of visitStartTime
        'totals.sessionQualityDim': [('sessionQualityDim', lambda x: x.dropna().max())],
        # Max value of sessionQualityDim
        'device.isMobile': [('isMobile', lambda x: x.dropna().max())],  # Max value of isMobile
        'visitNumber': [('visitNumber_max', lambda x: x.dropna().max())],  # Maximum number of visits.
        'totals.transactions': [('transactions', lambda x: x.dropna().sum())],
        # Summation of all the transaction counts.
        'date': [('days_before_the_period_start', lambda x: x.dropna().min() - data_mindate),
                 # days_before_the_period_start for current frame.
                 ('days_before_the_period_end', lambda x: data_maxdate - x.dropna().max()),
                 # days_before_the_period_end for current frame.
                 ('interval_dates', lambda x: x.dropna().max() - x.dropna().min()),
                 # interval calculated as the latest date on which customer visited - oldest date on which they visited.
                 ('unqiue_date_num', lambda x: len(set(x.dropna())))],
        # Unique number of dates customer visited.
    })

    # Drop the parent level of features and reset index
    data_df.columns = data_df.columns.droplevel()
    data_df = data_df.reset_index()

    data_df['interval_dates'] = data_df['interval_dates'].dt.days
    data_df['days_before_the_period_start'] = data_df['days_before_the_period_start'].dt.days
    data_df['days_before_the_period_end'] = data_df['days_before_the_period_end'].dt.days

    # Reading pretrained classification model:
    final_pred = 0
    for i in range(10):
        with open('dt_model_clf/dt_clf_itr_' + str(i) + '.txt', 'rb') as file:
            clf = pickle.load(file)
        clf_pred = clf.predict_proba(data_df.drop('fullVisitorId', axis=1))[:, 1]

        with open('dt_model_reg/dt_reg_itr_' + str(i) + '.txt', 'rb') as file:
            reg = pickle.load(file)
        reg_pred = reg.predict(data_df.drop('fullVisitorId', axis=1))

        final_pred = final_pred + (clf_pred * reg_pred)

    final_pred /= 10

    return final_pred

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    test_file = request.form['csvfile']
    test_df = pd.read_csv(test_file,dtype={'fullVisitorId': 'str'}, parse_dates=['date'])
    predictions = final_prediction(test_df[0:1])
    return render_template('index.html',fullVisitorId = 'fullVisitorId is: {}' .format(test_df['fullVisitorId'][0]),
                                        revenue_predictions='Predicted Revenue is: {}' .format(predictions[0]))

if __name__=='__main__':
    app.run(debug=True)