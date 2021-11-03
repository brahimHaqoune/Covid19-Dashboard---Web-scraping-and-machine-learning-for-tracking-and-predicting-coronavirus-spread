import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import datetime as dt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from datetime import date
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

covid_data = pd.read_csv("cleaned_data.csv")

covid_data = covid_data.replace(',','',regex=True)

covid_data = covid_data.set_index("date")

global_cases = pd.to_numeric(covid_data['Total Cases'], errors='coerce').sum()
global_recovered = pd.to_numeric(covid_data['Total Recovered'], errors='coerce').sum()
global_active = pd.to_numeric(covid_data['Active Cases'], errors='coerce').sum()
global_deaths = pd.to_numeric(covid_data['Total Deaths'], errors='coerce').sum()

cols = covid_data.columns.drop('country')
covid_data[cols] = covid_data[cols].apply(pd.to_numeric, errors='coerce')


# Data of yesterday
def data_scraping_worldometers():
    url = 'https://www.worldometers.info/coronavirus/'
    html_page = requests.get(url).text
    soup = BeautifulSoup(html_page, 'html.parser')
    get_table = soup.find("table", id="main_table_countries_yesterday")
    get_table_data = get_table.tbody.find_all("tr")

    dic = {}
    for i in range(len(get_table_data)):
        try:
            key = get_table_data[i].find_all("a", href = True)[0].string
        except:
            key = get_table_data[i].find_all("td")[0].string

        values = [j.string for j in get_table_data[i].find_all("td")]
        dic[key] = values

    column_names = ['Total Cases', 'New Cases', 'Total Deaths', 'New Deaths'
                    , 'Total Recovered', 'New Recovered' , 'Active Cases'
                    , 'Serious, Critical', 'Tot Cases/1M pop', 'Deaths/1M pop'
                    , 'Total Tests', 'Tests/1M pop', 'Population'
                    , '1 Case every X ppl', '1 Death every X ppl', '1 Test every X ppl'
                    , 'New Cases/1M pop', 'New Deaths/1M pop', 'Active Cases/1M pop']
    df = pd.DataFrame(dic).iloc[2:,:].T.iloc[1:,:19]

    df.index_name = "country"

    df.columns = column_names

    df.to_csv("data_yesterday")
    print("Done !")

data_scraping_worldometers()

data_yesterday = pd.read_csv("data_yesterday")

data_yesterday = data_yesterday.replace('\+','',regex=True)
data_yesterday = data_yesterday.replace(',','',regex=True)

data_yesterday['New Cases'] = pd.to_numeric(data_yesterday['New Cases'], errors='coerce')
data_yesterday['New Deaths'] = pd.to_numeric(data_yesterday['New Deaths'], errors='coerce')
data_yesterday['New Recovered'] = pd.to_numeric(data_yesterday['New Recovered'], errors='coerce')
data_yesterday['Active Cases'] = pd.to_numeric(data_yesterday['Active Cases'], errors='coerce')
data_yesterday['Total Cases'] = pd.to_numeric(data_yesterday['Total Cases'], errors='coerce')

# Add date of yesterday
data_yesterday["date"] = dt.date.fromordinal(dt.date.today().toordinal()-1)

data_yesterday.columns=['country', 'Total Cases', 'New Cases', 'Total Deaths', 'New Deaths',
       'Total Recovered', 'New Recovered', 'Active Cases', 'Serious, Critical',
       'Tot Cases/1M pop', 'Deaths/1M pop', 'Total Tests', 'Tests/1M pop',
       'Population', '1 Case every X ppl', '1 Death every X ppl',
       '1 Test every X ppl', 'New Cases/1M pop', 'New Deaths/1M pop',
       'Active Cases/1M pop','date']

topCountries = data_yesterday.sort_values(by=['Total Cases'], ascending=False)
topCountries = topCountries[['country','Total Cases','Total Deaths']]
topCountries = topCountries[:5]

factors1 = pd.DataFrame(columns = ['Total Tests'])
factors2 = pd.DataFrame(columns = ['Hospital beds'])
factors1['Total Tests'] = ['Population 65 above', 'Exports of good service', 'GDP per capital']
factors2['Hospital beds'] = ['Population 65 above', 'Risk of impoverishing expenditure for surgical care','Exports of good service']
####################################################################################
url_confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
url_deaths = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
url_recovered = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'

confirmed = pd.read_csv(url_confirmed)
deaths = pd.read_csv(url_deaths)
recovered = pd.read_csv(url_recovered)

# Unpivot data frames
date1 = confirmed.columns[4:]
total_confirmed = confirmed.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], value_vars=date1, var_name='date', value_name='confirmed')
date2 = deaths.columns[4:]
total_deaths = deaths.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], value_vars=date2, var_name='date', value_name='death')
date3 = recovered.columns[4:]
total_recovered = recovered.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], value_vars=date3, var_name='date', value_name='recovered')

# Merging data frames
covidData = total_confirmed.merge(right = total_deaths, how = 'left', on = ['Province/State', 'Country/Region', 'date', 'Lat', 'Long'])
covidData = covidData.merge(right = total_recovered, how = 'left', on = ['Province/State', 'Country/Region', 'date', 'Lat', 'Long'])

# Converting date column from string to proper date format
covidData['date'] = pd.to_datetime(covidData['date'])

# Replace naN with 0
covidData['recovered'] = covidData['recovered'].fillna(0)

# Create new column
covidData['active'] = covidData['confirmed'] - covidData['death'] - covidData['recovered']

covid_data1 = covidData.groupby(['date'])[['confirmed', 'death', 'recovered', 'active']].sum().reset_index()

covid_data2 = covidData.groupby(['date', 'Country/Region'])[['confirmed', 'death', 'recovered', 'active']].sum().reset_index()
def data_per_country(data, country):
    return data[data['Country/Region'] == country]

def appendData(data, nb):
    data2 = data
    for i in range(nb):
        data = pd.DataFrame(data.append(data2))
    return data
    
#########################################################################################################
def lstm(data, n_days_for_prediction, key):
    if key==1:
        columns1 = ['New Deaths', 'active','Air_transport_registered_carrier_departures_worldwide',
           'Risk_of_impoverishing_expenditure_for_surgical_care',
           'High_technology_exports', 'Government_expenditure_on_education_total',
           'Gross_capital_formation', 'Exports_of_goods_services', 'pop_65_above']
        columns2 = ['New Cases']
    else:
        columns1 = ['New Cases','active','Air_transport_registered_carrier_departures_worldwide',
                    'Risk_of_impoverishing_expenditure_for_surgical_care',
                    'High_technology_exports', 'Government_expenditure_on_education_total',
                    'Gross_capital_formation', 'Exports_of_goods_services', 'pop_65_above']
        columns2 = ['New Deaths']
    data = data.iloc[1:,:]
    X_train = data[columns1]
    y_train = data[columns2]
    sc = StandardScaler()

    X_train, y_train = sc.fit_transform(X_train), sc.fit_transform(y_train)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    regressor = Sequential()
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units=1))

    regressor.compile(optimizer='adam', loss='mean_squared_error')
    regressor.fit(X_train, y_train, epochs=15, batch_size=32)

    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

    predict_period_dates = pd.date_range(date.today() - dt.timedelta(days=15), periods=n_days_for_prediction+11, freq=us_bd).tolist()

    # Make prediction
    prediction = regressor.predict(X_train[-(n_days_for_prediction+11):])
    prediction = sc.inverse_transform(prediction)
    prediction = pd.DataFrame(prediction)
    prediction.columns = ['Predictions']
    prediction = prediction.round(0)
    dates = pd.DataFrame(predict_period_dates)
    dates.columns = ['date']
    dates = pd.DataFrame(dates.date.dt.strftime("%y-%m-%d"))
    return prediction, dates

app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}])

app.layout = html.Div(
children=[
        html.Div(
            className="row",
            children=[
                # Column for user controls
                html.Div(
                    className="four columns div-user-controls",
                    children=[
                        html.A(
                            html.Img(
                                className="logo",
                                src=app.get_asset_url("logo.png"),
                            ),
                        ),
                        html.H2("COVID19 - VISUALISATION AND PREDICTION", style={'color': 'white'}),
                        html.P(
                            """________________________"""
                        ),
                    html.Div([
                            html.P('Top 5 Countries', className='fix_label', style={'color': 'white', 'text-align': 'left', 'fontWeight':'bold'}),
                            dash_table.DataTable(
                            id='table3',
                            columns=[{"name": i, "id": i}
                            for i in topCountries.columns],
                            data=topCountries.to_dict('records'),
                            style_cell=dict(textAlign='center'),
                            style_header=dict(backgroundColor= '#012a3d', fontWeight='bold', color='white'),
                            style_data=dict(backgroundColor= '#003f5c', color='white')
                            )
                    ],className="table0"),
                html.P(
                    """________________________"""
                ),
                html.Div([
                    html.P('Factors Affecting the New Cases', className='fix_label', style={'color': 'white', 'text-align': 'left', 'fontWeight':'bold'}),
                    dash_table.DataTable(
                        id='table1',
                        columns=[{"name": i, "id": i}
                                 for i in factors1.columns],
                        data=factors1.to_dict('records'),
                        style_cell=dict(textAlign='center'),
                        style_header=dict(backgroundColor= '#003f5c', color='white'),
                        style_data=dict(backgroundColor= '#003f5c', color='white')
                    )
                ], className='table'),
                html.P(
                    """________________________"""
                ),
                html.Div([
                    html.P('Factors Affecting the New Deaths', className='fix_label', style={'color': 'white', 'text-align': 'left', 'fontWeight':'bold'}),
                    dash_table.DataTable(
                        id='table2',
                        columns=[{"name": i, "id": i}
                                 for i in factors2.columns],
                        data=factors2.to_dict('records'),
                        style_cell=dict(textAlign='center'),
                        style_header=dict(backgroundColor= '#003f5c', color='white'),
                        style_data=dict(backgroundColor= '#003f5c', color='white')
                    )
                ], className='table'),

                    html.P([html.Br()]),
                    html.P([html.Br()]),
                    html.A(
                        html.Img(
                            className="gif",
                            src=app.get_asset_url("respet_igiene.gif"),
                        ),
                    ),
                    html.P([html.Br()]),
                    html.P([html.Br()]),
                    dcc.Markdown(
                            """
                            Source: [worldometers](https://www.worldometers.info/coronavirus/) & [worldbank](https://data.worldbank.org/indicator)
                            """, style={'color': 'white'}
                    ),
                    html.P([html.Br()]),
                    html.P(
                            """Contact us: """, style={'color': 'white'}
                    ),
                    dcc.Markdown(
                            """
                            LinkedIn: [Brahim](https://www.linkedin.com/in/brahim-haqoune-304a90196/)  &  [Sara](https://www.linkedin.com/in/brahim-haqoune-304a90196/)
                            \n
                            Github: [Brahim](https://github.com/brahimHaqoune/)  &  [Sara](https://github.com/brahimHaqoune/)
                            """, style={'color': 'white'}
                    ),
                    html.P([html.Br()]),
                    html.P([html.Br()]),
                    html.Div([
                        html.P('© 2021', className='fix_label', style={'color': 'white', 'text-align': 'center', 'margin-top':35, 'margin-left':70}),

                        html.A(
                            html.Img(
                                className="logo",
                                src=app.get_asset_url("um6p.png"),
                            ),
                        ),
                    ],className="div-for-charts"),
            ],
                ),
                # Column for app graphs and plots
                html.Div(
                    className="eight columns div-for-charts1 bg-grey",
                    children=[
                        html.Div([
                            html.Div([
                                html.H6(children='Total Cases',
                                        style={'textAlign': 'center',
                                               'color': 'black',
                                               'fontSize': 16,
                                               'fontWeight':'bold'}),
                                html.P(f"{global_cases:,.0f}",
                                       style={'textAlign': 'center',
                                              'color': 'black',
                                              'fontSize': 20})
                            ],className="one-item1"),
                            html.Div([
                                html.H6(children='Total Deaths',
                                        style={'textAlign': 'center',
                                               'color': 'black',
                                               'fontSize':  16,
                                               'fontWeight':'bold'}),
                                html.P(f"{global_deaths:,.0f}",
                                        style={'textAlign': 'center',
                                               'color': 'black',
                                               'fontSize':  20})

                            ],className="two-item1"),
                            html.Div([
                                html.H6(children='Total Recovered',
                                        style={'textAlign': 'center',
                                               'color': 'black',
                                               'fontSize':  16,
                                               'fontWeight':'bold'}),
                                html.P(f"{global_recovered:,.0f}",
                                        style={'textAlign': 'center',
                                               'color': 'black',
                                               'fontSize':  20})

                            ],className="one-item"),
                            html.Div([
                                html.H6(children='Total Active Cases',
                                        style={'textAlign': 'center',
                                               'color': 'black',
                                               'fontSize':  16,
                                               'fontWeight':'bold'}),
                                html.P(f"{global_active:,.0f}",
                                        style={'textAlign': 'center',
                                               'color': 'black',
                                               'fontSize':  20})

                            ],className="two-item"),
                        ],className="div-for-charts"),
                        html.Div([
                            html.A(
                                html.Img(
                                    className="image",
                                    src=app.get_asset_url("stick.png"),
                                ),
                            ),
                            html.A(
                                html.Img(
                                    className="image",
                                    src=app.get_asset_url("stick.png"),
                                ),
                            ),
                        ], className='div-for-charts'),

                        html.Div([
                            html.P('Select a country', style={'color': 'blue', 'text-align': 'left', 'fontWeight':'bold'}),
                            html.Div([
                                    dcc.Dropdown(id = 'w_countries',
                                                         multi = False,
                                                         searchable= True,
                                                         value='Morocco',
                                                         placeholder= 'Select a country',
                                                         options= [{'label': c, 'value': c}
                                                                   for c in (data_yesterday['country'].unique())], className='select-item'),
                            ]),

                        ], className='table1'),
                            html.Div([
                                html.Div([
                                    html.P('date: ' + str(data_yesterday['date'][0]),
                                                   className='fix_label', style={'color': 'black', 'text-align': 'center', 'fontWeight':'bold'}),
                                    dcc.Graph(id = 'confirmed', config={'displayModeBar': False}, className='dcc_compon',
                                              style={'margin-top': '20px'}),
                                    dcc.Graph(id = 'death', config={'displayModeBar': False}, className='dcc_compon',
                                              style={'margin-top': '20px'}),
                                    dcc.Graph(id = 'recovered', config={'displayModeBar': False}, className='dcc_compon',
                                              style={'margin-top': '20px'}),
                                ], className='three columns oops'),

                                html.Div([
                                    dcc.Graph(id = 'pie_chart', config={'displayModeBar': 'hover'}
                                              )
                                ], className='seven columns'),
                        ]),

                        html.Div([
                            html.A(
                                html.Img(
                                    className="image",
                                    src=app.get_asset_url("stick.png"),
                                ),
                            ),
                            html.A(
                                html.Img(
                                    className="image",
                                    src=app.get_asset_url("stick.png"),
                                ),
                            ),
                        ], className='div-for-charts'),

                        html.Div([
                            dcc.Graph(
                                id='global-graph',
                            )
                        ], className='ten columns'),
                        html.Div([
                            html.A(
                                html.Img(
                                    className="image",
                                    src=app.get_asset_url("stick.png"),
                                ),
                            ),
                            html.A(
                                html.Img(
                                    className="image",
                                    src=app.get_asset_url("stick.png"),
                                ),
                            ),
                            html.A(
                                html.Img(
                                    className="image",
                                    src=app.get_asset_url("stick.png"),
                                ),
                            ),
                        ], className='div-for-charts'),
                    html.P([html.Br()]),
                    html.P('Select a country and a number of days :', className='fix_label', style={'color': 'blue', 'fontSize':14, 'fontWeight':'bold'}),
                    dcc.Dropdown(id='countries',
                             multi=False,
                             searchable=True,
                             value='Morocco',
                             placeholder='Select a country',
                             options=[{'label': c, 'value': c}
                                      for c in (data_yesterday['country'].unique())], className='select-item'),
                    html.Div([
                        dcc.RadioItems(
                                    id='n_days',
                                    options=[{'label': i, 'value': i} for i in [1, 3, 5, 7]],
                                    value = 1,
                                    labelStyle={'display': 'inline-block'},
                                    style={'color': 'black', 'fontSize':14}
                        ),
                    ], className='radio-item'),
                    dcc.Graph(id='line_chart3', config={'displayModeBar': 'hover'})
                    ],
                ),
            ],
        )
    ]
)

##########################################################################""

@app.callback(Output('confirmed', 'figure'),
              [Input('w_countries','value')])
def update_confirmed(w_countries):
    value_confirmed = int(data_yesterday[data_yesterday['country'] == w_countries]['New Cases'])
    delta_confirmed = covid_data[covid_data['country'] == w_countries]['New Cases'].iloc[-2] - \
                      covid_data[covid_data['country'] == w_countries]['New Cases'].iloc[-3]

    return {
        'data': [go.Indicator(
               mode='number+delta',
               value=value_confirmed,
                delta={'reference': delta_confirmed,
                   'position': 'right',
                   'valueformat': ',g',
                   'relative': False,
                   'font': {'size': 15}},
               number={'valueformat': ',',
                       'font': {'size': 16}},
               domain={'y': [0, 1], 'x': [0, 1]}
        )],

        'layout': go.Layout(
            title={'text': 'New Cases',
                   'y': 1,
                   'x': 0.5,
                   'xanchor': 'center',
                   'yanchor': 'top'},
            font=dict(color='black'),
            paper_bgcolor='white ',
            plot_bgcolor='white ',
            height = 50,

        )
    }

@app.callback(Output('death', 'figure'),
              [Input('w_countries','value')])
def update_confirmed(w_countries):
    value_death = int(data_yesterday[data_yesterday['country'] == w_countries]['New Deaths'])
    delta_death = covid_data[covid_data['country'] == w_countries]['New Deaths'].iloc[-2] - \
                  covid_data[covid_data['country'] == w_countries]['New Deaths'].iloc[-3]

    return {
        'data': [go.Indicator(
               mode='number+delta',
               value=value_death,
                delta = {'reference': delta_death,
                        'position': 'right',
                        'valueformat': ',g',
                        'relative': False,
                        'font': {'size': 15}},
               number={'valueformat': ',',
                       'font': {'size': 16}},
               domain={'y': [0, 1], 'x': [0, 1]}
        )],

        'layout': go.Layout(
            title={'text': 'New Deaths',
                   'y': 1,
                   'x': 0.5,
                   'xanchor': 'center',
                   'yanchor': 'top'},
            font=dict(color='black'),
            paper_bgcolor='white ',
            plot_bgcolor='white ',
            height = 50,

        )
    }



@app.callback(Output('recovered', 'figure'),
              [Input('w_countries','value')])
def update_confirmed(w_countries):
    value_recovered = int(data_yesterday[data_yesterday['country'] == w_countries]['New Recovered'])
    delta_recovered = covid_data[covid_data['country'] == w_countries]['New Recovered'].iloc[-2] - \
                  covid_data[covid_data['country'] == w_countries]['New Recovered'].iloc[-3]

    return {
        'data': [go.Indicator(
               mode='number+delta',
               value=value_recovered,
                delta={'reference': delta_recovered,
                   'position': 'right',
                   'valueformat': ',g',
                   'relative': False,
                   'font': {'size': 15}},
               number={'valueformat': ',',
                       'font': {'size': 16}},
               domain={'y': [0, 1], 'x': [0, 1]}
        )],

        'layout': go.Layout(
            title={'text': 'New Recovered',
                   'y': 1,
                   'x': 0.5,
                   'xanchor': 'center',
                   'yanchor': 'top'},
            font=dict(color='black'),
            paper_bgcolor='white ',
            plot_bgcolor='white ',
            height = 50,

        )
    }




@app.callback(Output('pie_chart', 'figure'),
              [Input('w_countries','value')])
def update_graph(w_countries):
    covid_data_2 = covid_data.groupby(['date', 'country'])[['Total Cases', 'Total Deaths', 'Total Recovered', 'Active Cases']].sum().reset_index()
    confirmed_value = covid_data_2[covid_data_2['country'] == w_countries]['Total Cases'].iloc[-1]
    death_value = covid_data_2[covid_data_2['country'] == w_countries]['Total Deaths'].iloc[-1]
    recovered_value = covid_data_2[covid_data_2['country'] == w_countries]['Total Recovered'].iloc[-1]
    active_value = covid_data_2[covid_data_2['country'] == w_countries]['Active Cases'].iloc[-1]
    colors = ['orange', '#dd1e35', '#FFFC45', 'green']

    return {
        'data': [go.Pie(
            labels=['Total Cases', 'Total Deaths', 'Active Cases', 'Recovered'],
            values=[confirmed_value, death_value, active_value,  recovered_value],
            pull=[0.1, 0, 0.1, 0.1],
            marker=dict(colors=colors),
            hoverinfo='label+value+percent',
            textinfo='label+value',
            rotation=220,
            insidetextorientation= 'radial'

        )],

        'layout': go.Layout(
            title={'text': (w_countries),
                   'y': 0.75,
                   'x': 0.5,
                   'xanchor': 'center',
                   'yanchor': 'top'},
            titlefont={'color': 'black',
                       'size': 16},
            font=dict(family='sans-serif',
                      color='black',
                      size=12),
            hovermode='closest',
            paper_bgcolor='white ',
            plot_bgcolor='white ',
            legend={'orientation': 'h',
                    'bgcolor': 'white ',
                    'xanchor': 'center', 'x': 0.5, 'y': 1.5}


        )
    }

@app.callback(
    Output('global-graph', 'figure'),
    [Input('w_countries', 'value')])
def update_graph(w_countries):
    covid_data_2 = covid_data.groupby(['date', 'country'])[['New Cases', 'New Deaths', 'New Recovered']].sum().reset_index()
    covid_data_3 = covid_data_2[covid_data_2['country'] == w_countries][['country', 'date', 'New Cases']].reset_index()
    covid_data_4 = covid_data_2[covid_data_2['country'] == w_countries][['country', 'date', 'New Deaths']].reset_index()
    covid_data_5 = covid_data_2[covid_data_2['country'] == w_countries][['country', 'date', 'New Recovered']].reset_index()

    return {
        'data': [go.Scatter(
                x=covid_data_3['date'],
                y=covid_data_3['New Cases'],
                mode='lines+markers',
                name='New Cases',
                line=dict(color='#33FF51', width=2),
                fill='tozeroy'
            ),
            go.Scatter(
                x=covid_data_5['date'],
                y=covid_data_5['New Recovered'],
                mode='lines+markers',
                name='New Recovered',
                line=dict(color='#3372FF', width=2),
                fill='tozeroy'
            ),
            go.Scatter(
                x=covid_data_4['date'],
                y=covid_data_4['New Deaths'],
                mode='lines+markers',
                name='New Deaths',
                line=dict(color='#FF3333', width=2),
                fill='tozeroy'
            )
        ],

        'layout': go.Layout(
            title={'text': 'Daily Statistics (30 days): ' + '<br>' + (w_countries),
                   'y': 0.85,
                   'x': 0.5,
                   'xanchor': 'center',
                   'yanchor': 'top'},
            titlefont={'color': 'black',
                       'size': 16},
            font=dict(family='sans-serif',
                      color='black',
                      size=12),
            hovermode='x',
            paper_bgcolor='white ',
            plot_bgcolor='white ',
            legend={'orientation': 'h',
                    'bgcolor': 'white ',
                    'traceorder': "normal",
                    'xanchor': 'center', 'x': 0.2, 'y': 1},

            xaxis=dict(title='<b>date</b>',
                       color='black',
                       showgrid=True,
                       gridwidth=1,
                       gridcolor='white',
                       showline=True,
                       showticklabels=True,
                       linecolor='black',
                       linewidth=1,
                       ticks='outside',
                       tickfont=dict(
                           family='Aerial',
                           color='black',
                           size=12
                       )),
            yaxis=dict(title=None,
                       color='black',
                       showline=True,
                       showgrid=True,
                       gridwidth=1,
                       gridcolor='white',
                       showticklabels=True,
                       linecolor='black',
                       linewidth=1,
                       ticks='outside',
                       tickfont=dict(
                           family='Aerial',
                           color='black',
                           size=12
                       )
                       )

        )
    }

@app.callback(Output('line_chart3', 'figure'),
              [Input('countries','value'),
               Input('n_days','value')])
def update_graph(countries, n_days):
    indicators = pd.read_csv('indicators')
    indicators = indicators.drop(columns="Unnamed: 0")
    indicators = indicators[indicators['country'] == countries].reset_index()
    data_2 = data_per_country(covid_data2, countries)
    data2 = data_2.iloc[-16:, :]
    data2.fillna(data2.mean(), inplace=True)
    data2['New Cases'] = data2['confirmed'].diff()
    data2['New Deaths'] = data2['death'].diff()
    data2 = data2.rename(columns = {'Country/Region' : 'country'}).reset_index()
    data2 = data2.drop(columns="index")
    indicators = appendData(indicators, 15).reset_index()
    indicators = indicators.drop(columns=["index", "level_0", "country"])
    indicators['date'] = data2['date']
    data = data2.merge(indicators, on='date', how='inner')
    prediction_New_Cases, dates =lstm(data, n_days, 1)
    prediction_New_Deaths, dates = lstm(data, n_days, 2)
    return {
        'data': [go.Scatter(
                x=dates['date'],
                y=prediction_New_Cases['Predictions'],
                mode='lines+markers',
                name='New Cases prediction',
                line=dict(width=3, color='orange')
                ),
                go.Scatter(
                    x=dates['date'],
                    y=prediction_New_Deaths['Predictions'],
                    mode='lines+markers',
                    name='New Deaths prediction',
                    line=dict(width=3, color='blue')
                    ),
                go.Scatter(
                    x=data['date'],
                    y=data['New Cases'],
                    mode='lines+markers',
                    name='New Cases',
                    line=dict(width=3, color='#33FF51')
                ),
                go.Scatter(
                    x=data['date'],
                    y=data['New Deaths'],
                    mode='lines+markers',
                    name='New Cases',
                    line=dict(width=3, color='#FF3333')
                ),
            ],

        'layout': go.Layout(
            title={'text': 'New Cases & New Deaths Prediction' + '<br>' + (countries),
                   'y': 0.93,
                   'x': 0.5,
                   'xanchor': 'center',
                   'yanchor': 'top'},
            titlefont={'color': 'black',
                       'size': 20},
            font=dict(family='sans-serif',
                      color='black',
                      size=12),
            hovermode='closest',
            paper_bgcolor='white ',
            plot_bgcolor='white ',
            legend={'orientation': 'h',
                    'bgcolor': 'white ',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.7},
            margin=dict(r=0),
            xaxis=dict(title='<b>date</b>',
                       color = 'black',
                       showline=True,
                       showgrid=True,
                       showticklabels=True,
                       linecolor='black',
                       linewidth=1,
                       ticks='outside',
                       tickfont=dict(
                           family='Aerial',
                           color='black',
                           size=12
                       )),
            yaxis=dict(title=None,
                       color='black',
                       showline=True,
                       showgrid=True,
                       showticklabels=True,
                       linecolor='black',
                       linewidth=1,
                       ticks='outside',
                       tickfont=dict(
                           family='Aerial',
                           color='black',
                           size=12
                       )
                       )


        )
    }



if __name__ == '__main__':
    app.run_server(debug=True)

