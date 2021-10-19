import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import requests
from bs4 import BeautifulSoup
import datetime as dt

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
                        html.H2("COVID19 - VISUALISATION AND PREDICTION"),
                        html.P(
                            """________________________"""
                        ),
                    html.Div([
                            html.P('Top 5 Countries', className='fix_label', style={'color': 'orange', 'text-align': 'left', 'fontWeight':'bold'}),
                            dash_table.DataTable(
                            id='table3',
                            columns=[{"name": i, "id": i}
                            for i in topCountries.columns],
                            data=topCountries.to_dict('records'),
                            style_cell=dict(textAlign='center'),
                            style_header=dict(backgroundColor="rgb(30, 30, 30)", fontWeight='bold', color='white'),
                            style_data=dict(backgroundColor="rgb(50, 50, 50)", color='white')
                            )
                    ],className="table0"),
                html.P(
                    """________________________"""
                ),
                html.Div([
                    html.P('Factors Affecting the New Cases', className='fix_label', style={'color': 'green', 'text-align': 'left', 'fontWeight':'bold'}),
                    dash_table.DataTable(
                        id='table1',
                        columns=[{"name": i, "id": i}
                                 for i in factors1.columns],
                        data=factors1.to_dict('records'),
                        style_cell=dict(textAlign='center'),
                        style_header=dict(backgroundColor="rgb(30, 30, 30)", color='white'),
                        style_data=dict(backgroundColor="rgb(30, 30, 30)", color='white')
                    )
                ], className='table'),
                html.P(
                    """________________________"""
                ),
                html.Div([
                    html.P('Factors Affecting the New Deaths', className='fix_label', style={'color': 'red', 'text-align': 'left', 'fontWeight':'bold'}),
                    dash_table.DataTable(
                        id='table2',
                        columns=[{"name": i, "id": i}
                                 for i in factors2.columns],
                        data=factors2.to_dict('records'),
                        style_cell=dict(textAlign='center'),
                        style_header=dict(backgroundColor="rgb(30, 30, 30)", color='white'),
                        style_data=dict(backgroundColor="rgb(30, 30, 30)", color='white')
                    )
                ], className='table'),
                    html.P(
                        """________________________"""
                    ),
                    dcc.Markdown(
                            """
                            Source: [worldometers](https://www.worldometers.info/coronavirus/) & [worldbank](https://data.worldbank.org/indicator)
                            """
                    ),
                    html.P(
                            """Contact us: """
                    ),
                    dcc.Markdown(
                            """
                            LinkedIn: [Brahim](https://www.linkedin.com/in/brahim-haqoune-304a90196/)  &  [Sara](https://www.linkedin.com/in/brahim-haqoune-304a90196/)
                            \n
                            Github: [Brahim](https://github.com/brahimHaqoune/)  &  [Sara](https://github.com/brahimHaqoune/)
                            """
                    ),
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
                                               'color': 'white',
                                               'fontSize': 16,
                                               'fontWeight':'bold'}),
                                html.P(f"{global_cases:,.0f}",
                                       style={'textAlign': 'center',
                                              'color': 'white',
                                              'fontSize': 20})
                            ],className="one-item1"),
                            html.Div([
                                html.H6(children='Total Deaths',
                                        style={'textAlign': 'center',
                                               'color': 'white',
                                               'fontSize':  16,
                                               'fontWeight':'bold'}),
                                html.P(f"{global_deaths:,.0f}",
                                        style={'textAlign': 'center',
                                               'color': 'white',
                                               'fontSize':  20})

                            ],className="two-item1"),
                            html.Div([
                                html.H6(children='Total Recovered',
                                        style={'textAlign': 'center',
                                               'color': 'white',
                                               'fontSize':  16,
                                               'fontWeight':'bold'}),
                                html.P(f"{global_recovered:,.0f}",
                                        style={'textAlign': 'center',
                                               'color': 'white',
                                               'fontSize':  20})

                            ],className="one-item"),
                            html.Div([
                                html.H6(children='Total Active Cases',
                                        style={'textAlign': 'center',
                                               'color': 'white',
                                               'fontSize':  16,
                                               'fontWeight':'bold'}),
                                html.P(f"{global_active:,.0f}",
                                        style={'textAlign': 'center',
                                               'color': 'white',
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
                            html.A(
                                html.Img(
                                    className="image",
                                    src=app.get_asset_url("stick.png"),
                                ),
                            ),
                        ], className='div-for-charts'),

                        html.Div([
                            html.P('Select a country', style={'color': 'orange', 'text-align': 'left', 'fontWeight':'bold'}),
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
                                    html.P('Date: ' + str(data_yesterday['date'][0]),
                                                   className='fix_label', style={'color': 'white', 'text-align': 'center', 'fontWeight':'bold'}),
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
            font=dict(color='white'),
            paper_bgcolor='#31302F ',
            plot_bgcolor='#31302F ',
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
            font=dict(color='white'),
            paper_bgcolor='#31302F ',
            plot_bgcolor='#31302F ',
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
            font=dict(color='white'),
            paper_bgcolor='#31302F ',
            plot_bgcolor='#31302F ',
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
            titlefont={'color': 'white',
                       'size': 16},
            font=dict(family='sans-serif',
                      color='white',
                      size=12),
            hovermode='closest',
            paper_bgcolor='#31302F ',
            plot_bgcolor='#31302F ',
            legend={'orientation': 'h',
                    'bgcolor': '#31302F ',
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
            titlefont={'color': 'white',
                       'size': 16},
            font=dict(family='sans-serif',
                      color='white',
                      size=12),
            hovermode='x',
            paper_bgcolor='#31302F ',
            plot_bgcolor='#31302F ',
            legend={'orientation': 'h',
                    'bgcolor': '#3b3939 ',
                    'traceorder': "normal",
                    'xanchor': 'center', 'x': 0.2, 'y': 1},

            xaxis=dict(title='<b>Date</b>',
                       color='white',
                       showgrid=True,
                       gridwidth=1,
                       gridcolor='#31302F',
                       showline=True,
                       showticklabels=True,
                       linecolor='white',
                       linewidth=1,
                       ticks='outside',
                       tickfont=dict(
                           family='Aerial',
                           color='white',
                           size=12
                       )),
            yaxis=dict(title=None,
                       color='white',
                       showline=True,
                       showgrid=True,
                       gridwidth=1,
                       gridcolor='#31302F',
                       showticklabels=True,
                       linecolor='white',
                       linewidth=1,
                       ticks='outside',
                       tickfont=dict(
                           family='Aerial',
                           color='white',
                           size=12
                       )
                       )

        )
    }



if __name__ == '__main__':
    app.run_server(debug=True)

