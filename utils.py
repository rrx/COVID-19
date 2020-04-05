import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dateparser
import seaborn as sns
sns.set(style="darkgrid")


def load_population():
    pop_filename='./general_data/world-pop.csv'
    pop = pd.read_csv(pop_filename)
    pop['region'] = pop['region'].fillna('')
    pop['population'] = pop['population'].apply(float)
    pop = pop.set_index(['country', 'region'])
    return pop


def load_data():
    pop = load_population()

    filename='./output.csv'
    df = pd.read_csv(filename, skiprows=0)
    df = df.drop(columns=['Deaths', 'Recovered'])

    df['region'] = df['region'].fillna('')
    df = df.set_index(['country', 'region'])#, 'date'])
    df['date'] = df['date'].apply(pd.to_datetime)
    df = pd.merge(df, pop,  how='inner', left_on=['country', 'region'], right_on=['country', 'region'])
    df['ratio'] = df['Confirmed']/df['population'] * 1000000.0
    df = df.fillna(0)
    df['Confirmed'] = df['Confirmed'].apply(int)
    df = df.reset_index()

    return df, pop

def load_world(pop):
    filename = "csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    world = pd.read_csv(filename, skiprows=0)
    world = world.drop(columns=['Lat', 'Long']).rename(columns={"Province/State": "region", "Country/Region": "country"})
    world['region'] = world['region'].fillna("")
    world = world.groupby(by=['country']).sum()
    world['region'] = ""

    world = pd.merge(world, pop,  how='inner', left_on=['country', 'region'], right_on=['country', 'region'])
    world = world.drop(columns=['region'])
    # rename columns
    columns = dict([(d, str(dateparser.parse(d).date())) for d in world.columns if d not in ['population']])
    world = world.rename(columns=columns)

    world_pop = world.copy()

    for d in world.columns:
        if d in ['country', 'population']:
            continue
        p = world['population']
        dd = str(dateparser.parse(d).date())
        world_pop[dd] = world[d].divide(p)*1000000.0
    world = world.drop(columns=['population'])
    world_pop = world_pop.drop(columns=['population'])
    last_date = max(world_pop.columns)
    world_pop = world_pop.sort_values(by=[last_date], ascending=False)
    last = world_pop[last_date]

    world_pop = world_pop.transpose()
    return world_pop


def plot_top_world(world_pop):
    w = world_pop.transpose()
    last_date = max(w.columns)
    last = w[last_date]
    countries = last[:50]
    labels = list(map(lambda x: "%s - %.1f/M" % (x[0], x[1]), zip(countries.index, countries.values)))
    ax = world_pop[countries.index].plot(rot=30, figsize=(16,16), title='Confirmed in Areas of Interest')
    ax.legend(labels, bbox_to_anchor=(0.5, 0.3), ncol=2)
    ax.set_xlabel("date")
    ax.set_ylabel("confirmed cases per million inhabitants")


def plot_na_aoi(usa, canada, logy=False):
    interest = [('US', 'Texas'), ('US', 'Utah'), ('US', 'California'), ('Canada', 'British Columbia'), ('Canada', 'Ontario')]

    rows = list(filter(lambda x: str(x) > '2020-03-07', usa.index))
    usa = usa.loc[rows]

    rows = list(filter(lambda x: str(x) > '2020-03-07', canada.index))
    canada = canada.loc[rows]

    def f(x):
        if x[0:2] not in interest:
            return False
        
        return True

    usa_areas = list(filter(f, usa.columns))
    canada_areas = list(filter(f, canada.columns))

    u = usa[['Texas', 'Utah', 'California']]
    c = canada[['Ontario', 'British Columbia']]

    of_interest = pd.DataFrame()
    for column in c.columns:
        of_interest[column] = c[column]
    for column in u.columns:
        of_interest[column] = u[column]
    of_interest = of_interest.fillna(0)

    last_date = max(of_interest.index)
    last = of_interest.loc[last_date].sort_values(ascending=False).fillna(0)

    labels = ["%s - %.f/M" % v for v in zip(last.index, last.values)]
    ax = of_interest[last.index].plot(logy=logy, rot=30, figsize=(16,8), title='Confirmed in Areas of Interest')
    ax.legend(labels, bbox_to_anchor=(0.8, 0.3))
    ax.set_xlabel("date")
    ax.set_ylabel("confirmed cases per million inhabitants")


def plot_aoi(world_pop):
    areas_of_interest = ['US', 'Canada', 'Iran', 'China', 'Korea, South', 'Italy', 'Spain',
                    'Germany', 'France', 'United Kingdom', 'Russia', 'Finland', 'Portugal',
                    'Brazil', 'Mexico']

    w = world_pop.transpose()
    last_date = max(w.columns)
    last = w[last_date]

    areas = list(filter(lambda x: x in areas_of_interest, last.index))

    countries = last[areas]
    labels = list(map(lambda x: "%s - %.f/M" % (x[0], x[1]), zip(countries.index, countries.values)))
    ax = world_pop[countries.index].plot(rot=30, figsize=(16,16), title='Confirmed in Areas of Interest')
    ax.legend(labels, bbox_to_anchor=(0.5, 0.3), ncol=2)
    ax.set_xlabel("date")
    ax.set_ylabel("confirmed cases per million inhabitants")
    return world_pop[countries.index]
    

def by_country(df, names):
    df = df.loc[names]
    df = df.fillna(0)
    for d in df.columns:
        if d in ['country', 'population']:
            continue
        p = df['population'].apply(float)
        v = df[d]
        v = v.apply(float)
        v = v.divide(p)*100.0
        df[d] = v
    df = df.transpose()
    return df


def plot_canada(df):
    canada = df.loc[df['country'] == 'Canada']
    canada = canada.drop(columns=['country'])

    canada = canada.set_index(['date', 'region'])['ratio'].unstack(level=-1)
    last_date = max(canada.index).date()
    x = pd.DataFrame(canada.loc[last_date].fillna(0).sort_values(ascending=False))
    canada = canada[x.index]

    labels = list(map(lambda x: "%s - %.1f/M" % x, zip(x.index, x[x.columns[0]].to_list())))

    rows = list(filter(lambda x: str(x) > '2020-03-07', canada.index))
    ax = canada.loc[rows].plot(rot=30, figsize=(16,8), title='Confirmed in Canada')
    ax.set_xlabel("date")
    ax.set_ylabel("confirmed cases per million inhabitants")
    ax.legend(labels)
    return canada


def plot_usa(df):
    usa = df.loc[df['country'] == 'US'].loc[df['region'] != ''].groupby(by=['country', 'region', 'date']).sum()
    last_date = str(max(usa.index.get_level_values(2)).date())
    last_date
    idx = pd.IndexSlice

    x = usa.loc[idx[:,:,[last_date]], idx[['ratio', 'Confirmed', 'population']]].sort_values(by=['ratio'], ascending=False)
    labels = list(map(lambda x: "%s - %.1f/M" % (x[0][1], x[1]), zip(x.index, x[x.columns[0]].to_list())))
    rows = list(filter(lambda x: str(x[2]) > '2020-03-07', usa.index))
    usa = usa.loc[rows]
    usa = usa.reset_index()
    usa = usa.drop(columns=['country'])

    usa = pd.pivot_table(usa, values='ratio', index=['date'],
                        columns=['region'], aggfunc=np.sum).fillna(0)[x.index.get_level_values(1)]

    ax = usa.plot(rot=30, figsize=(16,12), title='Confirmed in USA')
    ax.set_xlabel("date")
    ax.set_ylabel("confirmed cases per million inhabitants")
    ax.legend(labels, bbox_to_anchor=(0.5, 0.2), ncol=2)
    return usa


if __name__ == '__main__':
    df, pop = load_data()
    world_pop = load_world(pop)
    plot_top_world(world_pop)