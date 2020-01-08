import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import random as rd

# source:    GDP | https://www.thebalance.com/us-gdp-by-year-3305543
#                     S&P | yahoo finance

df = pd.read_csv('gdp.csv')
dk = pd.read_csv("SP5.csv")
del dk['Unnamed: 0']
dk = dk[::-1]

years = df['Year'].values
nom = df['Nominal GDP (trillions)'].values
real = df['Real GDP (trillions)'].values
growth = df['GDP Growth Rate'].values

closes = dk['Adj Close'].values
sdates = dk['Date'].values.tolist()

r0 = np.log(closes[1:]/closes[:-1])
del sdates[0]
sdates = np.array(sdates)

cmps = {i:np.sum([l for k, l in zip(sdates, r0) if str(i) in k]) for i in years}

fig = plt.figure(figsize=(12, 7), dpi=120)
ax = fig.add_subplot(111)
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.tick_params('x', colors='cyan')
ax.tick_params('y', colors='cyan')


def president():

    N = lambda n: np.array([i+1 for i in range(n)])
    x = N(len(years))

    for ii, (k, v) in enumerate(sorted(cmps.items())):
        ax.scatter(v, growth[ii], color='red', s=10)
        ax.set_xlabel('S&P 500 Yearly Total Log Return (Summed)', color='cyan')
        ax.set_ylabel('GDP Measured in $(Trillion)', color='cyan')
        plt.pause(0.001)

    plt.show()

def gatherPresidents():
    names = ('HerbertHoover', 'FDR', 'HarryTruman', 'DwightEisenhower', 'JFK', 'LBJ', 'RichardNixon',
                         'GeraldFord', 'JimmyCarter', 'RonaldReagan', 'GHWB', 'BillClinton', 'GWB', 'Obama', 'DT')
    imgs = [plt.imread('{}.png'.format(n)) for n in names]
    scld = [OffsetImage(ii, zoom=0.18) for ii in imgs]

    '''
    for i in range(len(scld)):
        xk, yk = rd.random(), rd.random()
        ax.scatter(xk, yk)
        au = AnnotationBbox(scld[i], (xk, yk), xycoords='data', frameon=False)
        ax.add_artist(au)
        plt.pause(0.001)
    plt.show()
    '''
    return scld

def getPres(cp, ps, mb):
    terms = ((1929, 1933), (1933, 1945), (1945, 1953), (1953, 1961), (1961, 1963), (1964, 1969),
                         (1969, 1974), (1974, 1977), (1977, 1981), (1981, 1989), (1989, 1993), (1993, 2001),
                         (2001, 2009), (2009, 2017), (2017, 2019))

    goat = ('Herbert Hoover', 'Franklin D. Roosevelt', 'Harry Truman', 'Dwight Eisenhower',
                     'John F. Kennedy', 'Lyndon Johnson', 'Richard Nixon', 'Gerald Ford', 'Jimmy Carter', 'Ronald Reagan',
                     'George H.W. Bush', 'Bill Clinton', 'George W. Bush', 'Barack Obama', 'Donald Trump')

    for ii, (t0, t1) in enumerate(terms):
        if ps >= t0 and ps < t1:
            return (t0, t1), cp[ps], mb[ii], ii, goat[ii]

    


pres = gatherPresidents()
ynom = real[1:]/real[:-1] - 1
yyrd = years[1:]

xx, yy = [], []

prez = {}
kly = {}

bLine = lambda a, b, n: np.array([a + i * (b - a)/(n - 1) for i in range(n)])

for yr, nm in zip(yyrd, ynom):
    king = getPres(cmps, yr, pres)
    if king:
        rtn = king[1]
        pic = king[2]
        ax.cla()
        ax.set_title("Year: {} | President: {}".format(yr, king[4]), color='cyan')
        xx.append(nm); yy.append(rtn)
        ax.scatter(xx, yy, color='black')

        ax.scatter(-0.17, -0.66, color='black')
        xMax, yMax = 0.25, 0.6

        bX, bY = bLine(0, xMax, 20), bLine(0, yMax, 20)

        ax.plot(np.zeros(len(bY)), bY, color='limegreen', linewidth=0.9)
        ax.plot(bX, np.zeros(len(bX)), color='limegreen', linewidth=0.9)

        ax.plot(xx, yy, color='red', linewidth=1.9, alpha=0.5)
        ax.plot(xx, yy, color='white', linewidth=1.2, alpha=0.5)
        ax.plot(xx, yy, color='blue', linewidth=0.5, alpha=0.5)
        
        aus = AnnotationBbox(pic, (nm, rtn), xycoords='data', frameon=False)
        prez[king[3]] = aus
        for goo, gow in prez.items():
            ax.add_artist(gow)

        ax.set_xlabel("Real GDP Yearly Change (%)", color='cyan')
        ax.set_ylabel("S&P 500 Yearly Return (% | Log Sum)", color='cyan')
        ax.set_xticklabels(['{0:.2f}%'.format(jo*100) for jo in ax.get_xticks()])
        ax.set_yticklabels(['{0:.2f}%'.format(jo*100) for jo in ax.get_yticks()])
        plt.pause(0.0007)



plt.show()
