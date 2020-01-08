import numpy as np
import pandas as pd
import matplotlib.cm as cm
import time
import datetime
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
fig.tight_layout()
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

for panel in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    panel.set_facecolor('black')
    panel.set_edgecolor('black')

ax.tick_params('x', color='red')
ax.tick_params('y', color='red')
ax.tick_params('z', color='red')
    

t = ('SP', 'AAPL', 'AMZN', 'MSFT', 'GS', 'GOOGL', 'IBM', 'HD')
quT = ('SP', 'AAPL', 'GOOGL', 'MSFT')
d = {i:pd.read_csv('{}.csv'.format(i))[:750] for i in t}


dates = d['SP']['Date'].values
closes = np.array([d[tick]['Adj Close'].values for tick in t]).transpose()

r = closes[:-1]/closes[1:] - 1
r = list(reversed(r.tolist()))
dates = list(reversed(dates))
r = np.array(r)

DT = lambda x: time.mktime(datetime.datetime.strptime(x, '%m-%d-%Y').timetuple())
DW = lambda x: datetime.datetime.fromtimestamp(int(x)).strftime('%m-%d-%Y')

months = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
molookup = {m:'0{}'.format(i+1) if i < 9 else '{}'.format(i+1) for i, m in enumerate(months)}

def plotDates(ax, Dates):
    y = ax.get_xticks()
    n = len(y)
    if len(Dates) > 1:
        d0, d1 = Dates[0], Dates[-1]
        d0 = d0.split('-')
        d1 = d1.split('-')
        d0 = DT('{}-{}-{}'.format(molookup[d0[0]], d0[1], d0[2]))
        d1 = DT('{}-{}-{}'.format(molookup[d1[0]], d1[1], d1[2]))
        dX = (d1 - d0) / (n - 1)
        return [DW(k) for k in np.arange(d0, d1+dX, dX)]
    else:
        return Dates


def optimize(r, tr):
    m, n = r.shape
    mu = (1/m)*np.ones(m).dot(r)
    cov = (1/(m-1))*(r - mu).transpose().dot(r - mu)
    ch = cov
    cov = (2*cov).tolist()
    for i, j in enumerate(cov):
        cov[i].append(mu[i])
        cov[i].append(1)

    cov.append(mu.tolist() + [0, 0])
    cov.append(np.ones(n).tolist() + [0, 0])
    cov = np.array(cov)
    bx = np.array([[y] for y in np.zeros(n).tolist() + [tr] + [1]])
    z = np.linalg.inv(cov).dot(bx)
    w = [h[0] for h in z][:-2]
    return mu, ch, w


mu, cov, weights = optimize(r, 0)
sd = np.sqrt(np.diag(cov))

sump = lambda x, y: np.sum([i*j for i, j in zip(x, y)])

a, b = 0, 40
T = 0

XT, YT, ZT = [], [], []
XQ, YQ, ZQ = [], [], []


THETA = -25
GAMMA = 22

Dates = []
while b <= len(r):
    try:
        rh = r[a:b]
        Dates.append(dates[b])
        mu, cov, weights = optimize(rh, 0)
        sd = np.sqrt(np.diag(cov))

        ax.cla()
        ax.scatter(T, sd, mu, color='cyan', s=18)

        for uk, ul in enumerate(t):
            ax.text(T, sd[uk], mu[uk], '%s' % (ul), size=14, color='white')
        
        g0, g1, dG = -0.003, 0.006, 0.0001
       
        xT, yT, zT = [], [], []
        xG, yG, zG = [], [], []
        for i in np.arange(g0, g1+dG, dG):
           
            mean, variance, weights = optimize(rh, i)
            vmu = sump(mean, weights)
            wh = np.array([[ww] for ww in weights])
            pv = wh.transpose().dot(variance.dot(wh))[0][0]
            xT.append(T); yT.append(np.sqrt(pv)); zT.append(vmu)
       
        XT.append(xT); YT.append(yT); ZT.append(zT)

        ax.plot_surface(np.array(XT), np.array(YT), np.array(ZT), cmap=cm.hsv, edgecolor='black', linewidth=0.3, alpha=0.45)
       
        Nn = len(ax.get_xticks())
        Mm = int(Nn/2)

        pdf = plotDates(ax, Dates)
      
        ax.set_xticklabels(pdf, rotation=33, color='red')
        ax.set_yticklabels(['{0:.2f}%'.format(num*100) for num in ax.get_yticks()], color='red')
        ax.set_zticklabels(['{0:.2f}%'.format(num*100) for num in ax.get_zticks()], color='red')

        if len(XT) > 30:
            del XT[0]
            del YT[0]
            del ZT[0]
            del Dates[0]

        if THETA <= 25:
            THETA += 2.7
        else:
            THETA = -25

        if GAMMA <= 32:
            GAMMA += 1
        else:
            GAMMA = 22

        ax.view_init(27, THETA)
        ax.grid(False)
   
        T += 1
        a += 1; b += 1
    except Exception as e:
        print(e)
    plt.pause(0.000001)

plt.show()

