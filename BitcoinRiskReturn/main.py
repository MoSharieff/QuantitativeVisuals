import numpy as np
import pandas as pd
import random as rd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import datetime

title = 'Bitcoin'

fig = plt.figure(figsize=(8, 8))
ax=fig.add_subplot(111, projection='3d')
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
fig.tight_layout()

plt.gcf().subplots_adjust(bottom=0.15)
plt.subplots_adjust(top=0.88)

for p in (ax.xaxis, ax.yaxis, ax.zaxis):
    yp = p.pane
    yp.set_facecolor('black')
    yp.set_edgecolor('black')

ax.grid(False)
for p in ('x', 'y', 'z'):
    ax.tick_params(p, colors='cyan')
df = pd.read_csv('bitcoin.csv')

dates = df['Date'].values
closes = np.array([float(str(k).replace(',','')) for k in df['Close'].values])

r0, d0 = closes[:-1]/closes[1:] - 1, dates[:-1]
r0, d0 = [p[::-1] for p in (r0, d0)]
c0 = closes[:-1][::-1]

mos = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
yrs = ['2020' if f == 10 else '201{}'.format(f) for f in range(3,11,1)]
mod = {j:'0{}'.format(i+1) if i < 9 else '{}'.format(i+1) for i, j in enumerate(mos)}
clrz = ('pink', 'teal', 'orange', 'yellow', 'salmon', 'blue', 'purple')

dT = lambda x: time.mktime(datetime.datetime.strptime(x, '%m-%d-%Y').timetuple())
dY = lambda x: datetime.datetime.fromtimestamp(int(x)).strftime('%m-%d-%Y')

VG = lambda n: [[] for j in range(n)]

def grouper(d, r):
    rtns = {y:{j:[] for j in mos} for y in yrs}
    for dg, rg in zip(d, r):
        dg = dg.replace(',','').split(' ')
        sg = '{}-{}-{}'.format(mod[dg[0]], dg[1], dg[2])
        rtns[dg[-1]][dg[0]].append((dT(sg), rg))
    return rtns

def dater(ax, dA):
    no = ax.get_xticks()
    n = len(no) - 1
    m0, m1= np.min(dA), np.max(dA)
    dM = (m1 - m0) / (n - 1)
    vh = []
    for i in range(n):
        vh.append(dY(m0 + i*dM))
    return vh

def gainSphere(x, y, z):
    mx0, mx1 = np.mean(x), np.max(x)
    my0, my1 = np.mean(y), np.max(y)
    mz0, mz1 = np.mean(z), np.max(z)
    rx, ry, rz = VG(3)
    for i in np.arange(0, np.pi+np.pi/16, np.pi/16):
        ux, uy, uz = VG(3)
        for j in np.arange(0, np.pi+np.pi/16, np.pi/16):
            ux.append((mx1 - mx0)*np.sin(i)*np.cos(j))
            uy.append((my1 - my0)*np.sin(i)*np.sin(j))
            uz.append((mz1 - mz0)*np.cos(i))
        rx.append(ux); ry.append(uy); rz.append(uz)
    return {'v': [np.array(h) for h in (rx, ry, rz)], 'c': (mx0, my0, mz0)}

def lossSphere(x, y, z):
    mx0, mx1 = np.mean(x), np.min(x)
    my0, my1 = np.mean(y), np.min(y)
    mz0, mz1 = np.mean(z), np.min(z)
    rx, ry, rz = VG(3)
    for i in np.arange(0, np.pi+np.pi/16, np.pi/16):
        ux, uy, uz = VG(3)
        for j in np.arange(0, np.pi+np.pi/16, np.pi/16):
            ux.append((mx1 - mx0)*np.sin(i)*np.cos(j))
            uy.append((my1 - my0)*np.sin(i)*np.sin(j))
            uz.append((mz1 - mz0)*np.cos(i))
        rx.append(ux); ry.append(uy); rz.append(uz)
    return {'v': [np.array(h) for h in (rx, ry, rz)], 'c': (mx0, my0, mz0)}
    
R = grouper(d0, r0)

dateAsses = ['']
ho = []


gx, gy, gz = VG(3)
ax.grid(False)
plt.pause(1)
theta = 27
for OK, (k, ik) in enumerate(R.items()):
    dateAsses.append(k)
    gx, gy, gz = VG(3)
    for PK, (l, il) in enumerate(ik.items()):
        if il:
            rmu = np.mean([i[1] for i in il])
            rsd = np.std([i[1] for i in il])
            tn = il[-1][0]

            gx.append(tn); gy.append(rsd); gz.append(rmu)

            sate = gainSphere(gx, gy, gz)
            ox, oy, oz = sate['v']
            qx, qy, qz = sate['c']

            mate = lossSphere(gx, gy, gz)
            yx, yy, yz = mate['v']
            bx, by, bz = mate['c']

            ax.scatter(tn, rsd, rmu, color='blue', s=18)
            ax.scatter(tn, rsd, rmu, color='cyan', s=11)
            ax.scatter(tn, rsd, rmu, color='black', s=4)
           
            if PK == len(ik.items()) - 1:
                ax.plot_surface(qx + ox, qy + oy, qz + oz, color='limegreen', alpha=0.3, edgecolor='white', linewidth=0.1)
                ax.plot_surface(bx + yx, by + yy, bz + yz, color='red', alpha=0.3, edgecolor='white', linewidth=0.1)

            ax.set_title(title + ' Monthly Risk/Return', color='cyan')
            ax.set_yticklabels(['{0:.2f}%'.format(hg*100) for hg in ax.get_yticks()])
            ax.set_zticklabels(['{0:.2f}%'.format(hg*100) for hg in ax.get_zticks()])
            ax.view_init(24, theta)
            theta -= 1
            if theta == -27:
                theta = 27
            plt.pause(0.000001)
        ax.set_xticklabels(dateAsses, rotation=30)


plt.locator_params(axis='x', nbins=10)
plt.pause(0.0001)
plt.show()
