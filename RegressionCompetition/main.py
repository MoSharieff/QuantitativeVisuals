import numpy as np
import pandas as pd
import random as rd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm

bg = 'black'
fg = 'cyan'

wins = {}

fig = plt.figure(figsize=(10, 6), dpi=120)
modelA=fig.add_subplot(221, projection='3d')
modelB=fig.add_subplot(222, projection='3d')
modelC=fig.add_subplot(223, projection='3d')
modelD=fig.add_subplot(224, projection='3d')
fig.patch.set_facecolor(bg)
fig.tight_layout()

for ii, model in enumerate((modelA, modelB, modelC, modelD)):
    wins[ii] = 0
    model.set_facecolor(bg)
    for ch in ('x', 'y', 'z'):
        model.tick_params(ch, colors=fg)
    for pane in (model.xaxis.pane, model.yaxis.pane, model.zaxis.pane):
        pane.set_facecolor(bg)
        pane.set_edgecolor(bg)


def regression(f):
    def solve(*a, **b):
        x, y = f(*a, **b)
        return x, y, np.linalg.inv(x.transpose().dot(x)).dot(x.transpose().dot(y))
    return solve

@regression
def ModelA(x, y):
    x = np.array([[1] + i for i in x])
    y = np.array([[i] for i in y])
    return x, y

def PltA(B, x, y):
    Z = []
    for xx, yy in zip(x, y):
        tZ = []
        for xxx, yyy in zip(xx, yy):
            tZ.append(B[0][0]*1 + B[1][0]*xxx + B[2][0]*yyy)
        Z.append(tZ)
    return np.array(Z)

@regression
def ModelB(x, y):
    x = np.array([[1, i[0], i[1], i[0]*i[1], i[0]**2, i[1]**2] for i in x])
    y = np.array([[i] for i in y])
    return x, y

def PltB(B, x, y):
    Z = []
    for xx, yy in zip(x, y):
        tZ = []
        for xxx, yyy in zip(xx, yy):
            tZ.append(B[0][0]*1 + B[1][0]*xxx + B[2][0]*yyy + B[3][0]*xxx*yyy + B[4][0]*xxx**2 + B[5][0]*yyy**2)
        Z.append(tZ)
    return np.array(Z)

@regression
def ModelC(x, y):
    x = np.array([[1, i[0]*i[1], i[0]**2, i[1]**2, i[0]**3, i[1]**3] for i in x])
    y = np.array([[i] for i in y])
    return x, y

def PltC(B, x, y):
    Z = []
    for xx, yy in zip(x, y):
        tZ = []
        for xxx, yyy in zip(xx, yy):
            tZ.append(B[0][0]*1 + B[1][0]*xxx*yyy + B[2][0]*xxx**2 + B[3][0]*yyy**2 + B[4][0]*xxx**3 + B[5][0]*yyy**3)
        Z.append(tZ)
    return np.array(Z)

@regression
def ModelD(x, y):
    x = np.array([[1, i[0]*i[1], i[0], i[1]] for i in x])
    y = np.array([[i] for i in y])
    return x, y

def PltD(B, x, y):
    Z = []
    for xx, yy in zip(x, y):
        tZ = []
        for xxx, yyy in zip(xx, yy):
            tZ.append(B[0][0]*1 + B[1][0]*xxx*yyy + B[2][0]*xxx + B[3][0]*yyy)
        Z.append(tZ)
    return np.array(Z)

def Bounds(x, y, n=30):
    minx, maxx = np.min(x), np.max(x)
    miny, maxy = np.min(y), np.max(y)
    dt = (maxx - minx) / (n - 1)
    dw = (maxy - miny) / (n - 1)
    H = np.arange(minx, maxx+dt, dt)
    I  = np.arange(miny, maxy+dw, dw)
    return np.meshgrid(H, I)

def Score(B, X, y):
    y = np.array(y)
    r = [v[0] for v in X.dot(B)]
    return np.sum([(y0 - y1)**2 for y0, y1 in zip(y, r)])

def ErrorLines(B, X, x, y, z, n=10):
    zz = [p[0] for p in X.dot(B)]
    errors = {}
    for ii, (x, y) in enumerate(zip(x, y)):
        minZ, maxZ = zz[ii], z[ii]
        dZ = (maxZ - minZ) / (n - 1)
        errors[ii] = {'x': [x for f in range(n)], 'y': [y for  f in range(n)], 'z': [minZ + f*dZ for f in range(n)]}
    return errors
        
        
        

n = 20
theta = 0
#plt.pause(5)
while True:
    X = [[rd.random(), rd.random()] for i in range(n)]
    Y = [rd.randint(1, 10) for i in range(n)]

    x, y, z = [p[0] for p in X], [p[1] for p in X], Y

    for ii, (name, model) in enumerate(zip(('Model A', 'Model B', 'Model C', 'Model D'), (modelA, modelB, modelC, modelD))):
        model.cla()
        model.set_title('{} | Wins: {}'.format(name, wins[ii]), color=fg)
        model.scatter(x, y, z, color='blue', s=11)
        model.scatter(x, y, z, color='cyan', s=5)
        model.grid(False)
        
    jx, jy = Bounds(x, y)
    scores = []
    
    for plot, beta, regress in zip((modelA, modelB, modelC, modelD), (ModelA, ModelB, ModelC, ModelD), (PltA, PltB, PltC, PltD)):
        Rx, Ry, Beta = beta(X, Y)
        jz = regress(Beta, jx, jy)
        scores.append(Score(Beta, Rx, z))
        for im, errz in ErrorLines(Beta, Rx, x, y, z).items():
            plot.plot(errz['x'], errz['y'], errz['z'], color='red', linewidth=0.3)
        plot.plot_surface(jx, jy, jz, cmap=cm.jet, alpha=0.31, edgecolor='yellow', linewidth=0.2)
        plot.view_init(28 + rd.randint(-4, 4), theta)

    minScore = np.min(scores)
    scoreID = scores.index(minScore)
    wins[scoreID] += 1

    if theta > 360:
        theta = 0
    else:
        theta += 7
   
    plt.pause(0.01)

plt.show()
