"""
Authors: Christoforos Papastergiopoulos, Tamposis Dimitris
Evolutionary algorithms 2019-2020
"""
from WoodProblemDefinition import Stock, Order1, Order2, Order3
from shapely.geometry import Point
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as pltcol
import math
import pandas as pd
import shapely
from descartes import PolygonPatch
from shapely.ops import cascaded_union
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import expit
from DEGL import DEGL
from plot_help_degl import plotShapelyPoly, PlotPatchHelper, FigureObjects, OutputFcn


# =============================================================================
# objective function
# =============================================================================
# Stock and Order are loaded via lamdba function from main
def ObjectiveFcn(particle, Stock, Order):

    currentOrder = [ shapely.affinity.rotate(shapely.affinity.translate(Order[j], xoff=particle[j*3], yoff=particle[j*3+1]),
                                             particle[j*3+2], origin='centroid') for j in range(len(Order))]
    currentStock = Stock      
          
    #calculate the area out of stock bounds  using demo_shapely.py tip
    outofboundsArea = shapely.ops.unary_union(currentOrder).difference(currentStock).area    
    
    #overlap area among shapes order = orderarea - unary_union.area
    orderArea = sum([currentOrder[i].area for i in range(0,len(currentOrder))])
    overlapArea = orderArea - shapely.ops.unary_union(currentOrder).area
    
    #measure smoothness criteria using formula from pdf
    remainingStock = currentStock.difference(shapely.ops.unary_union(currentOrder))
    convexhullStock = remainingStock.convex_hull
    alpha = 1.1
    el = (convexhullStock.area / (remainingStock.area + 0.01)) - 1
    smoothness = 1 / (1 + alpha * el) 
    #remainingStock = currentStock.difference(shapely.ops.unary_union(currentOrder))
    #compactness = remainingStock.length ** 2 / (4* np.pi * remainingStock.area)
    #percentage of remaining area
    #remainingpercentageArea = remainingStock.area / currentStock.area
    #distance to zero( different implementations)
    distBL = sum([Point(0,0).distance(currentOrder[i].centroid) for i in range(0,len(currentOrder))])
    #distBL = 0 
    #for i in range(0,len(currentOrder)):
    #    minx, miny, maxx, maxy = currentOrder[i].bounds
    #    distBL += Point(0,0).distance(Point(minx,miny))
    
    #fitnessFunction = remainingpercentageArea * 100 +  100* outofboundsArea + 100 *overlapArea
    fitnessFunction = 10000 * outofboundsArea + 10000 *overlapArea + 1 * distBL
    return fitnessFunction



# =============================================================================
#  main   
# =============================================================================

if __name__ == "__main__":
    # in case someone tries to run it from the command-line...    
    plt.ion()    
        
# =============================================================================
# function that calls PSO for a specific order and stock   
# =============================================================================
def callDEGL(Order, Stock):
    areaOrder = sum([Order[i].area for i in range(0,len(Order))])
    areaStock = [Stock[i].area for i in range(0,len(Stock))]
    for num, stock in enumerate(Stock):
        if areaStock[num] >= areaOrder:
            figObj = FigureObjects(0, 0)
            outFun = lambda x: OutputFcn(x, figObj, stock, Order)
            objFun = lambda x: ObjectiveFcn(x, stock, Order)
            nVars = 3 * len(Order)
            minx, miny, maxx, maxy = stock.bounds
            LowerBounds = [minx, miny, 0]   * len(Order)
            UpperBounds = [maxx, maxy, 360] * len(Order)
            deglobj = DEGL(objFun, nVars, LowerBounds= LowerBounds, UpperBounds=UpperBounds, 
                 OutputFcn=outFun, UseParallel=False, MaxStallIterations=20)
            deglobj.optimize()
            if checkPlacement(deglobj.GlobalBestPosition, Order, stock)<0.001:
                return deglobj.GlobalBestFitness, deglobj.GlobalBestPosition, num
    #scenario where there is no stock that fits order        
    return None, None, None
 



# =============================================================================
# checks if our solution has shapes that are out of bounds or overlapping
# =============================================================================
def checkPlacement(particle, Order, Stock):
    #calculate area
    currentOrder = [ shapely.affinity.rotate(shapely.affinity.translate(Order[j], xoff=particle[j*3], yoff=particle[j*3+1]),
                                               particle[j*3+2], origin='centroid') for j in range(len(Order))]
    #calculate the area out of stock bounds  using demo_shapely.py tip
    outofboundsArea = shapely.ops.cascaded_union(currentOrder).difference(Stock).area    
    
    #overlap area among shapes order = orderarea - unary_union.area
    orderArea = sum([currentOrder[i].area for i in range(len(currentOrder))])
    overlapArea = orderArea - shapely.ops.cascaded_union(currentOrder).area
    
    return (outofboundsArea + overlapArea)


# =============================================================================
# splits an order in half
# =============================================================================
def split_order(Order):
    half = len(Order)//2
    return Order[:half], Order[half:]


# =============================================================================
# update Stock
# =============================================================================
def update_stock(particle, Stock, Order):
    newOrder = [ shapely.affinity.rotate(shapely.affinity.translate(Order[j], xoff=particle[j*3], yoff=particle[j*3+1]),
                                             particle[j*3+2], origin='centroid') for j in range(len(Order))]
    
    remaining = Stock.difference(shapely.ops.cascaded_union(newOrder))
    #joinStyle = shapely.geometry.JOIN_STYLE.mitre
    #opening = remaining.buffer(-0.3, join_style=joinStyle).buffer(0.3, join_style=joinStyle)  
    return remaining


# =============================================================================
# loop that solves our problem
# =============================================================================
'''
Orders are calculated in sequence order1>order2>order3.
 - callDEGL calculates the area of order and stock pieces and sequentally looks for an acceptable solution
     in stock pieces whose area is bigger than order area. The first acceptable answer(no overlap area) is returned.
     If there are no pieces that are big enough , callDEGL returns a (None, None, None) to avoid exceptions in our code
     and our main loop splits the order in half. Loop continues until the whole list of orders is calculated.
 - update_stock function updates our Stock after each successful order calculation
 - split_order function splits an order in half
 - checkPlacement function calculates the sum of outofboundsArea and overlapArea. If the sum is smaller than our 
     acceptable 0.1 tolerance the solution is accepted, else the order is split.
    
'''





col = ['number of orders', 'placed properly', 'time', 'total fitness']
degl_results = pd.DataFrame(columns= col)

for k in range(1):
    orderList = [Order1, Order2, Order3]
    #copy Stock in a new variable so we can renew Stock after order placement
    remainingStock = Stock.copy()
    i = 0
    start = time.time()
    placed_correctly = 0
    total_gbf = 0
    while True:
        gbf, gbp, best_stock_num = callDEGL(orderList[i],remainingStock)
        #this weird if covers the case that callPSO doesnt find a stock that fits the order
        #and never runs pso
        if gbf == None:
            area = 1000
        else:
            area = checkPlacement(gbp, orderList[i], remainingStock[best_stock_num])
        if area <= 0.001 : 
            remainingStock[best_stock_num] = update_stock(gbp, remainingStock[best_stock_num], orderList[i])
            placed_correctly += len(orderList[i])
            total_gbf +=gbf
            i+=1
        else:
            #split order and add them to the orderlist
            if len(orderList[i]) >=2:
                temp_order = orderList[i].copy()
                orderList.remove(orderList[i])
                temp1, temp2 = split_order(temp_order)
                orderList.insert(i,temp1)
                orderList.insert(i+1,temp2)
            elif len(orderList[i]) ==1:
                #placement failed
                i +=1
        #check stopFlag
        if i == len(orderList):
            break
            
    end = time.time()
    results = [[len(orderList), placed_correctly, end-start, total_gbf]]
    degl_results = pd.concat([degl_results, pd.DataFrame(results, columns=col)], sort=True )



print(degl_results)

#degl_results.to_excel("degl.xlsx")



fig, ax = plt.subplots(ncols=4,nrows=2, figsize=(16,9))
fig.canvas.set_window_title('remaining Stock')
for i in range(2):
    for j in range(4):
        plotShapelyPoly(ax[i][j], remainingStock[(i*4)+j])
        (minx, miny, maxx, maxy) = remainingStock[(i*4)+j].bounds
        ax[i][j].set_ylim(bottom=miny,top=maxy)
        ax[i][j].set_xlim(left=minx ,right=maxx)
        ax[i][j].set_title('Stock[%d]' %((i*4)+j))