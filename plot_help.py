"""
Authors: Christoforos Papastergiopoulos, Tamposis Dimitris
Evolutionary algorithms 2019-2020

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as pltcol
from mpl_toolkits.mplot3d import Axes3D
import shapely
from descartes import PolygonPatch


# Simple helper class for getting matplotlib patches from shapely polygons with different face colors
class PlotPatchHelper:
    # a colormap with 41 colors
    CMapColors = np.array([
            [0,0.447,0.741,1],
            [0.85,0.325,0.098,1],
            [0.929,0.694,0.125,1],
            [0.494,0.184,0.556,1],
            [0.466,0.674,0.188,1],
            [0.301,0.745,0.933,1],
            [0.635,0.078,0.184,1],
            [0.333333333,0.333333333,0,1],
            [0.333333333,0.666666667,0,1],
            [0.666666667,0.333333333,0,1],
            [0.666666667,0.666666667,0,1],
            [1,0.333333333,0,1],
            [1,0.666666667,0,1],
            [0,0.333333333,0.5,1],
            [0,0.666666667,0.5,1],
            [0,1,0.5,1],
            [0.333333333,0,0.5,1],
            [0.333333333,0.333333333,0.5,1],
            [0.333333333,0.666666667,0.5,1],
            [0.333333333,1,0.5,1],
            [0.666666667,0,0.5,1],
            [0.666666667,0.333333333,0.5,1],
            [0.666666667,0.666666667,0.5,1],
            [1,0,0.5,1],
            [1,0.333333333,0.5,1],
            [1,0.666666667,0.5,1],
            [1,1,0.5,1],
            [0,0.333333333,1,1],
            [0,0.666666667,1,1],
            [0,1,1,1],
            [0.333333333,0,1,1],
            [0.333333333,0.333333333,1,1],
            [0.333333333,0.666666667,1,1],
            [0.333333333,1,1,1],
            [0.666666667,0,1,1],
            [0.666666667,0.333333333,1,1],
            [0.666666667,0.666666667,1,1],
            [0.666666667,1,1,1],
            [1,0,1,1],
            [1,0.333333333,1,1],
            [1,0.666666667,1,1]
            ])
    
    
    # Alpha controls the opaqueness, Gamma how darker the edge line will be and LineWidth its weight
    def __init__(self, Gamma=1.3, Alpha=0.9, LineWidth=2.0):
        self.Counter = 0
        self.Gamma = Gamma          # darker edge color if Gamma>1 -> faceColor ** Gamma; use np.inf for black
        self.Alpha = Alpha          # opaqueness level (1-transparency)
        self.LineWidth = LineWidth  # edge weight
    
    # circles through the colormap and returns the FaceColor and the EdgeColor (as FaceColor^Gamma)
    def nextcolor(self):
        col = self.CMapColors[self.Counter,:].copy()
        self.Counter = (self.Counter+1) % self.CMapColors.shape[0]
        return (col, col**self.Gamma)
    
    # returns a list of matplotlib.patches.PathPatch from the provided shapely polygons, using descartes; a list is 
    # returned even for a single polygon for common handling
    def get_patches(self, poly):
        if not isinstance(poly, list): # single polygon, make it a one element list for common handling
            poly = [poly]
        patchList = []
        for p in poly:
            fCol, eCol = self.nextcolor()
            patchList.append(PolygonPatch(p, alpha=self.Alpha, FaceColor=fCol, EdgeColor=eCol, 
                                          LineWidth=self.LineWidth))        
        return patchList
    
    
# Plots one or more shapely polygons in the provided axes ax. The named parameter values **kwargs are passed into
# PlotPatchHelper's constructor, e.g. you can write plotShapelyPoly(ax, poly, LineWidth=3, Alpha=1.0). Returns a list
# with the drawn patches objects even for a single polygon, for common handling
def plotShapelyPoly(ax, poly, **kwargs):
    return [ax.add_patch(p) for p in PlotPatchHelper(**kwargs).get_patches(poly)]    
    
# =============================================================================
# figure object
# =============================================================================
class FigureObjects:
    """ Class for storing and updating the figure's objects.
        
        The initializer creates the figure given only the lower and upper bounds (scalars, since the bounds are 
        typically equal in both dimensions).
        
        The update member function accepts a DynNeighborPSO object and updates all elements in the figure.
        
        The figure has a top row of two subplots. The left one is a 3D plot of the peaks function with only the global 
        best-so-far solution (red dot). The right one is the peaks function contour, together with the best-so-far 
        solution (red dot) and the positions of all particles in the current iteration's swarm (smaller black dots).
        The bottom row shows the best-so-far global finess value achieved by the algorithm.
    """
    
    def __init__(self, LowerBound, UpperBound):
        """ Creates the figure that will be updated by the update member function.
            
        All line objects (best solution, swarm, global fitness line) are initialized with NaN values, as we only 
        setup the style. Best-so-far fitness 
        
        The input arguments LowerBound & UpperBound must be scalars, otherwise an assertion will fail.
        """
        
        assert np.isscalar(LowerBound), "The input argument LowerBound must be scalar."
        assert np.isscalar(UpperBound), "The input argument LowerBound must be scalar."
        
        # figure
        self.fig = plt.figure()
        self.ax=[1,2,3]
        self.ax[0] = plt.subplot(211)
        
        self.ax[0].set_title('Best-so-far global best fitness: {:g}'.format(np.nan))
        self.lineBestFit, = self.ax[0].plot([], [])

        self.ax[1] = plt.subplot(223)
        self.ax[1].set_title('Rotated & translated order')
        self.ax[2] = plt.subplot(224)
        self.ax[2].set_title('Remaining after set difference')
        
        # auto-arrange subplots to avoid overlappings and show the plot
        self.fig.tight_layout()
    
    
    def update(self, pso, Stock, Order):
        """ Updates the figure in each iteration provided a PSODynNeighborPSO object. """
        # pso.Iteration is the PSO initialization; setup the best-so-far fitness line xdata and ydata, now that 
        # we know MaxIterations
        if pso.Iteration == -1:
            xdata = np.arange(pso.MaxIterations+1)-1
            self.lineBestFit.set_xdata(xdata)
            self.lineBestFit.set_ydata(pso.GlobalBestSoFarFitnesses)
        
        # update  global best fitness line
        self.lineBestFit.set_ydata(pso.GlobalBestSoFarFitnesses)
        self.ax[0].relim()
        self.ax[0].autoscale_view()
        self.ax[0].title.set_text('Best-so-far global best fitness: {:g}'.format(pso.GlobalBestFitness))
        
        
        particle = pso.GlobalBestPosition
        newOrder = [ shapely.affinity.rotate(shapely.affinity.translate(Order[j], xoff=particle[j*3], yoff=particle[j*3+1]),
                                             particle[j*3+2], origin='centroid') for j in range(len(Order))]
    
    
        remaining = Stock.difference(shapely.ops.cascaded_union(newOrder))
        
        self.ax[1].cla()
        self.ax[2].cla()
        self.ax[1].set_title('Rotated & translated order')
        self.ax[2].set_title('Remaining after set difference')
        pp = plotShapelyPoly(self.ax[1], [Stock]+newOrder)
        pp[0].set_facecolor([1,1,1,1])
        plotShapelyPoly(self.ax[2], remaining)
        self.ax[1].relim()
        self.ax[1].autoscale_view()
        self.ax[2].set_xlim(self.ax[1].get_xlim())
        self.ax[2].set_ylim(self.ax[1].get_ylim())
        
        # because of title and particles positions changing, we cannot update specific artists only (the figure
        # background needs updating); redrawing the whole figure canvas is expensive but we have to
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
# =============================================================================
# output function
# =============================================================================
# we added Stock and Order needed for calculations on update function, loaded via lambda function from main        
def OutputFcn(pso, figObj, Stock, Order):
    """ Our output function: updates the figure object and prints best fitness on terminal.
        
        Always returns False (== don't stop the iterative process)
    """
    if pso.Iteration == -1:
        print('Iter.    Global best')
    print('{0:5d}    {1:.5f}'.format(pso.Iteration, pso.GlobalBestFitness))
    
    figObj.update(pso, Stock, Order)
    
    return False    
   
    
