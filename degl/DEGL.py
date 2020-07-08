import numpy as np
import warnings
from random import randint, random

try:
    from joblib import Parallel, delayed
    import multiprocessing
    HaveJoblib = True
except ImportError:
    HaveJoblib = False



class DEGL:
    """ Differential Evolution Algorithm with global and local neighborhood topologies.
        
        deglobj = DEGL(ObjectiveFcn, nVars, ...) creates the DEGL object stored in variable deglobj and 
            performs all initialization tasks (including calling the output function once, if provided).
        
        deglobj.optimize() subsequently runs the whole iterative process.
        
        After initialization, the degl object has the following properties that can be queried (also during the 
            iterative process through the output function):
            o All the arguments passed during boject (e.g., pso.MaxIterations, pso.ObjectiveFcn,  pso.LowerBounds, 
                etc.). See the documentation of the __init__ member below for supported options and their defaults.
            o Iteration: the current iteration. Its value is -1 after initialization 0 or greater during the iterative
                process.
            o CurrentGenFitness: the current generation fitness for all chromosomes (D x 1)
            o PreviousBestPosition: the best-so-far positions found for each individual (D x nVars)
            o PreviousBestFitness: the fitnesses of the best-so-far individuals (D x 1)
            o GlobalBestFitness: the overall best fitness attained found from the beginning of the iterative process
            o GlobalBestPosition: the overall best position found from the beginning of the iterative process
            o k: the minimum neighborhood size allowed
            o StallCounter: the stall counter value (for updating inertia)
            o StopReason: string with the stopping reason (only available at the end, when the algorithm stops)
            o GlobalBestSoFarFitnesses: a numpy vector that stores the global best-so-far fitness in each iteration. 
                Its size is MaxIterations+1, with the first element (GlobalBestSoFarFitnesses[0]) reserved for the best
                fitness value of the initial swarm. Accordingly, pso.GlobalBestSoFarFitnesses[pso.Iteration+1] stores 
                the global best fitness at iteration pso.Iteration. Since the global best-so-far is updated only if 
                lower that the previously stored, this is a non-strictly decreasing function. It is initialized with 
                NaN values and therefore is useful for plotting it, as the ydata of the matplotlib line object (NaN 
                values are just not plotted). In the latter case, the xdata would have to be set to 
                np.arange(pso.MaxIterations+1)-1, so that the X axis starts from -1.
    """
  
    
    def __init__( self
                , ObjectiveFcn
                , nVars
                , LowerBounds = None
                , UpperBounds = None
                , D = None
                , Nf = 0.1
                , alpha = 0.8
                , beta = 0.8
                , wmin = 0.4
                , wmax = 0.8
                , FunctionTolerance = 1.0e-6
                , MaxIterations = None
                , MaxStallIterations = 20
                , OutputFcn = None
                , UseParallel = False
                ):
        """ The object is initialized with two mandatory positional arguments:
                o ObjectiveFcn: function object that accepts a vector (the particle) and returns the scalar fitness 
                                value, i.e., FitnessValue = ObjectiveFcn(Particle)
                o nVars: the number of problem variables
            The algorithm tries to minimize the ObjectiveFcn.
            
            The arguments LowerBounds & UpperBounds lets you define the lower and upper bounds for each variable. They 
            must be either scalars or vectors/lists with nVars elements. If not provided, LowerBound is set to -1000 
            and UpperBound is set to 1000 for all variables. If vectors are provided and some of its elements are not 
            finite (NaN or +-Inf), those elements are also replaced with +-1000 respectively.
            
            The rest of the arguments are the algorithm's options:
                o D (default:  min(200,10*nVars)): Number of chromosomes, an integer greater than 1.
                o Nf (default: 0.1): Neighborhood size as a fraction.
                o alpha (default: 0.8): Scaling factor.
                o beta (default: 0.8): Scaling factor.
                o wmin (default: 0.4): Minimum weight.
                o wmax (default: 0.8): Maximum weight.
                o FunctionTolerance (default: 1e-6): Iterations end when the relative change in best objective function 
                    value over the last MaxStallIterations iterations is less than options.FunctionTolerance.
                o MaxIterations (default: 200*nVars): Maximum number of iterations.
                o MaxStallIterations (default: 20): Iterations end when the relative change in best objective function 
                    value over the last MaxStallIterations iterations is less than options.FunctionTolerance.
                o OutputFcn (default: None): Output function, which is called at the end of each iteration with the 
                    iterative data and they can stop the solver. The output function must have the signature 
                    stop = fun(pso), returning True if the iterative process must be terminated. pso is the 
                    DynNeighborPSO object (self here). The output function is also called after swarm initialization 
                    (i.e., within this member function).
                o UseParallel (default: False): Compute objective function in parallel when True. The latter requires
                    package joblib to be installed (i.e., pip install joplib or conda install joblib).

        """
        
        self.ObjectiveFcn = ObjectiveFcn
        self.nVars = nVars
        #degl
        self.alpha = alpha
        self.beta = beta
        self.wmin = wmin
        self.wmax = wmax
        self.Nf=Nf
       
        # assert options validity (simple checks only) & store them in the object
        if D is None:
            self.D = min(200, 10*nVars)
        else:
            assert np.isscalar(D) and D > 1, \
                "The D option must be a scalar integer greater than 1."
            self.D = max(2, int(round(self.D)))
        
       
        assert np.isscalar(FunctionTolerance) and FunctionTolerance >= 0.0, \
                "The FunctionTolerance option must be a scalar number greater or equal to 0."
        self.FunctionTolerance = FunctionTolerance
        
        
        if MaxIterations is None:
            self.MaxIterations = 100*nVars
        else:
            assert np.isscalar(MaxIterations), "The MaxIterations option must be a scalar integer greater than 0."
            self.MaxIterations = max(1, int(round(MaxIterations)))
        assert np.isscalar(MaxStallIterations), \
            "The MaxStallIterations option must be a scalar integer greater than 0."
        self.MaxStallIterations = max(1, int(round(MaxStallIterations)))
        
        self.OutputFcn = OutputFcn
        assert np.isscalar(UseParallel) and (isinstance(UseParallel,bool) or isinstance(UseParallel,np.bool_)), \
            "The UseParallel option must be a scalar boolean value."
        self.UseParallel = UseParallel
       
        # lower bounds
        if LowerBounds is None:
            self.LowerBounds = -1000.0 * np.ones(nVars)
        elif np.isscalar(LowerBounds):
            self.LowerBounds = LowerBounds * np.ones(nVars)
        else:
            self.LowerBounds = np.array(LowerBounds, dtype=float)
        self.LowerBounds[~np.isfinite(self.LowerBounds)] = -1000.0
        assert len(self.LowerBounds) == nVars, \
            "When providing a vector for LowerBounds its number of element must equal the number of problem variables."
        # upper bounds
        if UpperBounds is None:
            self.UpperBounds = 1000.0 * np.ones(nVars)
        elif np.isscalar(UpperBounds):
            self.UpperBounds = UpperBounds * np.ones(nVars)
        else:
            self.UpperBounds = np.array(UpperBounds, dtype=float)
        self.UpperBounds[~np.isfinite(self.UpperBounds)] = 1000.0
        assert len(self.UpperBounds) == nVars, \
            "When providing a vector for UpperBounds its number of element must equal the number of problem variables."
        
        assert np.all(self.LowerBounds <= self.UpperBounds), \
            "Upper bounds must be greater or equal to lower bounds for all variables."
        
        
        # check that we have joblib if UseParallel is True
        if self.UseParallel and not HaveJoblib:
            warnings.warn("""If UseParallel is set to True, it requires the joblib package that could not be imported; swarm objective values will be computed in serial mode instead.""")
            self.UseParallel = False

            
        # DEGL initialization
        
        # Initial matrices: randomly in [lower,upper] and if any is +-Inf in [-1000, 1000]
        lbMatrix = np.tile(self.LowerBounds, (self.D, 1))
        ubMatrix = np.tile(self.UpperBounds, (self.D, 1))
        bRangeMatrix = ubMatrix - lbMatrix
        self.z = lbMatrix + np.random.rand(self.D,nVars) * bRangeMatrix       
        
        # Initial fitness
        self.CurrentGenFitness = np.zeros(self.D)
        self.__evaluateGenInit()
        
        # Initial best-so-far individuals and global best
        self.PreviousBestPosition = self.z.copy()
        self.PreviousBestFitness = self.CurrentGenFitness.copy()
        
        bInd = self.CurrentGenFitness.argmin()
        self.GlobalBestFitness = self.CurrentGenFitness[bInd].copy()
        self.GlobalBestPosition = self.PreviousBestPosition[bInd, :].copy()
        
        # iteration counter starts at -1, meaning initial population
        self.Iteration = -1;
        
        # neighborhood size
        self.k = int(np.floor(self.D*self.Nf))
        
        self.StallCounter = 0;
        
        # Keep the global best of each iteration as an array initialized with NaNs. First element is for initial swarm,
        # so it has self.MaxIterations+1 elements. Useful for output functions, but is also used for the insignificant
        # improvement stopping criterion.
        self.GlobalBestSoFarFitnesses = np.zeros(self.MaxIterations+1)
        self.GlobalBestSoFarFitnesses.fill(np.nan)
        self.GlobalBestSoFarFitnesses[0] = self.GlobalBestFitness
        
        # call output function, but neglect the returned stop flag
        if self.OutputFcn:
            self.OutputFcn(self)
    
    
    def __evaluateGenInit(self):
        """ Helper private member function that evaluates the population, by calling ObjectiveFcn either in serial or
            parallel mode, depending on the UseParallel option during initialization.
        """
        if self.UseParallel:
            nCores = multiprocessing.cpu_count()
            self.CurrentGenFitness[:] = Parallel(n_jobs=nCores)( 
                    delayed(self.ObjectiveFcn)(self.z[i,:]) for i in range(self.D) )
        else:
            self.CurrentGenFitness[:] = [self.ObjectiveFcn(self.z[i,:]) for i in range(self.D)]
            
    def __evaluateGen(self):
        """ Helper private member function that evaluates the population, by calling ObjectiveFcn either in serial or
            parallel mode, depending on the UseParallel option during initialization.
        """
        if self.UseParallel:
            nCores = multiprocessing.cpu_count()
            self.CurrentGenFitness[:] = Parallel(n_jobs=nCores)( 
                    delayed(self.ObjectiveFcn)(self.u[i,:]) for i in range(self.D) )
        else:
            self.CurrentGenFitness[:] = [self.ObjectiveFcn(self.u[i,:]) for i in range(self.D)]
    
#        
    def optimize( self ):
        """ Runs the iterative process on the initialized population. """
        nVars = self.nVars
        k = self.k      
        # start the iteration
        doStop = False 
        #initialize matrices used for mutation and crossover
        L =np.zeros([self.D,nVars])
        g =np.zeros([self.D,nVars])
        y =np.zeros([self.D,nVars])
        self.u =np.zeros([self.D,nVars])
        
        while not doStop:
            self.Iteration += 1
            
            #step 3 calculate weight
            weight = self.wmin + (self.wmax-self.wmin) * ((self.Iteration-1)/(self.MaxIterations-1))
            
            for p in range(self.D):                
                #calculate mutation(step 8)
                #p-q indeces, modulo for negative/bigger than self.D indeces, i!=p to avoid choosing p
                indeces = np.array([((i+self.D) % self.D) for i in range((p-k), (p+k+1)) if i!= p])                
                select_pq = np.random.choice(indeces, 2, replace=False)
                p = select_pq[0]
                q = select_pq[1]
                
                #chromosome in local neighborhood with best fitness              
                bInd = self.PreviousBestFitness[indeces].argmin()
                bestNeighbor = indeces[bInd]
                zbest_neighbor = self.z[bestNeighbor]
                
                #calculate local mutation
                L[p,:]=self.z[p,:] + self.alpha * (zbest_neighbor-self.z[p,:])+self.beta*(self.z[p,:]-self.z[q,:])                
                
                #chromosome  with best fitness globally
                g_indeces = np.array([i for i in range(self.D) if i!= p])
                select_r1r2 = np.random.choice(g_indeces, 2, replace=False)
                r1 = select_r1r2[0]
                r2 = select_r1r2[1]
                
                bInd = self.PreviousBestFitness.argmin()
                zbest = self.z[bInd]               
                
                #calculate global mutation
                g[p,:]=self.z[p,:] + self.alpha * (zbest-self.z[p,:])+self.beta*(self.z[r1,:]-self.z[r2,:])
                
                #total mutation
                y[p,:]=weight*g[p,:] + (1-weight)*L[p,:]
                
                
                #calculate Crossover(step 9), crossover mutation set to 0.8
                cr = 0.8
                jrand = randint(0,self.D)
                
                for i in range(0, nVars):
                    if(random()<(cr or i==jrand)):
                        self.u[p,i] = y[p,i]
                    else:
                        self.u[p,i] = self.z[p,i]  
                

                
                # check bounds violation(step 10)
                posInvalid = self.u[p,:] < self.LowerBounds
                self.u[p,posInvalid] = self.LowerBounds[posInvalid]
                
                posInvalid = self.u[p,:] > self.UpperBounds
                self.u[p,posInvalid] = self.UpperBounds[posInvalid]
            
            
            # calculate new fitness & update best(step 11)
            self.__evaluateGen()
            genProgressed = self.CurrentGenFitness < self.PreviousBestFitness
            self.PreviousBestPosition[genProgressed, :] = self.u[genProgressed, :]
            #step(17)
            self.z[genProgressed, :] = self.u[genProgressed, :]
            self.PreviousBestFitness[genProgressed] = self.CurrentGenFitness[genProgressed]
            
            # update global best, adaptive neighborhood size and stall counter
            newBestInd = self.CurrentGenFitness.argmin()
            newBestFit = self.CurrentGenFitness[newBestInd]
            
            if newBestFit < self.GlobalBestFitness:
                self.GlobalBestFitness = newBestFit
                self.GlobalBestPosition = self.z[newBestInd, :].copy()                
                self.StallCounter = max(0, self.StallCounter-1)
            else:
                self.StallCounter += 1
                
 #               
            # first element of self.GlobalBestSoFarFitnesses is for self.Iteration == -1
            self.GlobalBestSoFarFitnesses[self.Iteration+1] = self.GlobalBestFitness
 #           
            # run output function and stop if necessary
            if self.OutputFcn and self.OutputFcn(self):
                self.StopReason = 'OutputFcn requested to stop.'
                doStop = True
                continue
 #           
            # stop if max iterations
            if self.Iteration >= self.MaxIterations-1:
                self.StopReason = 'MaxIterations reached.'
                doStop = True
                continue
 #           
            # stop if insignificant improvement
            if self.Iteration > self.MaxStallIterations:
                # The minimum global best fitness is the one stored in self.GlobalBestSoFarFitnesses[self.Iteration+1]
                # (only updated if newBestFit is less than the previously stored). The maximum (may be equal to the 
                # current) is the one  in self.GlobalBestSoFarFitnesses MaxStallIterations before.
                minBestFitness = self.GlobalBestSoFarFitnesses[self.Iteration+1]
                maxPastBestFit = self.GlobalBestSoFarFitnesses[self.Iteration+1-self.MaxStallIterations]
                if (maxPastBestFit == 0.0) and (minBestFitness < maxPastBestFit):
                    windowProgress = np.inf  # don't stop
                elif (maxPastBestFit == 0.0) and (minBestFitness == 0.0):
                    windowProgress = 0.0  # not progressed
                else:
                    windowProgress = abs(minBestFitness - maxPastBestFit) / abs(maxPastBestFit)
                if windowProgress <= self.FunctionTolerance:
                    self.StopReason = 'Population did not improve significantly the last MaxStallIterations.'
                    doStop = True
            
  #      
        # print stop message
        print('Algorithm stopped after {} iterations. Best fitness attained: {}'.format(
                self.Iteration+1,self.GlobalBestFitness))
        print(f'Stop reason: {self.StopReason}')
        
            



