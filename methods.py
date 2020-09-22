import numpy as np

class linear_regression:
    
    """
    Create an object that performs a linear regression. You can through this obtain 
    the respective coefficients and the coefficient of determination.
    
    Parameters
    ---------
    parameter *args : variable number of arguments 
        These are the regressors. Each returned regressor must be a list.
    
    parameter y : list
        This is the observed response.
    
    Attributes
    ---------
    X : numpy.matrix
        This is the regressor matrix
        
    Y : numpy.matrix 
        This is the response matrix
    
    k : int 
        Number of regressors
    
    n : int 
        Number of data or rows
    
    coef : numpy.matrix 
        Coefficients of linear regression
    
    R2 : float 
        Determination coefficient
    
    R2_adj : float 
        Adjusted coefficient of determination
    
    Methods
    -------
    summary() : None
        Print a summary of the linear regression on the screen
    
    prediction_(x_pred) : numpy.float64 
        Returns a prediction according to the values of the regressors entered
    
    Notes
    -----
    This class can solve a simple linear regression
    
    .. math:: y = α + β x + \epsilon
    
    But you can also solve a multiple linear regression
    
    .. math:: y = {β}_0 +{β}_1 {x}_1 +{β}_2 {x}_2 + \cdots +{β}_n {x}_n + \epsilon
           
    
    Examples
    --------
    Simple linear regression example
    
    >>> x = [1.3, 3.5, 4.2, 8.0]
    >>> y = [4.1, 8.4, 10.14, 19.1]
    >>> lm1 = linear_regression(x,y=y)
    >>> lm1.summary()
    Coefficient 0 : 0.83051
    Coefficient 1 : 2.25988
    R2: 0.99753
    R2_adjusted: 0.99629
    
    Multiple linear regression example
    
    >>> x1 = [2.3, 3.1, 4.6, 6.4]
    >>> x2 = [3.4, 9.1, 11.3, 12.3]
    >>> y = [15.2, 33.87, 45.04, 50.24]
    >>> lm2 = linear_regression(x1,x2,y=y)
    >>> lm2.summary()
    Coefficient 0 : 0.16005
    Coefficient 1 : 1.81865
    Coefficient 2 : 3.15468
    R2: 0.9982
    R2_adjusted: 0.99459
    
    """
    
    def __init__(self,*args,y):
        self.X, self.Y = self.create_matrix(*args,y=y)
        self.k = len(args)
        self.n = len(y)
        self.coef = self.calculate_b(self.X,self.Y)
        self.R2 = self.R2_()
        self.R2_adj = self.R2_adjust_()
    
    def summary(self):
        """
        Print a summary of the linear regression
        """
        
        for i,coef in enumerate(self.coef):
            print("Coefficient {} : {}".format(i, round(coef.item(0),5)))
        print("R2: {}".format(round(self.R2,5)))
        print("R2_adjusted: {}".format(round(self.R2_adj,5)))
    
    def create_matrix(self,*args,y):
        """    
        Create the X matrix for the regressors and the Y matrix for the answer

        Parameters
        ----------
        *args : variable number of lists
            these are the regressors
        
        y : list
            this is the observed response
        
        Returns
        -------
        X,Y : numpy.matrix 
            Returns the X matrix for the regressors and the Y matrix for the answer
        
        """
        
        n = len(y)
        k = len(args)
        ones = np.ones(n)
        X = np.zeros((k+1, n))

        j=0
        X[0] = ones
        for arg in args:
            j+=1
            X[j] = arg

        X = np.matrix(X)
        Y = np.matrix([y])
        return X, Y 
    
    def calculate_b(self,X,Y):
        """
        Calculate the coefficients for linear regression

        Parameters
        ----------
        X : numpy.matrix
            This is the regressor matrix
        
        Y : numpy.matrix
            This is the matrix of the observed response
        
        Returns
        -------
        b : numpy.matrix 
            Returns the coefficients for linear regression
        
        """ 
        
        XX = X*X.transpose()
        inverseXX = np.linalg.inv(XX)
        b = inverseXX * X * Y.transpose()
        return b
    
    def R2_(self):
        """Returns the coefficient of determination R2"""
        
        return (self.SCR_matrix(self.X,self.Y,self.n)/self.STCC_matrix(self.Y,self.n))
    
    def R2_adjust_(self):
        """Returns the adjusted coefficient of determination R2_adjusted"""
        
        numerator = self.SCE_matrix(self.X,self.Y,self.n)/(self.n-self.k-1)
        denominator = self.STCC_matrix(self.Y,self.n)/(self.n-1)
        return 1-(numerator/denominator)
    
    def SCE_matrix(self,X,Y,n):
        """
        Returns SCE

        Parameters
        ----------
        X : numpy.matrix 
            This is the regressor matrix
        
        Y : numpy.matrix 
            This is the matrix of the observed response
        
        n : int
            It is the number of data
        
        """
        
        XX = X*X.transpose()
        inverseXX = np.linalg.inv(XX)

        identity = np.identity(n)
        SCE = Y *(identity - ((X.transpose())*inverseXX*X) ) * Y.transpose()
        return SCE.item(0)
    
    def SCR_matrix(self,X,Y,n):
        """
        Returns SCR

        Parameters
        ----------
        X : numpy.matrix 
            This is the regressor matrix
        
        Y : numpy.matrix 
            This is the matrix of the observed response
        
        n : int
            It is the number of data
        
        """
        
        XX = X*X.transpose()
        inverseXX = np.linalg.inv(XX)

        #we define ones
        ones = np.ones(n)
        ones_matrix = np.matrix([ones])
        oneone = ones_matrix*(ones_matrix.transpose())
        ones_final = ones_matrix.transpose() * np.linalg.inv(oneone) * ones_matrix

        SCR = (Y * (((X.transpose())*inverseXX*X) - ones_final) * Y.transpose()).item(0)
        return SCR
    
    def STCC_matrix(self,Y,n):
        """
        Returns STCC

        Parameters
        ---------------------------------
        X : numpy.matrix  
            This is the regressor matrix
        
        n : int
            It is the number of data
        
        """
        
        ones = np.ones(n)
        ones_matrix = np.matrix([ones])
        oneone = ones_matrix*(ones_matrix.transpose())
        ones_final = ones_matrix.transpose() * np.linalg.inv(oneone) * ones_matrix

        STCC = (Y *(np.identity(n) - ones_final)*Y.transpose()).item(0)
        return STCC
   
    def prediction_(self,x_pred):
        """ 
        Predict a value according to the values of the regressors entered
        
        Parameters
        ----------
        x_pred : list
            It is a list with the values of x1, x2, ...  to make the estimation of y
        
        Returns
        -------
        np.sum(y_pred) : numpy.float64 
            Returns an estimated y value
        
        """
        
        length = len(self.coef)
        x_pred.insert(0,1.0)
        y_pred = [self.coef[i]*x_pred[i] for i in range(length)]
        return np.sum(y_pred)
        

    

        
        

