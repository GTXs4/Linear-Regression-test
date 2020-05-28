
class linear_regression:
    
    """
    Crea un objeto que realiza una regresion lineal. Puedes a traves de este obtener 
    los coeficientes respectivos y el coeficiente de determinacion.
    
    Parametros
    ---------
    :parametro *args: Estos son los regresores. Cada regresor entregado debe ser una lista.
    :parametro y: (list) Esta es la respuesta observada. 
    
    Atributos
    ---------
    :X: (numpy.matrix) Esta es la matriz de regresores
    :Y: (numpy.matrix) Esta es la matriz de respuesta
    :k: (int) Numero de regresores
    :n: (int) Numero de datos o filas
    :coef: (numpy.matrix) Coeficientes de la regresion lineal
    :R2: (float) Coeficiente de determinacion
    :R2_adj: (float) Coeficiente de determinacion ajustado
    
    Metodos
    -------
    :resumen(): Te imprime en pantalla un resumen de la regresion lineal
    :prediccion_(x_pred): (numpy.float64) Retorna una prediccion segun los valores de los regresores que se le ingrese
    
    Notas
    -----
    Esta clase puede resolver una regresion lineal simple
    
    .. math:: y = α + β x + \epsilon
    
    Pero tambien puede resolver una regresion lineal multiple
    
    .. math:: y = {β}_0 +{β}_1 {x}_1 +{β}_2 {x}_2 + \cdots +{β}_n {x}_n + \epsilon
           
    
    Ejemplos
    --------
    Ejemplo de regresion lineal simple
    
    >>> x = [1.3, 3.5, 4.2, 8.0]
    >>> y = [4.1, 8.4, 10.14, 19.1]
    >>> lm1 = linear_regression(x,y=y)
    >>> lm1.resumen()
    Coeficiente 0 : 0.83051
    Coeficiente 1 : 2.25988
    R2: 0.99753
    R2_ajustado: 0.99629
    
    Ejemplo de regresion lineal multiple
    
    >>> x1 = [2.3, 3.1, 4.6, 6.4]
    >>> x2 = [3.4, 9.1, 11.3, 12.3]
    >>> y = [15.2, 33.87, 45.04, 50.24]
    >>> lm2 = linear_regression(x1,x2,y=y)
    >>> lm2.resumen()
    Coeficiente 0 : 0.16005
    Coeficiente 1 : 1.81865
    Coeficiente 2 : 3.15468
    R2: 0.9982
    R2_ajustado: 0.99459
    
    """
    
    def __init__(self,*args,y):
        self.X, self.Y = self.crear_matrices(*args,y=y)
        self.k = len(args)
        self.n = len(y)
        self.coef = self.calcular_b(self.X,self.Y)
        self.R2 = self.R2_()
        self.R2_adj = self.R2_ajust_()
    
    def resumen(self):
        """
        Imprime un resumen de la regresion lineal
        """
        for i,coeficiente in enumerate(self.coef):
            print("Coeficiente {} : {}".format(i, round(coeficiente.item(0),5)))
        print("R2: {}".format(round(self.R2,5)))
        print("R2_ajustado: {}".format(round(self.R2_adj,5)))
    
    def crear_matrices(self,*args,y):
        """    
        Crea las matriz X para los regresores y la matriz Y para la respuesta

        PARAMETROS
        ----------
        :*args: estos son los regresores
        :y: esta es la respuesta observada
        
        RETORNO
        -------
        :X,Y: (numpy.matrix) Devuelve las matriz X para los regresores y la matriz Y para la respuesta
        """
        import numpy as np
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
    
    def calcular_b(self,X,Y):
        """
        Calcula los coeficientes para la regresion lineal

        PARAMETROS
        ----------
        :X: Esta es la matriz de regresores
        :Y: Esta es la matriz de la respuesta observada
        
        RETORNO
        -------
        :b: (numpy.matrix) Devuelve los coeficientes para la regresion lineal
        """ 
        import numpy as np
        XX = X*X.transpose()
        inverseXX = np.linalg.inv(XX)
        b = inverseXX * X * Y.transpose()
        return b
    
    def R2_(self):
        """Retorna el coeficiente de determinacion R2"""
        import numpy as np
        return (self.SCR_matricial(self.X,self.Y,self.n)/self.STCC_matricial(self.Y,self.n))
    
    def R2_ajust_(self):
        """Retorna el coeficiente de determinacion ajustado R2_adjusted"""
        import numpy as np
        numerador = self.SCE_matricial(self.X,self.Y,self.n)/(self.n-self.k-1)
        denominador = self.STCC_matricial(self.Y,self.n)/(self.n-1)
        return 1-(numerador/denominador)
    
    def SCE_matricial(self,X,Y,n):
        """
        Retorna SCE

        PARAMETROS
        ----------
        :X: Esta es la matriz de regresores
        :Y: Esta es la matriz de la respuesta observada
        :n: Es el numero de datos
        """
        import numpy as np
        XX = X*X.transpose()
        inverseXX = np.linalg.inv(XX)

        identidad = np.identity(n)
        SCE = Y *(identidad - ((X.transpose())*inverseXX*X) ) * Y.transpose()
        return SCE.item(0)
    
    def SCR_matricial(self,X,Y,n):
        """
        Retorna SCR

        PARAMETROS
        ----------
        :X: Esta es la matriz de regresores
        :Y: Esta es la matriz de la respuesta observada
        :n: Es el numero de datos
        """
        import numpy as np
        XX = X*X.transpose()
        inverseXX = np.linalg.inv(XX)

        #definimos ones
        ones = np.ones(n)
        ones_matrix = np.matrix([ones])
        oneone =ones_matrix*(ones_matrix.transpose())
        ones_final = ones_matrix.transpose() * np.linalg.inv(oneone) * ones_matrix

        SCR = (Y * (((X.transpose())*inverseXX*X) - ones_final) * Y.transpose()).item(0)
        return SCR
    
    def STCC_matricial(self,Y,n):
        """
        Retorna STCC

        PARAMETROS
        ---------------------------------
        :X: Esta es la matriz de regresores
        :n: Es el numero de datos
        """
        import numpy as np
        ones = np.ones(n)
        ones_matrix = np.matrix([ones])
        oneone =ones_matrix*(ones_matrix.transpose())
        ones_final = ones_matrix.transpose() * np.linalg.inv(oneone) * ones_matrix

        STCC = (Y *(np.identity(n) - ones_final)*Y.transpose()).item(0)
        return STCC
   
    def prediccion_(self,x_pred):
        """ 
        Predice un valor segun los valores de los regresores ingresados
        
        PARAMETROS
        ----------
        :x_pred: Es una lista con los valores de x1,x2,..etc para hacer la estimacion de y 
        
        RETORNO
        -------
        (numpy.float64) Retorna un valor de y estimado
        
        """
        import numpy as np
        largo = len(self.coef)
        x_pred.insert(0,1.0)
        y_pred = [self.coef[i]*x_pred[i] for i in range(largo)]
        return np.sum(y_pred)
        

    

        
        

