
class linear_regression:
    
    def __init__(self,*args,y):
        self.X, self.Y = self.crear_matrices(*args,y=y)
        self.k = len(args)
        self.n = len(y)
        self.coef = self.calcular_b(self.X,self.Y)
        self.R2 = self.R2_()
        self.R2_adj = self.R2_ajust_()
    
    def resumen(self):
        for i,coeficiente in enumerate(self.coef):
            print("Coeficiente {} : {}".format(i, round(coeficiente.item(0),5)))
        print("R2: {}".format(round(self.R2,5)))
        print("R2_ajustado: {}".format(round(self.R2_adj,5)))
    
    def crear_matrices(self,*args,y):
        """    
        Devuelve las matriz X para los regresores y la matriz Y para la respuesta

        PARAMETROS
        ---------------------------------
        *args: estos son los regresores
        y: esta es la respuesta observada
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
        """Devuelve los coeficientes para la regresion lineal

        PARAMETROS
        ---------------------------------
        X: Esta es la matriz de regresores
        Y: Esta es la matriz de la respuesta observada
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
        """Retorna SCE

        PARAMETROS
        ---------------------------------
        X: Esta es la matriz de regresores
        Y: Esta es la matriz de la respuesta observada
        n: Es el numero de datos
        """
        import numpy as np
        XX = X*X.transpose()
        inverseXX = np.linalg.inv(XX)

        identidad = np.identity(n)
        SCE = Y *(identidad - ((X.transpose())*inverseXX*X) ) * Y.transpose()
        return SCE.item(0)
    
    def SCR_matricial(self,X,Y,n):
        """Retorna SCR

        PARAMETROS
        ---------------------------------
        X: Esta es la matriz de regresores
        Y: Esta es la matriz de la respuesta observada
        n: Es el numero de datos
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
        """Retorna STCC

        PARAMETROS
        ---------------------------------
        X: Esta es la matriz de regresores
        n: Es el numero de datos
        """
        import numpy as np
        ones = np.ones(n)
        ones_matrix = np.matrix([ones])
        oneone =ones_matrix*(ones_matrix.transpose())
        ones_final = ones_matrix.transpose() * np.linalg.inv(oneone) * ones_matrix

        STCC = (Y *(np.identity(n) - ones_final)*Y.transpose()).item(0)
        return STCC
   
    def prediccion_(self,x_pred):
        """ Retorna un valor de y estimado
        
        PARAMETROS
        ---------------------------------
        x_pred: Es una lista con los valores de x1,x2,..etc para hacer la estimacion de y """
        import numpy as np
        largo = len(self.coef)
        x_pred.insert(0,1.0)
        y_pred = [self.coef[i]*x_pred[i] for i in range(largo)]
        return np.sum(y_pred)
        

    

        
        

