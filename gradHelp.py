import numpy as np

class gradHelp(help):    
    def SGO_G(vector_x, delta):
        #return delta f(x + delta*u), where u~B
        return 0
    
    def Suffix_SGD(time_Tf, set_K, GradOracle):
        
        sigma = 0.5
        sum_x = 0
        
        initPoint_x = np.zeros[10000]
        initPoint_x[0] = set_K[0]
        
        for t in range (1, time_Tf):
            napla = 1/(t*sigma)
            g_t = GradOracle[t]
            
            # PI_K(y) = || x - y||, where x in set_K
            initPoint_x[t] = napla + g_t
        
        for i in range(time_Tf/2+1, time_Tf): 
            sum_x += initPoint_x[i]
            
        result = 2/time_Tf*sum_x
            
        return result
