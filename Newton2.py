import random
import numpy

def metodo_newton2():
    X = random.uniform(-1, 1)
    fgh = cal_grad(X)
    alfa = 1
    f = fgh[0]
    g = fgh[1]
    h = fgh[2]
    while numpy.norm(g) > 1.0e-5:
        d = numpy.linalg.inv(h) * g
        alfa = metodo_bissecao(X, d)
        X = X + alfa * d
        """
        [f, g, ~] = cal_grad(X)
        disp(sprintf('f = %g, alfa = %g', f, alfa))
        """
        break

def cal_grad(X):
    f = 0.5 * X[0] ^ 2 + 2.5 * X[1] ^ 2
    g = [X[0],5*X[1]]
    h = [[1,0], [0,5]]
    return [f,g,h]

def metodo_bissecao(X,d):
    alfa_l = 0
    alfa_u = random.random()
    Xnew = X + alfa_u * d
    """
    [f,g,~]=cal_grad(Xnew)
    h=g'*d
    while h<0
        alfa_u = 2*alfa_u
        Xnew = X + alfa_u*d
        [f,g,~]=cal_grad(Xnew)
        h=g'*d
    end

    alfa_m = (alfa_l+alfa_u)/2
    k = ceil(log((alfa_u-alfa_l)/1.0e-5))
    nit = 0

    while nit<k & abs(h)>1.0e-5
        Xnew = X + alfa_m*d
        [f,g,~]=cal_grad(Xnew)
        h=g'*d
        if h>0
            alfa_u = alfa_m
        else
            alfa_l = alfa_m
        end
        alfa_m = (alfa_l+alfa_u)/2
    end
    alfa = alfa_m
    end
    """
