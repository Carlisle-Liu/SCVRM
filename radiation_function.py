import torch

def Exponential_Radiation(x):
    return 1 - torch.exp(-x**2)

def Logistic_Radiation(x):
    # print(torch.exp(-x))
    return 2 / (1 + torch.exp(-x)) - 1

def Hyperbolic_Radiation(x):
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))



class Radiation_Function:
    def __init__(self, mode) -> None:
        if mode == 'Exponential':
            self.radiation = Exponential_Radiation
        elif mode == 'Logistic':
            # print("Logistic Radiation Function")
            self.radiation = Logistic_Radiation
        elif mode == 'Hyperbolic':
            self.radiation = Hyperbolic_Radiation
    
    def radiate(self, x):
        return self.radiation(x)


