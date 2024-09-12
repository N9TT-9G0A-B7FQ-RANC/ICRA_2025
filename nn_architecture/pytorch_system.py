import torch
import numpy as np

class vanderpol(torch.nn.Module):

    def __init__(self):
        
        super(vanderpol, self).__init__()
        self.mu = 2.
        self.state_variables = ['x', 'y']
    
    def forward(self, X, U):
        x, y = torch.split(X, split_size_or_sections=1, dim=1)
        xdt = self.mu * (x - 1/3 * x**3 - y)
        ydt = 1/self.mu * x
        return torch.concat((xdt, ydt), dim=1)
            
class duffing_oscillator(torch.nn.Module):

    def __init__(self):

        super(duffing_oscillator, self).__init__()
        self.gamma = 0.002
        self.omega = 1
        self.beta = 1
        self.alpha = 1
        self.delta = 0.3
        self.state_variables = ['x', 'y']
     
    def forward(self, X, U):
        x0, x1  = torch.split(X, split_size_or_sections=1, dim=1)
        f = U[:, 0]
        d_x0__dt = x1
        d_x1__dt = -self.gamma * x1 - self.alpha * x0 - self.beta * x0**3 + f
        return torch.concat((d_x0__dt, d_x1__dt), dim=1)
    
class dof2(torch.nn.Module):

    def __init__(self):
        super(dof2, self).__init__()
        self.parameter = {
                'g':9.81,
                'm':1500,
                'lf':2,
                'l':4.5,
                'iz':10000,
                'By1': 7.1,
                'Cy1': 1.3,
                'Dy1': 3678.75,
                'Ey1': -1.0,
                'By2': 7.1,
                'Cy2': 1.3,
                'Dy2': 3678.75,
                'Ey2': -1.0,
            }
            
       
        self.state_variables = ['vy', 'psidt']

    def get_lateral_force(self, x, index):
        By = self.parameter[f'By{index}']
        Cy = self.parameter[f'Cy{index}']
        Dy = self.parameter[f'Dy{index}']
        Ey = self.parameter[f'Ey{index}']
        return Dy * torch.sin(Cy * torch.arctan(By * x - Ey * (By * x - torch.arctan(By * x))))
    
    def get_slipping_angle(self, vx, vy, delta):
        v = torch.clip(delta - torch.arctan(vy / vx), -1, 1)
        return v

    def forward(self, X, U):

        vy, psidt = torch.split(X, split_size_or_sections=1, dim=1)
        vx, d1 = torch.split(U[:, 0], split_size_or_sections=1, dim=1)
        d1 = d1 * 2
        # d2 = torch.zeros(d1.shape).to(X.device)
        m = self.parameter['m']
        lf = self.parameter['lf']
        lr = self.parameter['l'] - lf
        iz = self.parameter['iz']
        self.m_inv = 1 / m
        self.iz_inv = 1 / iz


        # Project on tire frame
        vxp1 = vx * torch.cos(d1) #+ (vy + lf * psidt) * torch.sin(d1)
        vxp2 = vx #* torch.cos(d2) # + (vy - lr * psidt) * torch.sin(d2)
       
        vyp1 = (vy + lf * psidt) #* torch.cos(d1) - vx * torch.sin(d1) #
        vyp2 = (vy - lr * psidt) #* torch.cos(d2) # - vx * torch.sin(d2) 

        # Compute lateral slip angles
        alpha1 = self.get_slipping_angle(vxp1, vyp1, d1)
        alpha2 = self.get_slipping_angle(vxp2, vyp2, 0) 

        # Compute lateral tire forces
        fyp1 = self.get_lateral_force(alpha1, 1)
        fyp2 = self.get_lateral_force(alpha2, 2)        

        # Project on carbody frame
        self.fy1 = fyp1 * torch.cos(d1) 
        self.fy2 = fyp2 # * torch.cos(d2)

        vydt = self.m_inv * (self.fy1 + self.fy2) - vx * torch.sin(d1) * psidt
        psidt2 = self.iz_inv * (lf * self.fy1 - lr * self.fy2)
        return torch.concat((vydt, psidt2), dim=1)
      