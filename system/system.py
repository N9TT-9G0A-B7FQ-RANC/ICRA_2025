import numpy as np

def vanderpol(X, mu = 2):
    x, y = X[0], X[1]
    dx__dt = mu * (x - 1/3 * x**3 - y)
    dy__dt = 1/mu * x
    return [dx__dt, dy__dt]

def duffing_oscillator(X, f, gamma = 0.002, beta = 1, alpha = 1):
    x0, x1 = X[0], X[1]
    d_x0__dt = x1
    d_x1__dt = -gamma * x1 - alpha * x0 - beta * x0**3 + f
    return [d_x0__dt, d_x1__dt]

def get_slipping_angle(vx, vy, delta):
    return np.clip(delta - np.arctan(vy / vx), -1, 1)

def get_lateral_force(parameters, x, index):
    By = parameters[f'By{index}']
    Cy = parameters[f'Cy{index}']
    Dy = parameters[f'Dy{index}']
    Ey = parameters[f'Ey{index}']
    return Dy * np.sin(Cy * np.arctan(By * x - Ey * (By * x - np.arctan(By * x))))

def dof2(X, U, parameter):
    
    vy, psidt = X[0], X[1]
    vx, d1 = U[0], U[1]
    d1 = 2 * d1

    # known parameters
    m = parameter['m']
    g = parameter['g']
    lf = parameter['lf']
    lr = parameter['l'] - lf
    iz = parameter['iz']
    m_inv = 1 / m
    iz_inv = 1 / iz     

    # Project on tire frame
    vxp1 = vx * np.cos(d1) #+ (vy + lf * psidt) * np.sin(d1)
    vxp2 = vx #* ca.cos(d2) # + (vy - lr * psidt) * torch.sin(d2)

    vyp1 = (vy + lf * psidt) #* np.cos(d1) - vx * np.sin(d1) # + b
    vyp2 = (vy - lr * psidt) # - vx * torch.sin(d2) # + b

    # Compute lateral slip angles
    alpha1 = get_slipping_angle(vxp1, vyp1, d1)
    alpha2 = get_slipping_angle(vxp2, vyp2, 0)

    # Compute lateral tire forces
    fyp1 = get_lateral_force(parameter, alpha1, 1) #/ mf
    fyp2 = get_lateral_force(parameter, alpha2, 2) #/ mr

    # Project on carbody frame
    fy1 = fyp1 * np.cos(d1) 
    fy2 = fyp2 # * torch.cos(d2)

    vydt = m_inv * (fy1 + fy2) - vx * np.sin(d1) * psidt
    psidt2 = iz_inv * (lf * fy1 - lr * fy2)

    return [vydt, psidt2]

SimplifiedPacejka_parameters = {
    'Bx': 7.2,
    'Cx': 1.65, 
    'Dx': 3678,
    'Ex': -0.5,
    'By': 7.1,
    'Cy': 1.3,
    'Dy': 3678.75,
    'Ey': -1.0,
}

DOF7SimplifiedPacejka_parameters = {
    'vehicle':{
        'g' : 9.81,
        'm' : 1500,
        'ms' : 1500,
        'lf' : 2,
        'lr' : 2.5,
        'h' : 0.54,
        'L1': 2,
        'L2': 2,
        'r' : 0.395,
        'ix' : 1049,
        'iy' : 2613,
        'iz' : 10000,
        'ir' : 0.89,
        'ra' : 1.292,
        's' : 1.85,
        'cx' : 0.37,
    },
    'tire1': SimplifiedPacejka_parameters,
    'tire2': SimplifiedPacejka_parameters,
    'tire3': SimplifiedPacejka_parameters,
    'tire4': SimplifiedPacejka_parameters,
}

DOF10SimplifiedPacejka_parameters = {
    'vehicle':{
        'g' : 9.81,
        'm' : 1500,
        'ms' : 1500,
        'lf' : 2,
        'lr' : 2.5,
        'h' : 0.54,
        'L1': 2,
        'L2': 2,
        'r' : 0.395,
        'ix' : 1049,
        'iy' : 2613,
        'iz' : 10000,
        'ir' : 0.89,
        'ra' : 1.292,
        's' : 1.85,
        'cx' : 0.37,
    },
    'suspension1' : {
        'ks1': 50000,
        'ds1': 20000,
    },
    'suspension2' : {
        'ks2': 50000,
        'ds2': 20000,
    },
    'suspension3' : {
        'ks3': 50000,
        'ds3': 20000,
    },
    'suspension4' : {
        'ks4': 50000,
        'ds4': 20000,
    },
    'tire1': SimplifiedPacejka_parameters,
    'tire2': SimplifiedPacejka_parameters,
    'tire3': SimplifiedPacejka_parameters,
    'tire4': SimplifiedPacejka_parameters,
}

class SimplifiedPacejkaTireModel:
    
    def __init__(self, parameters):
        
        self.fz0 = parameters['fz0']

        # Lateral coefficients
        self.By = parameters['By']
        self.Cy = parameters['Cy']
        self.Dy = parameters['Dy']
        self.Ey = parameters['Ey']

        # Longitudinal coefficients
        self.Bx = parameters['Bx']
        self.Cx = parameters['Cx']
        self.Dx = parameters['Dx']
        self.Ex = parameters['Ex']

    def get_fx0(self, fz, sigma):
        dfz = fz / self.fz0
        return self.Dx * np.sin(self.Cx * np.arctan(self.Bx * sigma - self.Ex*(self.Bx * sigma - np.arctan(self.Bx * sigma)))) * dfz

    def get_fy0(self, fz, alpha):
        dfz = fz / self.fz0
        return self.Dy * np.sin(self.Cy * np.arctan(self.By * alpha - self.Ey*(self.By * alpha - np.arctan(self.By * alpha)))) * dfz
    

    
class DOF10SimplifiedPacejka:

    def __init__(self, parameters):
        
        vehicle_parameters = parameters['vehicle']
        self.g = vehicle_parameters['g']
        self.ms = vehicle_parameters['ms']
        self.m = vehicle_parameters['m']
        self.lf = vehicle_parameters['lf']
        self.lr = vehicle_parameters['lr']
        self.h = vehicle_parameters['h']
        self.l = self.lf + self.lr
        self.L1 = vehicle_parameters['L1']
        self.L2 = vehicle_parameters['L2']
        self.L = self.L1 + self.L2
        self.r = vehicle_parameters['r']
        self.ix = vehicle_parameters['ix']
        self.iy = vehicle_parameters['iy']
        self.iz = vehicle_parameters['iz'] 
        self.ir = vehicle_parameters['ir']
        self.ra = vehicle_parameters['ra']
        self.s = vehicle_parameters['s']
        self.cx = vehicle_parameters['cx']
        
        suspension1_parameters = parameters['suspension1']
        self.ks1 = suspension1_parameters['ks1']
        self.ds1 = suspension1_parameters['ds1']
        suspension2_parameters = parameters['suspension2']
        self.ks2 = suspension2_parameters['ks2']
        self.ds2 = suspension2_parameters['ds2']
        suspension3_parameters = parameters['suspension3']
        self.ks3 = suspension3_parameters['ks3']
        self.ds3 = suspension3_parameters['ds3']
        suspension4_parameters = parameters['suspension4']
        self.ks4 = suspension4_parameters['ks4']
        self.ds4 = suspension4_parameters['ds4']

        # Compute constants
        self.fz012 = (self.ms * self.lr * self.g) / (2 * self.l)
        self.fz034 = (self.ms * self.lf * self.g) / (2 * self.l)
        self.m_inv = 1 / self.m
        self.ms_inv = 1 / self.ms
        self.ix_inv = 1 / self.ix
        self.iy_inv = 1 / self.iy
        self.iz_inv = 1 / self.iz

        # Tires
        parameters['tire1']['fz0'] = self.fz012
        parameters['tire2']['fz0'] = self.fz012
        parameters['tire3']['fz0'] = self.fz034
        parameters['tire4']['fz0'] = self.fz034
        self.fl_tire_model = SimplifiedPacejkaTireModel(parameters['tire1'])
        self.fr_tire_model = SimplifiedPacejkaTireModel(parameters['tire2'])
        self.rl_tire_model = SimplifiedPacejkaTireModel(parameters['tire3'])
        self.rr_tire_model = SimplifiedPacejkaTireModel(parameters['tire4'])
            
    def get_slipping_rate(self, vxp, w, r):
        
        # Rolling phase
        if w * r == vxp:
            return 0.

        # Traction phase 
        elif w * r > vxp:
            if w == 0:
                return 0
            sigma = (r * w - vxp) / (r * np.abs(w))
            if sigma > 1:
                return 1
            else:
                return sigma

        # Braking phase
        elif w * r < vxp :
            if vxp == 0:
                return 0
            sigma = (r * w - vxp) / np.abs(vxp)
            if sigma < -1:
                return -1
            else:
                return sigma

    def get_slipping_rates(self, vxp1, w1, vxp2, w2, vxp3, w3, vxp4, w4, r):

        sigma1 = self.get_slipping_rate(vxp1, w1, r)
        sigma2 = self.get_slipping_rate(vxp2, w2, r)
        sigma3 = self.get_slipping_rate(vxp3, w3, r)
        sigma4 = self.get_slipping_rate(vxp4, w4, r)

        return sigma1, sigma2, sigma3, sigma4
    
    def get_wheel_speed_in_vehicle_frame(self, vx, vy, psidt, idx):
        if idx == 1:
            return vx - self.L1 * psidt, vy + self.lf * psidt
        elif idx == 2:
            return vx + self.L1 * psidt, vy + self.lf * psidt
        elif idx == 3:
            return vx - self.L2 * psidt, vy - self.lr * psidt
        elif idx == 4:
            return vx + self.L2 * psidt, vy - self.lr * psidt
        
    def get_slipping_angle(self, vx, vy, delta):
        if vx == 0:
            return 0.
        else:
            return np.clip(delta - np.arctan(vy / vx), -1, 1)
        
    def get_slipping_angles(self, vx1, vy1, vx2, vy2, vx3, vy3, vx4, vy4, delta1, delta2, delta3, delta4):

        alpha1 = self.get_slipping_angle(vx1, vy1, delta1)
        alpha2 = self.get_slipping_angle(vx2, vy2, delta2)
        alpha3 = self.get_slipping_angle(vx3, vy3, delta3)
        alpha4 = self.get_slipping_angle(vx4, vy4, delta4)

        return alpha1, alpha2, alpha3, alpha4
    
    def get_suspension_travel(self, theta, phi):

        dzg1 = self.L1 * np.sin(theta) - self.lf * np.cos(theta) * np.sin(phi)
        dzg2 = -self.L1 * np.sin(theta) - self.lf * np.cos(theta) * np.sin(phi)
        dzg3 = self.L2 * np.sin(theta) + self.lr * np.cos(theta) * np.sin(phi)
        dzg4 = -self.L2 * np.sin(theta) + self.lr * np.cos(theta) * np.sin(phi)

        return dzg1, dzg2, dzg3, dzg4
    
    def get_suspension_speed_travel(self, theta, thetadt, phi, phidt):

        dzg1dt = self.L1 * thetadt * np.cos(theta) + self.lf * thetadt * np.sin(theta) * np.sin(phi) - self.lf * phidt * np.cos(theta) * np.cos(phi)
        dzg2dt = -self.L1 * thetadt * np.cos(theta) + self.lf * thetadt * np.sin(theta) * np.sin(phi) - self.lf * phidt * np.cos(theta) * np.cos(phi)
        dzg3dt =  self.L2 * thetadt * np.cos(theta) - self.lr * thetadt * np.sin(theta) * np.sin(phi) + self.lr * phidt * np.cos(theta) * np.cos(phi)
        dzg4dt = -self.L2 * thetadt * np.cos(theta) - self.lr * thetadt * np.sin(theta) * np.sin(phi) + self.lr * phidt * np.cos(theta) * np.cos(phi)

        return dzg1dt, dzg2dt, dzg3dt, dzg4dt
    
    def get_vertical_normal_forces(self, dzg1, dzg2, dzg3, dzg4, dzg1dt, dzg2dt, dzg3dt, dzg4dt):

        fs1 = -self.ks1 * dzg1 - self.ds1 * dzg1dt
        fs2 = -self.ks2 * dzg2 - self.ds2 * dzg2dt
        fs3 = -self.ks3 * dzg3 - self.ds3 * dzg3dt
        fs4 = -self.ks4 * dzg4 - self.ds4 * dzg4dt

        fz1 = self.fz012 + fs1
        fz2 = self.fz012 + fs2
        fz3 = self.fz034 + fs3
        fz4 = self.fz034 + fs4

        return fz1, fz2, fz3, fz4, fs1, fs2, fs3, fs4
    
    def get_longitudinal_tire_forces(self, sigma1, alpha1, fz1, sigma2, alpha2, fz2, sigma3, alpha3, fz3, sigma4, alpha4, fz4):
        '''Return longitudinal tires forces in the tire frame'''

        # Compute longitudinal tire forces
        fx1 = self.fl_tire_model.get_fx0(fz1, sigma1)
        fx2 = self.fl_tire_model.get_fx0(fz2, sigma2)
        fx3 = self.fl_tire_model.get_fx0(fz3, sigma3)
        fx4 = self.fl_tire_model.get_fx0(fz4, sigma4)

        return fx1, fx2, fx3, fx4
    
    def get_lateral_tire_forces(self, sigma1, alpha1, fz1, sigma2, alpha2, fz2, sigma3, alpha3, fz3, sigma4, alpha4, fz4):
        '''Return lateral tire forces in the tire frame'''

        # Compute lateral tire forces
        fy1 = self.fl_tire_model.get_fy0(fz1, alpha1)
        fy2 = self.fl_tire_model.get_fy0(fz2, alpha2)
        fy3 = self.fl_tire_model.get_fy0(fz3, alpha3)
        fy4 = self.fl_tire_model.get_fy0(fz4, alpha4)

        return fy1, fy2, fy3, fy4
    
    def project_from_tire_to_carbody_frame(self, uxt, uyt, delta, phi, theta, fz):
        uxc = (uxt * np.cos(delta) - uyt * np.sin(delta)) * np.cos(phi) + fz * np.sin(phi)
        uyc = (uxt * np.cos(delta) - uyt * np.sin(delta)) * np.sin(theta) * np.sin(phi) + (uxt * np.sin(delta) + uyt * np.cos(delta)) * np.cos(theta) + fz * np.sin(theta) * np.cos(phi)
        return uxc, uyc
    
    def project_from_carbody_to_tire_frame(self, uxc, uyc, delta):
        uxt = uxc * np.cos(delta) + uyc * np.sin(delta)
        uyt = -uxc * np.sin(delta) + uyc * np.cos(delta) 
        return uxt, uyt
    
    def get_faero(self, vx):
        return 0.5 * self.ra * self.s * self.cx * vx**2
    
    def get_state_derivatives(self, x2, x4, x5, x6, x7, x8, x9, x10, x11, x12, u1, u2, u3, u4, 
                              fx1, fx2, fx3, fx4, 
                              fy1, fy2, fy3, fy4,
                              fz1, fz2, fz3, fz4,
                              fs1, fs2, fs3, fs4,
                              fxp1, fxp2, fxp3, fxp4,
                              faero, d1):

        fz1 = fz1 * np.cos(x9) * np.cos(x7)
        fz2 = fz2 * np.cos(x9) * np.cos(x7)
        fz3 = fz3 * np.cos(x9) * np.cos(x7)
        fz4 = fz4 * np.cos(x9) * np.cos(x7)

        self.x1dt = x2 * np.cos(x11) - x4 * np.sin(x11)
        self.x2dt = x12 * x4 - x10 * x6 + self.m_inv * (fx1 + fx2 + fx3 + fx4 - faero * np.cos(x9))
        self.x3dt = x2 * np.sin(x11) + x4 * np.cos(x11)
        self.x4dt = -x12 * x2 * np.sin(d1) + x8 * x6 + self.m_inv * (fy1 + fy2 + fy3 + fy4) 
        self.x5dt = x6
        self.x6dt = self.ms_inv * (fz1 + fz2 + fz3 + fz4) - self.g * np.cos(x7) * np.cos(x9)
        
        self.x7dt = x8
        self.x8dt = self.ix_inv * ( self.L1 * (fz1 + fz3 - fz2 - fz4) + x5 * (fy1 + fy2 + fy3 + fy4) )
        self.x9dt = x10
        self.x10dt = self.iy_inv * ( -self.lf * (fz1 + fz2 ) + self.lr * (fz3 + fz4) - x5 * (fx1 + fx2 + fx3 + fx4) )
        self.x11dt = x12
        self.x12dt = self.iz_inv * ( self.lf * (fy1 + fy2) - self.lr * (fy3 + fy4) + self.L1 * (fx2 + fx4 - fx1 - fx3) )
        
        self.x13dt = (u1 - self.r * fxp1) / (self.ir)
        self.x14dt = (u2 - self.r * fxp2) / (self.ir)
        self.x15dt = (u3 - self.r * fxp3) / (self.ir)
        self.x16dt = (u4 - self.r * fxp4) / (self.ir)
    
    def run(self, X, U):

        ### Control input correspondances
        # u1 = Tomega1
        # u2 = Tomega2
        # u3 = Tomega3
        # u4 = Tomega4
        # u5 = deltaf
        # u6 = deltar

        ### State correspondances
        # x1 = x
        # x2 = Vx
        # x3 = y
        # x4 = Vy
        # x5 = z
        # x6 = Vz
        # x7 = theta
        # x8 = thetadt
        # x9 = phi
        # x10 = phidt
        # x11 = psi
        # x12 = psidt
        # x13 = omega1
        # x14 = omega2
        # x15 = omega3
        # x16 = omega4

        u1, u2, u3, u4, u5, u6 = U

        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16 = X
        
        # Compute vxp for each wheels in wheel basis

        vxp1 = (x2 - self.L1 * x12) * np.cos(u5) + (x4 + self.lf * x12) * np.sin(u5)
        vxp2 = (x2 + self.L1 * x12) * np.cos(u5) + (x4 + self.lf * x12) * np.sin(u5)
        vxp3 = (x2 - self.L2 * x12) * np.cos(u6) + (x4 - self.lr * x12) * np.sin(u6)
        vxp4 = (x2 + self.L2 * x12) * np.cos(u6) + (x4 - self.lr * x12) * np.sin(u6)

        vyp1 = (x4 + self.lf * x12) * np.cos(u5) - (x2 - self.L1 * x12) * np.sin(u5)
        vyp2 = (x4 + self.lf * x12) * np.cos(u5) - (x2 + self.L1 * x12) * np.sin(u5)
        vyp3 = (x4 - self.lr * x12) * np.cos(u6) - (x2 - self.L2 * x12) * np.sin(u6)
        vyp4 = (x4 - self.lr * x12) * np.cos(u6) - (x2 + self.L2 * x12) * np.sin(u6)

        # Compute slipping rates and angles
        self.alpha1, self.alpha2, self.alpha3, self.alpha4 = self.get_slipping_angles(vxp1, vyp1, vxp2, vyp2, vxp3, vyp3, vxp4, vyp4, u5, u5, u6, u6)
        self.sigma1, self.sigma2, self.sigma3, self.sigma4 = self.get_slipping_rates(vxp1, x13, vxp2, x14, vxp3, x15, vxp4, x16, self.r)
        
        # Compute suspension and normal reaction forces
        dzg1, dzg2, dzg3, dzg4 = self.get_suspension_travel(theta = x7, phi = x9)
        
        dzg1dt, dzg2dt, dzg3dt, dzg4dt = self.get_suspension_speed_travel(theta = x7, thetadt = x8, phi = x9, phidt = x10)
        self.fz1, self.fz2, self.fz3, self.fz4, self.fs1, self.fs2, self.fs3, self.fs4 = self.get_vertical_normal_forces(dzg1, dzg2, dzg3, dzg4, dzg1dt, dzg2dt, dzg3dt, dzg4dt)
        
        # Compute tire forces in tire basis
        fxp1, fxp2, fxp3, fxp4 = self.get_longitudinal_tire_forces(self.sigma1, self.alpha1, self.fz1, self.sigma2, self.alpha2, self.fz2, self.sigma3, self.alpha3, self.fz3, self.sigma4, self.alpha4, self.fz4)
        fyp1, fyp2, fyp3, fyp4 = self.get_lateral_tire_forces(self.sigma1, self.alpha1, self.fz1, self.sigma2, self.alpha2, self.fz2, self.sigma3, self.alpha3, self.fz3, self.sigma4, self.alpha4, self.fz4)
        
        # Get tire forces in vehicle basis

        self.fx1, self.fy1 = self.project_from_tire_to_carbody_frame(fxp1, fyp1, u5, x9, x7, self.fz1)
        self.fx2, self.fy2 = self.project_from_tire_to_carbody_frame(fxp2, fyp2, u5, x9, x7, self.fz2)
        self.fx3, self.fy3 = self.project_from_tire_to_carbody_frame(fxp3, fyp3, u6, x9, x7, self.fz3)
        self.fx4, self.fy4 = self.project_from_tire_to_carbody_frame(fxp4, fyp4, u6, x9, x7, self.fz4)

        # TODO
        faero = self.get_faero(x2)

        self.get_state_derivatives(x2, x4, x5, x6, x7, x8, x9, x10, x11, x12, u1, u2, u3, u4, 
                            self.fx1, self.fx2, self.fx3, self.fx4, 
                            self.fy1, self.fy2, self.fy3, self.fy4,
                            self.fz1, self.fz2, self.fz3, self.fz4,
                            self.fs1, self.fs2, self.fs3, self.fs4,
                            fxp1, fxp2, fxp3, fxp4,
                            faero, u5)

        return [self.x1dt, self.x2dt, self.x3dt, self.x4dt, self.x5dt, self.x6dt, self.x7dt, self.x8dt, self.x9dt, self.x10dt, self.x11dt, self.x12dt, self.x13dt, self.x14dt, self.x15dt, self.x16dt]
                
class DOF7SimplifiedPacejka:

    def __init__(self, parameters):

        vehicle_parameters = parameters['vehicle']
        self.g = vehicle_parameters['g']
        self.m = vehicle_parameters['m']
        self.lf = vehicle_parameters['lf']
        self.lr = vehicle_parameters['lr']
        self.h = vehicle_parameters['h']
        self.l = self.lf + self.lr
        self.L1 = vehicle_parameters['L1']
        self.L2 = vehicle_parameters['L2']
        self.L = self.L1 + self.L2
        self.r = vehicle_parameters['r']
        self.iz = vehicle_parameters['iz'] 
        self.ir = vehicle_parameters['ir']
        self.ra = vehicle_parameters['ra']
        self.s = vehicle_parameters['s']
        self.cx = vehicle_parameters['cx']
        
        # Compute constants
        self.fz012 = (self.m * self.lr * self.g) / (2 * self.l)
        self.fz034 = (self.m * self.lf * self.g) / (2 * self.l)
        self.m_inv = 1 / self.m
        self.iz_inv = 1 / self.iz
        self.ir_inv = 1 / self.ir

        # Tires
        parameters['tire1']['fz0'] = self.fz012
        parameters['tire2']['fz0'] = self.fz012
        parameters['tire3']['fz0'] = self.fz034
        parameters['tire4']['fz0'] = self.fz034
        self.f1 = SimplifiedPacejkaTireModel(parameters['tire1'])
        self.f2 = SimplifiedPacejkaTireModel(parameters['tire2'])
        self.f3 = SimplifiedPacejkaTireModel(parameters['tire3'])
        self.f4 = SimplifiedPacejkaTireModel(parameters['tire4'])

    def set_parameters(self, parameters):
        vehicle_parameters = parameters['vehicle']
        self.g = vehicle_parameters['g']
        self.m = vehicle_parameters['m']
        self.lf = vehicle_parameters['lf']
        self.lr = vehicle_parameters['lr']
        self.h = vehicle_parameters['h']
        self.l = self.lf + self.lr
        self.L1 = vehicle_parameters['L1']
        self.L2 = vehicle_parameters['L2']
        self.L = self.L1 + self.L2
        self.r = vehicle_parameters['r']
        self.iz = vehicle_parameters['iz'] 
        self.ir = vehicle_parameters['ir']
        self.ra = vehicle_parameters['ra']
        self.s = vehicle_parameters['s']
        self.cx = vehicle_parameters['cx']
        
        # Compute constants
        self.fz012 = (self.m * self.lr * self.g) / (2 * self.l)
        self.fz034 = (self.m * self.lf * self.g) / (2 * self.l)
        self.m_inv = 1 / self.m
        self.iz_inv = 1 / self.iz
        self.ir_inv = 1 / self.ir

        # Tires
        parameters['tire1']['fz0'] = self.fz012
        parameters['tire2']['fz0'] = self.fz012
        parameters['tire3']['fz0'] = self.fz034
        parameters['tire4']['fz0'] = self.fz034
        self.f1 = SimplifiedPacejkaTireModel(parameters['tire1'])
        self.f2 = SimplifiedPacejkaTireModel(parameters['tire2'])
        self.f3 = SimplifiedPacejkaTireModel(parameters['tire3'])
        self.f4 = SimplifiedPacejkaTireModel(parameters['tire4'])
    
    def get_slipping_angle(self, vx, vy, delta):
        if vx == 0:
            return 0.
        else:
            return delta - np.arctan(vy / vx)
    
    def get_slipping_rate(self, vxp, w, r):
        
        # Rolling phase
        if w * r == vxp:
            return 0.

        # Traction phase 
        elif w * r > vxp:
            if vxp != 0 and w == 0:
                return 1
            sigma = (r * w - vxp) / (r * np.abs(w))
            if sigma > 1:
                return 1
            else:
                return sigma

        # Braking phase
        elif w * r < vxp :
            if vxp == 0 and w != 0:
                return -1
            sigma = (r * w - vxp) / np.abs(vxp)
            if sigma < -1:
                return -1
            else:
                return sigma
    
    def get_state_derivatives(self, vx, vy, fx1, fx2, fx3, fx4, fy1, fy2, fy3, fy4, fxp1, fxp2, fxp3, fxp4, psi, psidt, t1, t2, t3, t4, faero, d1):

        self.xdt = vx * np.cos(psi) - vy * np.sin(psi)
        self.vxdt = vy * psidt  + self.m_inv * (fx1 + fx2 + fx3 + fx4 - faero) 
        self.ydt = vx * np.sin(psi) + vy * np.cos(psi)
        self.vydt =  self.m_inv * (fy1 + fy2 + fy3 + fy4) - vx * np.sin(d1) * psidt
        
        self.psidt = psidt
        self.psidt2 = self.iz_inv * (self.lf * (fy1 + fy2) - self.lr * (fy3 + fy4))

        self.omega1dt = self.ir_inv * (t1 - fxp1 * self.r)
        self.omega2dt = self.ir_inv * (t2 - fxp2 * self.r)
        self.omega3dt = self.ir_inv * (t3 - fxp3 * self.r)
        self.omega4dt = self.ir_inv * (t4 - fxp4 * self.r)

    def get_faero(self, vx):
        return 0.5 * self.ra * self.s * self.cx * vx**2

    def run(self, X, U):

        x, vx, y, vy, psi, psidt, w1, w2, w3, w4 = X

        t1, t2, t3, t4, d1, d2 = U

        # Project on tire frame
        vxp1 = (vx - self.L1 * psidt) * np.cos(d1) + (vy + self.lf * psidt) * np.sin(d1)
        vxp2 = (vx + self.L1 * psidt) * np.cos(d1) + (vy + self.lf * psidt) * np.sin(d1)
        vxp3 = (vx - self.L2 * psidt) * np.cos(d2) + (vy - self.lr * psidt) * np.sin(d2)
        vxp4 = (vx + self.L2 * psidt) * np.cos(d2) + (vy - self.lr * psidt) * np.sin(d2)

        vyp1 = (vy + self.lf * psidt) * np.cos(d1) - (vx - self.L1 * psidt) * np.sin(d1)
        vyp2 = (vy + self.lf * psidt) * np.cos(d1) - (vx + self.L1 * psidt) * np.sin(d1)
        vyp3 = (vy - self.lr * psidt) * np.cos(d2) - (vx - self.L2 * psidt) * np.sin(d2)
        vyp4 = (vy - self.lr * psidt) * np.cos(d2) - (vx + self.L2 * psidt) * np.sin(d2)

        # Compute lateral slip angles
        self.alpha1 = self.get_slipping_angle(vxp1, vyp1, d1)
        self.alpha2 = self.get_slipping_angle(vxp2, vyp2, d1)
        self.alpha3 = self.get_slipping_angle(vxp3, vyp3, d2)
        self.alpha4 = self.get_slipping_angle(vxp4, vyp4, d2)

        # Compute longitudinal slip
        self.sigma1 = self.get_slipping_rate(vxp1, w1, self.r)
        self.sigma2 = self.get_slipping_rate(vxp2, w2, self.r)
        self.sigma3 = self.get_slipping_rate(vxp3, w3, self.r)
        self.sigma4 = self.get_slipping_rate(vxp4, w4, self.r)
        
        # Compute lateral tire forces
        fyp1 = self.f1.get_fy0(self.fz012, self.alpha1)
        fyp2 = self.f2.get_fy0(self.fz012, self.alpha2)
        fyp3 = self.f3.get_fy0(self.fz034, self.alpha3)
        fyp4 = self.f4.get_fy0(self.fz034, self.alpha4)

        fxp1 = self.f1.get_fx0(self.fz012, self.sigma1)
        fxp2 = self.f2.get_fx0(self.fz012, self.sigma2)
        fxp3 = self.f3.get_fx0(self.fz034, self.sigma3)
        fxp4 = self.f4.get_fx0(self.fz034, self.sigma4)

        # Project on carbody frame
        self.fx1 = fxp1 * np.cos(d1) - fyp1 * np.sin(d1)
        self.fx2 = fxp2 * np.cos(d1) - fyp2 * np.sin(d1)
        self.fx3 = fxp3 * np.cos(d2) - fyp3 * np.sin(d2)
        self.fx4 = fxp4 * np.cos(d2) - fyp4 * np.sin(d2)

        self.fy1 = fyp1 * np.cos(d1) + fxp1 * np.sin(d1)
        self.fy2 = fyp2 * np.cos(d1) + fxp2 * np.sin(d1)
        self.fy3 = fyp3 * np.cos(d2) + fxp3 * np.sin(d2)
        self.fy4 = fyp4 * np.cos(d2) + fxp4 * np.sin(d2)

        faero = self.get_faero(vx)

        self.get_state_derivatives(vx, vy, self.fx1, self.fx2, self.fx3, self.fx4, self.fy1, self.fy2, self.fy3, self.fy4, fxp1, fxp2, fxp3, fxp4, psi, psidt, t1, t2, t3, t4, faero, d1)

        return [self.xdt, self.vxdt, self.ydt, self.vydt, self.psidt, self.psidt2, self.omega1dt, self.omega2dt, self.omega3dt, self.omega4dt]
