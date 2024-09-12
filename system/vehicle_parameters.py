
import numpy as np

def control(time, t0, t1, t2, t3, a):
    out = np.zeros(time.shape)
    coeff1 = 1 / (t1 - t0)
    coeff2 = 1 / (t3 - t2)
    for i, t in enumerate(time):
        if t < t0:
            out[i] = 0
        elif t >= t0 and t < t1:
            out[i] = (t - t0) * coeff1 * a
        elif t >= t1 and t < t2:
            out[i] = a
        elif t >= t2 and t < t3:
            out[i] = a - (t - t2) * coeff2 * a
        elif t >= t3:
            out[i] = 0
        
    return out

h = 0.54
dt = 0.001
trajectory_duration = 20
nb_sample = int(trajectory_duration / dt)
time = np.linspace(0, trajectory_duration, nb_sample+1)

vx = []
x = []
y = []
vy = []
z = []
vz = []
theta = []
thetadt = []
phi = []
phidt = []
psi = []
psidt = []
df = []
dr = []
tf = []
tr = []
omega1 = []
omega2 = []
omega3 = []
omega4 = []

np.random.seed(42)

for i in range(400):

    vx_initial = 5 + np.random.rand() * 20
    acceleration_control = [50 + np.random.rand() * 50] * len(time)

    x.append(0)
    vx.append(vx_initial)
    y.append(0)
    vy.append(0)
    z.append(h)
    vz.append(0)
    theta.append(0)
    thetadt.append(0)
    phi.append(0)
    phidt.append(0)
    psi.append(0)
    psidt.append(0)

    df.append(np.sin(2*np.pi*1/10*time) * 2 * (np.random.rand() - 0.5) * 5 * np.pi/180)
    dr.append(np.zeros(time.shape))
    tf.append(acceleration_control)
    tr.append(np.zeros(time.shape))

    omega1.append(vx_initial / 0.395)
    omega2.append(vx_initial / 0.395)
    omega3.append(vx_initial / 0.395)
    omega4.append(vx_initial / 0.395)

print()

# Control shape
# p0 = 0.5/trajectory_duration
# p1 = 2.5/trajectory_duration
# p2 = 6/trajectory_duration
# p3 = 2.5/trajectory_duration
# p4 = 0.5/trajectory_duration

# max_steering_angle = 10 * np.pi / 180
# max_torque = 50
# max_speed = 25
# min_speed = 5

# for i in range(200):

#     vx_initial = np.random.rand() * max_speed + min_speed
#     acceleration = np.random.rand() * 2 * max_torque - max_torque
#     steering_angle = np.random.rand() * 2 * max_steering_angle - max_steering_angle
#     s1 = np.random.rand() * 8/trajectory_duration
#     s2 = np.random.rand() * 8/trajectory_duration

#     t0 = (p0 + s1) * trajectory_duration
#     t1 = (p0 + p1 + s1) * trajectory_duration
#     t2 = (p0 + p1 + p2 + s1) * trajectory_duration
#     t3 = (p0 + p1 + p2 + p3 + s1) * trajectory_duration
#     steering_control = control(time, t0 = t0, t1 = t1, t2 = t2, t3 = t3, a = steering_angle * (min_speed/vx_initial)**2)

#     t0 = (p0 + s2) * trajectory_duration
#     t1 = (p0 + p1 + s2) * trajectory_duration
#     t2 = (p0 + p1 + p2 + s2) * trajectory_duration
#     t3 = (p0 + p1 + p2 + p3 + s2) * trajectory_duration
#     acceleration_control = control(time, t0 = t0, t1 = t1, t2 = t2, t3 = t3, a = acceleration)

#     x.append(0)
#     vx.append(vx_initial)
#     y.append(0)
#     vy.append(0)
#     z.append(h)
#     vz.append(0)
#     theta.append(0)
#     thetadt.append(0)
#     phi.append(0)
#     phidt.append(0)
#     psi.append(0)
#     psidt.append(0)

#     df.append(steering_control)
#     dr.append(np.zeros(time.shape))
#     tf.append(acceleration_control)
#     tr.append(np.zeros(time.shape))
#     omega1.append(vx_initial / 0.395)
#     omega2.append(vx_initial / 0.395)
#     omega3.append(vx_initial / 0.395)
#     omega4.append(vx_initial / 0.395)