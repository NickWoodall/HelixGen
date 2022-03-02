#Rotation Methods

import numpy as np
from math import cos,sin,tan,asin,acos,radians,sqrt,degrees,atan,copysign,atan2
import math

def euler_to_R(phi,theta,psi):
    
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(phi), -np.sin(phi) ],
                    [0,         np.sin(phi), np.cos(phi)  ]
                    ])

    R_y = np.array([[np.cos(theta),    0,      np.sin(theta)  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta),   0,      np.cos(theta)  ]
                    ])

    R_z = np.array([[np.cos(psi),    -np.sin(psi),    0],
                    [np.sin(psi),    np.cos(psi),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))
    
    return R

def R_to_euler(R) :
    assert(isRotationMatrix(R))

    s_y = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = s_y < 1e-6

    if  not singular :
        phi = np.arctan2(R[2,1] , R[2,2])
        theta = np.arctan2(-R[2,0], s_y)
        psi = np.arctan2(R[1,0], R[0,0])
    else :
        phi = np.arctan2(-R[1,2], R[1,1])
        theta = np.arctan2(-R[2,0], s_y)
        psi = 0

    return np.array([phi, theta, psi])


def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm


def euler_to_quaternion(euler_angles):
    yaw = euler_angles[0]
    pitch = euler_angles[1]
    roll = euler_angles[2]
    
    cy = cos(yaw * 0.5)
    sy = sin(yaw * 0.5)
    cp = cos(pitch * 0.5)
    sp = sin(pitch * 0.5)
    cr = cos(roll * 0.5)
    sr = sin(roll * 0.5)
    
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    
    q = [w, x, y, z]
    
    return q

def quaternion_to_euler(q):
    #all glory to Isacc for this code
    if len(q) == 1:
        q = q[0]
    
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if (abs(sinp) >= 1):
        pitch = copysign(mPI / 2, sinp)
    else:
        pitch = asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = atan2(siny_cosp, cosy_cosp)

    euler_angles = [yaw, pitch, roll]
    
    return euler_angles

def quaternion_multiply(quaternion1, quaternion0):
    #all glory to Isacc for this code
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)


def quaternion_rel_rotation(q1,q2):
    # from q1 to q2
    #all glory to Isacc for this code
    
    if len(q1) == 1:
        q1 = q1[0]
    if len(q2) == 1:
        q2 = q2[0]
    
    inv_q1 = [q1[0], -q1[1], -q1[2], -q1[3]]
    return quaternion_multiply(inv_q1, q2)

#https://github.com/toji/gl-matrix/blob/f0583ef53e94bc7e78b78c8a24f09ed5e2f7a20c/src/gl-matrix/quat.js#L54
def vectors_to_quaternion(v1,v2):
    #careful might be broke
    eps = 1e-6
    u = normalize(v1)
    v = normalize(v2)
    
    q = np.array([0.0,0.0,0.0,0.0])
    
    dProd = np.dot(u,v)
    
    if dProd < -1+eps:
        #180deg rotation
        print('180')
        xUnit = np.array([1,0,0])
        yUnit = np.array([0,1,0])
        
        cProd = np.cross(xUnit,u)
        if normalize(cProd) < eps:
            cProd = np.cross(yUnit,u)
        cProd = normalize(cProd)
        q = setAxisAngle_quaternion(cProd,math.pi)
    elif dProd > 1-eps:
        #no Rot
        q = [0.0,0.0,0.0,1.0]
    else:
        cProd = np.cross(u,v)
        q[0] = cProd[0]
        q[1] = cProd[1]
        q[2] = cProd[2]
        q[3] = 1 + dProd
        
    return normalize(q)

    

#https://github.com/toji/gl-matrix/blob/f0583ef53e94bc7e78b78c8a24f09ed5e2f7a20c/src/gl-matrix/quat.js#L54
def setAxisAngle_quaternion(axis,angle):
    q = np.array([0.0,0.0,0.0,0.0])
    angle = angle*0.5
    s = sin(angle)
    q[0] = s *axis[0]
    q[1] = s * axis[1]
    q[2] = s * axis[2]
    q[3] = cos(angle)
    

    
    return q

def quaternion_to_rotationmatrix(qIn):
    #WARNING THIS PROBABLY WRONG, definition of q not the same
    q = normalize(qIn)
    
    sq0 = q[0]*q[0]
    sq1 = q[1]*q[1]
    sq2 = q[2]*q[2]
    sq3 = q[3]*q[3]
    
    r = np.zeros((3,3))
    
    r[0][0] = 1-2*sq2-2*sq3
    r[0][1] = 2*q[1]*q[2]+2*q[0]*q[3]
    r[0][2] = 2*q[1]*q[3]-2*q[0]*q[2]
    r[1][0] = 2*q[1]*q[2]-2*q[0]*q[3]
    r[1][1] = 1-2*sq1-2*sq3
    r[1][2] = 2*q[2]*q[3]+2*q[0]*q[1]
    r[2][0] = 2*q[1]*q[3]+2*q[0]*q[2]
    r[2][1] = 2*q[2]*q[3]-2*q[0]*q[1]
    r[2][2] = 1-2*sq1-2*sq2

    assert isRotationMatrix(r)
    return r
    

def zero_rot_xyz(hel,r):
    #all glory to Isacc for this code
    
    translate_x=hel[0]
    translate_y=hel[1]
    translate_z=hel[2]

    axis = np.array([[0,0,0],[0,0,0]])
    padded_axis = np.append(axis,[[1]]*axis.shape[0], axis = 1)
    R = r
    M = np.append(np.append(R,[[0]*3],axis=0),[[0]]*4,axis=1)

    M[3][3] = 1
    M[0][3] = translate_x
    M[1][3] = translate_y
    M[2][3] = translate_z
    transformed_axis = np.linalg.inv(M).dot(padded_axis.T).T
    xyz = np.delete(transformed_axis,3,axis=1)[0]

    return xyz

def dihedral_from_euler(p1,p2,eu1,eu2):

    zUnit = np.array([0,0,-1] ,dtype=np.float64)
    R1 = euler_to_R(eu1[0],eu1[1],eu1[2])
    R2 = euler_to_R(eu2[0],eu2[1],eu2[2])
    
    
    v1 = np.matmul(R1,zUnit)
    v2 = np.matmul(R2,zUnit)

    
    return dihedral_with_points(v1,v2,p1,p2)
    
    

def dihedral_with_points(v1,v2,p1,p2):
    """Calculates dihedral from helix vectors and midpoints, vectors travel from NtoC"""
    return dihedral(-v1,p2-p1,v2)

def dihedral(v1,v2,v3):
    """Calculates dihedral angle, -v1,v3 are helix vectors, v2 is vector between helix midpoints"""
    #https://math.stackexchange.com/questions/47059/how-do-i-calculate-a-dihedral-angle-given-cartesian-coordinates
    
    n1 = normalize(np.cross(v1,v2))
    n2 = normalize(np.cross(v2,v3))
    m1 = normalize(np.cross(n1,v2))
    x = np.dot(n1,n2)
    y = np.dot(m1,n2)
    angle = atan2(y,x)
    return angle
            
    
    return dListList

def angle_two_vectors(v1,v2):
    #assuming normalize
    #https://onlinemschool.com/math/library/vector/angl/

    # cos α = 	a·b
    #         |a|·|b|

    dp = np.dot(v1,v2)
    return  acos(dp)

def rotation(u,v):
    #a = np.array([0,0,1]) #Ryan I think made this a [0,0,-1] is normal reference? Ask Ryan
    #b = random_tilt_u(sigma)
    a = normalize(u)
    b = normalize(v)
    
    v = np.cross(a,b)
    s = np.linalg.norm(v) #?
    c = a.dot(b)
    I = np.identity(3)
    
    #vXStr = '{} {} {}; {} {} {}; {} {} {}'.format(0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0)
    #k = np.matrix(vXStr)
    
    k = np.array([[0 , -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    
    R = I + k + np.matmul(k,k) * ((1 -c)/(s**2))
    assert(isRotationMatrix(R))
    return R