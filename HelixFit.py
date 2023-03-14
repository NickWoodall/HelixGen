from math import cos,sin,tan,asin,acos,radians,sqrt,degrees,atan,atan2,copysign
from math import pi as mPI
import numpy as np
import io
import time
import math
import sys
import util.npose_util as nu

import util.RotationMethods as rm
from pymol import cmd, stored, selector
import sys

import numpy as np
from lmfit import Parameters,minimize,report_fit # fot fitting helix
import os
import copy

import warnings

import argparse
import textwrap

zero_ih = nu.npose_from_file('util/zero_ih.pdb')
tt = zero_ih.reshape(int(len(zero_ih)/5),5,4)
stub = tt[7:10].reshape(15,4)

class HelixParameters():
    """Stores Helical and Axis Parameters Describing a Single Straight Helix

    Straight Helices 
    
    ----Parameters ----
    Helical Twist - phi1: radians , Straight Helix = ~100 degrees (delta omega1)
    Helix Length - in residues
    Axis of Superhelical radius - translate_x, translate_y, translate_z (Helix Midpoint)
                                - rotate_phi, rotate_theta, rotate_psi (Euler angles rotation from vector [0,0,-1]
    d = 1.51 - Fixed, distance between successive residues along the helical axis, [angstrom] BGS=z1
    r1 = 2.26 - FIXED, helical radius, [angstrom] -- BundleGridSampler (BGS)=r1_peratom  
    
    Helix Endpoints
    'A little outside the realm of this class, (endpoints not stored here): Helices can be reconstructed from Endpoints'
    -stored in ParametericFit Object under fit. Can be taken from HelicalProtein Class via get_end_points
    ---Parameters
    x1,y1,z1 and x2,y2,z2
    
    
    Constructors - 
    Blank - Set values yourself or from Fit
    FromDict - Reads in values from a dictionary - Currently doing straight helices only
    FromEndpoints - Calculates midpoints and euler angles from helix endpoints return Hparam Object Straight Only
    
    Also contains the methods for the rounds of fitting to lmfit. Maybe better stored elsewhere
    Also contains some rotation matrix/euler methods
    
    
    """
    
    def __init__(self, omega1=None, phi1=None, helix_length=None, translate_x=None, translate_y=None,
                 translate_z=None, rotate_phi=None, rotate_theta=None, rotate_psi=None, rmsd=100):
        
        """Stores Helical and Axis Parameters"""
        
        #Helical Parameters
        self._length, self._d, self._r1, self._omega1, self._phi1 = (None,)*5
        
        #Axis Parameters
        self._translate_x, self._translate_y, self._translate_z  = (None,)*3
        self._rotate_phi, self._rotate_theta,self._rotate_psi = (None,)*3
        #euler methods of referencing rotations are nice for matching with lmfit
        
        self.omega1(omega1)
        self.phi1(phi1)
        self.length(helix_length)
        self.rmsd(rmsd)
        self._d = 1.51 # FIXED, distance between successive residues along the helical axis, [angstrom] 
        #-- BundleGridSampler=z1
        self._r1 = 2.26 # FIXED, helical radius, [angstrom] -- BundleGridSampler=r1_peratom        
        
        
        self.translate_x(translate_x)
        self.translate_y(translate_y)
        self.translate_z(translate_z)
        self.rotate_phi(rotate_phi)
        self.rotate_theta(rotate_theta)
        self.rotate_psi(rotate_psi)
            
    
    
    @classmethod
    def fromEndPoints(cls,p1,p2,phi1=0):
        """Take in xyz numpy arrays of helix endpoints. Estimates parameters for the helix"""
        vector = rm.normalize(p1-p2)
        
        zUnit = np.array([0,0,1] ,dtype=np.float64)
        R = rm.rotation(zUnit,vector)
        eas = rm.R_to_euler(R)
        
        
        length = int(np.linalg.norm(p2-p1)/1.51)
        halfLen = int(length/2)
        #No longer need since addition is on p2 side?
#         if length % 2 != 0:
#             pHalf = halfLen + 1
#         else:
#             pHalf = halfLen
        
        
        midPoint = p1-vector*halfLen*1.51
        
        
        #Regenerate
        hpObj = cls()
        
        hpObj.translate_x(midPoint[0])
        hpObj.translate_y(midPoint[1])
        hpObj.translate_z(midPoint[2])
        hpObj.rotate_phi(eas[0])
        hpObj.rotate_theta(eas[1])
        hpObj.rotate_psi(eas[2])
        
        #define straight helix(omega1)
        hpObj.omega1(radians(100))
        #set random phi1 since rotation lost in endpoint representation
        hpObj.phi1(phi1)
        hpObj.length(length)
        
        return hpObj

    
  
    @classmethod
    def fromDict(cls,dicIn):
        """Initializes from Dictionary of parameters"""

        hpObj = cls()
        hpObj.translate_x(dicIn['translate_x'])
        hpObj.translate_y(dicIn['translate_y'])
        hpObj.translate_z(dicIn['translate_z'])
        hpObj.rotate_phi(dicIn['rotate_phi'])
        hpObj.rotate_theta(dicIn['rotate_theta'])
        hpObj.rotate_psi(dicIn['rotate_psi'])
        hpObj.phi1(dicIn['phi1'])
        hpObj.length(dicIn['length'])
        
        
        return hpObj
    
    def __repr__(self):
        return f'Helical Parameters: omega1={self.omega1()}, phi1={self.phi1()},d={self.d()},r1={self.r1()},'                f'translate_x={self.translate_x()},translate_y={self.translate_y()},translate_z={self.translate_z()},'                f'rotate_phi={self.rotate_phi()},rotate_theta={self.rotate_theta()},rotate_psi={self.rotate_psi()},'                f'length={self.length()},rmsd={self.rmsd()}'
    
    #hybrid setters/getters. They will always return the value you ask for, but if you pass 
    #something to the function, it'll update the stored value and then return the (updated) value
    def omega1(self,omega1:float=None):    
        if omega1 is not None:
            self._omega1 = omega1
        return self._omega1
    
    def phi1(self,phi1:float=None):
        if phi1 is not None:
            self._phi1 = phi1
        return self._phi1
    
    def length(self,length:int=None):
        if length is not None:
            self._length = length
        return self._length
    
    def d(self,d:float=None):
        if d is not None:
            warnings.warn("HelixParamaters variable d is a physical constraint should not be touched")
            self._d = d
        return self._d
    
    def r1(self,r1:float=None):
        if r1 is not None:
            warnings.warn("HelixParamaters variable r1 is a physical constraint should not be touched")
            self._r1 = r1
        return self._r1
    
    #new axis independent code
    def translate_x(self,x:float=None):
        if x is not None:
            self._translate_x = x
        #print(f"self._translate_x: {self._translate_x}")
        return self._translate_x
    
    def translate_y(self,y:float=None):
        if y is not None:
            self._translate_y = y
        #print(f"self._translate_y: {self._translate_y}")
        return self._translate_y
    
    def translate_z(self,z:float=None):
        if z is not None:
            self._translate_z = z
        #print(f"self._translate_z: {self._translate_z}")
        return self._translate_z
    
    def rotate_phi(self,phi:float=None):
        if phi is not None:
            self._rotate_phi = phi
        #print(f"self._rotate_phi: {self._rotate_phi}")
        return self._rotate_phi
    
    def rotate_theta(self,theta:float=None):
        if theta is not None:
            self._rotate_theta = theta
        #print(f"self._rotate_theta: {self._rotate_theta}")
        return self._rotate_theta
    
    def rotate_psi(self,psi:float=None):
        if psi is not None:
            self._rotate_psi = psi
        #print(f"self._rotate_psi: {self._rotate_psi}")
        return self._rotate_psi
    
    def rmsd(self,rmsd:float=None):
        if rmsd is not None:
            self._rmsd = rmsd
        return self._rmsd
    
    def transformation_matrix(self):
        #moves 
            
        R = rm.euler_to_R(self.rotate_phi(),self.rotate_theta(),self.rotate_psi())
        #print(R)
        #R.resize(4,4)
        M = np.append(np.append(R,[[0]*3],axis=0),[[0]]*4,axis=1)
        #print(M)
        M[3][3] = 1
        M[0][3] = self.translate_x()
        M[1][3] = self.translate_y()
        M[2][3] = self.translate_z()
        #print(M)
        return M
    

    
    def to_lmfit_parameters(self, round_num=None):
        """Tells Parametric Protein what parameters to fit and the order if rounds exist"""
        
       #Straight helix guess
        params = Parameters()
        params.add('omega1', value=self.omega1(), vary=False)
        params.add('phi1', value=self.phi1(),vary=True)

        params.add('omega1', value=self.omega1(), vary=False)

        #axis independent code:
        params.add('translate_x', value=self.translate_x(),vary=True)
        params.add('translate_y', value=self.translate_y(),vary=True)
        params.add('translate_z', value=self.translate_z(),vary=True)

        #use helix fitter line to determine axis
        params.add('rotate_phi', value=self.rotate_phi(),min=-np.pi,max=np.pi,vary=False)
        params.add('rotate_theta', value=self.rotate_theta(),min=-np.pi,max=np.pi,vary=False)
        params.add('rotate_psi',value=self.rotate_psi(),min=-np.pi,max=np.pi,vary=False)

        return params
    
    def from_lmfit(self, fit):
        self.omega1(fit.valuesdict()['omega1'])
        self.phi1(fit.valuesdict()['phi1'])
        
        self.translate_x(fit.valuesdict()['translate_x'])
        self.translate_y(fit.valuesdict()['translate_y'])
        self.translate_z(fit.valuesdict()['translate_z'])
        self.rotate_phi(fit.valuesdict()['rotate_phi'])
        self.rotate_theta(fit.valuesdict()['rotate_theta'])
        self.rotate_psi(fit.valuesdict()['rotate_psi'])
        
       
    def get_dict(self):
        d = {
            'omega1':self.omega1(),
            'phi1':self.phi1(),
            'd':self.d(),
            'r1':self.r1(),
            'translate_x':self.translate_x(),
            'translate_y':self.translate_y(),
            'translate_z':self.translate_z(),
            'rotate_phi':self.rotate_phi(),
            'rotate_theta':self.rotate_theta(),
            'rotate_psi':self.rotate_psi(),
            'length':self.length(),
            'rmsd':self.rmsd()
        }
        
        return d
    
    def get_dict_set(self):
        d = {
            'omega1':self.omega1,
            'phi1':self.phi1,
            'd':self.d,
            'r1':self.r1,
            'translate_x':self.translate_x,
            'translate_y':self.translate_y,
            'translate_z':self.translate_z,
            'rotate_phi':self.rotate_phi,
            'rotate_theta':self.rotate_theta,
            'rotate_psi':self.rotate_psi,
            'length':self.length,
            'rmsd':self.rmsd
        }
        
        return d
    
    def copy(self):
        """Broke this need to fix"""
        return HelixParameters(omega1=self.omega1(), phi1=self.phi1(),helix_length=self.length(), 
                               translate_x=self.translate_x(), translate_y=self.translate_y(), 
                               translate_z=self.translate_z(), rotate_phi=self.rotate_phi(), 
                               rotate_theta=self.rotate_theta(), rotate_psi=self.rotate_psi(), 
                               rmsd=self.rmsd())
    
    def convert_to_endpoints(self):
        """Calculates Helical Endpoints Based on Helical Parameters"""
        
        endPointList = []
        zUnit  = np.array([0,0,-1])
        
        R = rm.euler_to_R(self.rotate_phi(),self.rotate_theta(),self.rotate_psi())
        vec = rm.normalize(np.matmul(R,zUnit))
        
        length = self.length()

        risePerRes = 1.51
        revVec = -vec
        multLen = int(length/2)
        revMultLen = multLen

        if length % 2 != 0:
            multLen += 1

        multLen = multLen*risePerRes
        revMultLen = revMultLen*risePerRes
        
        midpoint = [self.translate_x(),self.translate_y(), self.translate_z()]

        p1 = midpoint - revMultLen*vec
        p2 = midpoint + multLen*vec
        endPointList.append(p1)
        endPointList.append(p2)
        
        return endPointList
    
    def to_npose(self):
        """Uses a reference helix and rotates it into position based on Helical Parameters"""

        zUnit  = np.array([0,0,1])

        R = rm.euler_to_R(self.rotate_phi(),self.rotate_theta(),self.rotate_psi())
        vector = rm.normalize(np.matmul(R,zUnit))
        axisRot = rm.normalize(np.cross(vector,zUnit))
        ang = rm.angle_two_vectors(vector, zUnit)

        aRot=np.hstack((axisRot,[1]))
        
        #reference helix , #global
        len_zero_ih = int(len(zero_ih)/5)

        hLen = int((len_zero_ih-self.length())/2)
        xform1 = nu.xform_from_axis_angle_rad(aRot,-ang)
        xform1[0][3] = self.translate_x() 
        xform1[1][3] = self.translate_y() 
        xform1[2][3] = self.translate_z()

        #reversing here #key step to fix
        if self.length() % 2 == 1:
            self.npose = nu.xform_npose(xform1, zero_ih[(hLen*5):(-hLen*5)] )
        else:
            self.npose = nu.xform_npose(xform1, zero_ih)
            self.npose = self.npose[((hLen)*5):(-(hLen+1)*5)]
            
            
        xform2 = nu.xform_from_axis_angle_rad(vector,self.phi1())
        self.npose[:,0] = self.npose[:,0] -  self.translate_x() 
        self.npose[:,1] = self.npose[:,1] -  self.translate_y() 
        self.npose[:,2] = self.npose[:,2] -  self.translate_z()    
        
        self.npose = nu.xform_npose(xform2, self.npose )
        
        
        self.npose[:,0] = self.npose[:,0] +  self.translate_x() 
        self.npose[:,1]= self.npose[:,1] +  self.translate_y() 
        self.npose[:,2] = self.npose[:,2] +  self.translate_z()
        
        
        

        return self.npose
    
    
    
    def export_helix(self,labels=False,decPoint = 2):
        """Returns parameters comma seperated: Mostly for Helical Protein"""
        outString = ""
        label = ""
        #Dictionary of parameters
        pDict = self.get_dict()
        
        
        
        for key,val in pDict.items():
            label += key + ","
            if val is None:
                outString += f'{0:.{decPoint}f}' + ","
            else:
                outString += f'{val:.{decPoint}f}' + ","
            
            
        if labels:
            return label[:-1], outString[:-1] #removes comma at end
        else:
            return outString[:-1]


class ParametricHelix():
    """Fits a helical pose to Parameters, Builds a Single Helix Parameters to a Pose. 
    
    Fit Methods do functionality via lmfit - talks with HelicalParameters do get how to do it
    Fit methods chosen via booleans 
    
    Fit Methods: -fits based on rmsd to Calpha positions, no sidechain data
    
    fit_target_helix - supercoiled fit
    fit_str_helix_axis - straight helix (reduced parameters) or axis_only (just helix endpoints - slightly out of scope)
    fit_helical_center_line - called by fit_str_helix_axis to do fit

    Building Pose - 
    pose() function - calls
    pose_from_ca() - onces the Calpha atoms are retrieved from parameters uses a stub file from egp.py to build residue
    Ca_coords_from_stored_params() - generates Calpha coordinates from Parameters calls
    moving() - converts from 0 index residue at center to 0 index at N-term
    cartesian() - given indices from moving function generates xyz based on parameters - straight helix has own method
    
    
    """
    def __init__(self,name:str=None):
        self.helix_parameters = HelixParameters() #stores helical parameters
        self._name = None
        self.name(name)
    def name(self,name:str=None):
        if name is not None:
            self._name = name
        return self._name
    
    def is_named(self):
        return self._name is not None
    
    def get_helix_parameters(self): 
        return self.helix_parameters.copy()#I think I broke this

    #### FUNCTIONS ####


    
    # Parametric helix equation based on Huang et al. equations as described in SI 
    # Contains small patch to make it compatible with BundleGridSampler mover (due to differences in origin definition)
    #----------MAKE CARTESIAN COORDINATES FOR CA OF RESIDUE t---------------
    
    def straightHelix_Cartesian(self,t):
        #equaiton to generates a straight helix
        d = self.helix_parameters.d() # helical rise per residue (fixed)1.51ang
        r1 = self.helix_parameters.r1() #helical radius (fixed) 2.26ang
        omega1 = self.helix_parameters.omega1() # helical twist (fixed) 100deg
        phi1 = self.helix_parameters.phi1() #helical phase varies
         
        t = -t # the orientation of the reference helix zero_ih.pdb is inverted so invert here
        x = r1*cos(omega1*t+phi1)
        y = r1*sin(omega1*t+phi1)
        z = d*(t) #reversed

        return [x,y,z]
    

    #---------MAKE ARRAY OF XYZ COORDINATES FOR ALL CA-----------------------
    #residue_indices would be given like Huang's convention but 0 indexed (0 = first res). Going under the first res means negative numbers
    def moving(self,residue_indices=None,build=False):
        
        
        if residue_indices is None:
            # PATCH TO BRIDGE DIFFERENCES IN HOW THE 'ORIGIN' IS DEFINED
            delta_t=int(self.helix_parameters.length()/2)# define an offset of half-helix length (in number of residues) -- BundleGridSampler=delta_t
            # 're-number' indices +/- around middle of helix
            # to patch Vikram's convention (start from middle of helix) and Huang's convention (start at resid 1) 
            
            # Correct for helices that have odd numbers of residues (otherwise fitting helix will be one residue short)
            if (self.helix_parameters.length() % 2) == 0:
                residue_renumber_indices=np.arange(-delta_t,+delta_t,1)
            if (self.helix_parameters.length() % 2) != 0:
                residue_renumber_indices=np.arange(-delta_t,+delta_t+1,1)
        
        else:
            residue_renumber_indices = residue_indices

        moving_coordinates = np.array([self.straightHelix_Cartesian  (t) for t in residue_renumber_indices])

        return moving_coordinates


    def Ca_coords_from_stored_params(self,residue_indices=None):
        
        move = self.moving(residue_indices)
            
        padded_move = np.append(move,[[1]]*move.shape[0], axis = 1)
        transformed = self.helix_parameters.transformation_matrix().dot(padded_move.T).T
        striped_transformed = np.delete(transformed,3,axis=1)
        
        axis = np.array([[0,0,-10],[0,0,10]]) #reversed
        padded_axis = np.append(axis,[[1]]*axis.shape[0], axis = 1)
        transformed_axis = self.helix_parameters.transformation_matrix().dot(padded_axis.T).T
        striped_axis = np.delete(transformed_axis,3,axis=1)

        return striped_transformed, striped_axis
    
#-----------Objective RMSD function used during minimization---------------    
    def rmsd_array(self,params, target, dummy):

        self.helix_parameters.from_lmfit(params)
        

        striped_transformed,striped_axis = self.Ca_coords_from_stored_params()
        
        subtract_coord=striped_transformed-target

        rmsd_array=np.sqrt(np.sum(np.power(subtract_coord,2),axis=1))
        rmsd=np.sqrt((1/self.helix_parameters.length())*np.sum(np.sum(np.power(subtract_coord,2),axis=1)))
        self.helix_parameters.rmsd(np.sum(rmsd_array)/self.helix_parameters.length())
        
        
        return rmsd_array
    
   
        
    def fit_helical_center_line(self, target_helix, methodIn='least_sq',write_axis=False):
        """Find central helical axis from Calpha atoms.

        This function works by minimizing the distance from each Calpha to be 2.26 (ideal radius)
        to the central line (defined by point1/point2)
        """
        arrayIn = target_helix
        
        
        def perpDistance(helix_point, point1, point2):
            """Finds shortest distance between helix point and line defined by point1/point2"""
            S = point2 - point1
            V = point2 - helix_point
            W = np.cross(V,S)/np.linalg.norm(S)
            return np.linalg.norm(W)
        
        def avg_distance_line(params, arraypoints):
            """Takes an array of points and calculates distance to line defined by two points"""
            distanceList = []


            vals = params.valuesdict()
            point1x = vals['point1x']
            point1y = vals['point1y']
            point1z = vals['point1z']
            point2x = vals['point2x']
            point2y = vals['point2y']
            point2z = vals['point2z']

            point1 = np.array([point1x, point1y, point1z])
            #print(point1)
            point2 = np.array([point2x, point2y, point2z])
            #print(point2)

            #loss function, radius ideal helix is 2.26 (moves distance to 0)
            for x in arraypoints:
                distanceList.append(np.square(perpDistance(x, point1, point2)-2.26))
            
            rmsdKinda = (1/len(distanceList))*np.sqrt(np.sum(distanceList))

            return distanceList

        #initializes at helix endpoints
        #constrains parameters for line endpoints within one radius (r=2.3) rounded up
        fit_params = Parameters()
        fit_params.add('point1x', min=target_helix[0][0]-3, max=target_helix[0][0]+3, value=target_helix[0][0])
        fit_params.add('point1y', min=target_helix[0][1]-3, max=target_helix[0][1]+3, value=target_helix[0][1])
        fit_params.add('point1z', min=target_helix[0][2]-3, max=target_helix[0][2]+3, value=target_helix[0][2])
        fit_params.add('point2x', min=target_helix[-1][0]-3, max=target_helix[-1][0]+3, value=target_helix[-1][0])
        fit_params.add('point2y', min=target_helix[-1][1]-3, max=target_helix[-1][1]+3, value=target_helix[-1][1])
        fit_params.add('point2z', min=target_helix[-1][2]-3, max=target_helix[-1][2]+3, value=target_helix[-1][2])


        out = minimize(avg_distance_line, fit_params, args=(arrayIn,), method = methodIn)

        return out
    
     

    def fit_str_helix_axis(self, target_helix, method ='leastsq', write_axis=False, axis_only=False):
        """Fits a straight helix to a single helix pose, param = axis+phi1"""
    
        helix_length=len(target_helix)
        invert_guess = True
        
        # VARY, helical phase (around the internal axis of that helix), 
        #[degrees] -- BundleGridSampler=delta_omega1
        phi1_guess=radians(0) 
        omega1_guess=radians(+100.00) 
        
        
        #finds axis through helical center of target center
        fit_axis = self.fit_helical_center_line(target_helix)
        
        vals = fit_axis.params.valuesdict()
        p1 = np.array([vals['point1x'], vals['point1y'], vals['point1z']])
        p2 = np.array([vals['point2x'], vals['point2y'], vals['point2z']])
        
        #use fit to guess initial paramters
        helical_pseudo_origin = (p1+p2)/2
        helical_pseudo_axis = rm.normalize(p1-p2)
        
        
        
        #rotation and translation approximates from pseudo axis and origin
        rotate_phi_guess,rotate_theta_guess,rotate_psi_guess = rm.R_to_euler(rm.rotation([0,0,1],helical_pseudo_axis))
        translate_x_guess,translate_y_guess,translate_z_guess = helical_pseudo_origin
        
        orig_hp = HelixParameters(omega1_guess,phi1_guess,helix_length,translate_x_guess,translate_y_guess,
                                  translate_z_guess,rotate_phi_guess,rotate_theta_guess,rotate_psi_guess)
        
        self.helix_parameters=orig_hp

        # FIT
        params = self.helix_parameters.to_lmfit_parameters(round_num=4)
        fit=minimize(self.rmsd_array,params,method=method,args=(target_helix,True),**{"ftol":1.0e-3}) #"maxiter":1000 #,**{"ftol":1.e-3}
        self.helix_parameters.from_lmfit(fit.params)#retrieve last fit #not necessary I think

        self.fit = fit

        return self.fit
        
    

    def set_helix_parameters(self,params:HelixParameters):
        self.helix_parameters = params
        
    def get_fit(self):
        return self.fit
    
    def hparams_FromEndPoints(self):
        
        p1,p2 = self.fit.get_axis_end_points_()
        
        self.helix_parameters = hp.HelixParameters.fromEndPointsAxis(p1,p2)
        
    



class HelicalProtein():
    """Wraps Parametric Helix to make working with entire proteins easier.
       For fitting or Reconstruction of atom coordinates from Parameters
       riginally coded with Rosetta, recoded for speed using pymol, therefore some 
       awkard organization
   
   Fitting Helical Parameters Usage:
   
   Initialize  from __init__
   Extracts Helices using dssp definition and minimum helix length, Gets indices to use for fit
      
   paraHelix_ holds the parametric Fit objects - work horse
   

   Methods with _ ending should only be called after fitting
   
    """
    def __init__(self,file_name,direc='',name='fitted',minLen=7,imported=False,expected_helices=4):
        
        self.imported = imported

        self.file_name = file_name
        self.input_direc = direc
        self.error_thrown = False
        
        #fromDict usually means from fitted parameters or reconstruction from network
        if not self.imported:
            self.minLen = minLen #minimum length of dssp indentified secondary structure to count as a helix
            
            # list of helix indices from start pose
            self.helixRes, self.helix_ca_list = self.get_helical_res(direc,file_name,expected_helices=expected_helices) 
            self.name = name

        else:         
            if name=='fitted':
                self.name='imported'
            else:
                self.name = name
    
    @classmethod
    def from_endpoints(cls,dataIn,numHelices=4,name='fitted',imported=True,fromDict=False):
        """Reconstruct Parameters: from list of endpoints or Helical Endpoints dictionary."""
        #Helical endpoints dictionary of the form x1, y1 ,z1 etc two points per helix, define helix number
        helicalProt = cls(None,imported=True,name=name)
        helicalProt.paraHelix_ = []
        
        if fromDict:
            for i in range(1,numHelices*2+1,2):
                helicalProt.paraHelix_.append(ParametricHelix())
                p1 = np.array([dataIn[f'x{i}'],dataIn[f'y{i}'],dataIn[f'z{i}']])
                p2 = np.array([dataIn[f'x{i+1}'],dataIn[f'y{i+1}'],dataIn[f'z{i+1}']])

                helicalProt.paraHelix_[-1].helix_parameters = HelixParameters.fromEndPoints(p1,p2)
        else:
            
            for i in range(0,len(dataIn),2):
                helicalProt.paraHelix_.append(ParametricHelix())
                p1 = np.array([dataIn[i][0],dataIn[i][1],dataIn[i][2]])
                p2 = np.array([dataIn[i+1][0],dataIn[i+1][1],dataIn[i+1][2]])

                helicalProt.paraHelix_[-1].helix_parameters = HelixParameters.fromEndPoints(p1,p2)
            
            
            
        return helicalProt
            
        
    @classmethod
    def fromDict(cls,dicIn,name='fitted'):
        """Takes List of Dictionary  of parameters, One helix per dictionary Makes ParametricHelix Objects"""
        
        helicalProt = cls(None,imported=True,name=name)
        helicalProt.paraHelix_ = []
        
        for x in dicIn:
            helicalProt.paraHelix_.append(ParametricHelix())
            helicalProt.paraHelix_[-1].helix_parameters = HelixParameters.fromDict(x)
            
        return helicalProt
               


        #target_helix_resis = all_helix_resis[helix_number]
#----------Methods to extract helical residues and xyz of c alpha atoms using pymol library
    @staticmethod
    def hSel(helix_list, helix_num=0):
        resString = f'{helix_list[helix_num][0]}-{helix_list[helix_num][-1]}'
        return f'resi {resString} and name CA'

    def get_helical_res(self, direc,fname,expected_helices=4, warning_on = True):
        """Use pymol to get helical residue indices and CA List"""

        cmd.delete("all") #clear current protein in pymol
        name = fname[:-4] # remove extension
        
        # trying to open a file
        try:
            cmd.load(f'{direc}/{name}.pdb')
        except:
            print(f'Error loading {fname}')
            self.helixRes = [[]]
            self.helix_ca_list = [[]]
            return self.helixRes, self.helix_ca_list

        
        stored.resi = []
        
        
        #get ride of spaces for pymol
        pymol_name = name.replace(" ", "_")
        cmd.iterate_state(1, selector.process(f"{pymol_name} and ss 'H' and n. CA"), "stored.resi.append(resi)")

        self.helixRes = []
        xNow = -1
        for x in stored.resi:
            if int(x)> xNow:
                xNow = int(x)
                self.helixRes.append([])
            self.helixRes[-1].append(int(x))
            xNow = xNow+1
            
        warn=False

        for x in self.helixRes:
            if len(x) < self.minLen:
                warn=True
                self.error_thrown = True
                
        if not len(self.helixRes) == expected_helices:
            warn=True
            self.error_thrown = True

        if warn and warning_on:
            print(f'Check Protein {fname}: Helices not as expected. Remember load directory separately')

        self.helix_ca_list = []

        for i,c in enumerate(self.helixRes):
            self.helix_ca_list.append((np.array(cmd.get_coords(HelicalProtein.hSel(self.helixRes,helix_num=i),1))))
            
        return self.helixRes, self.helix_ca_list

#---Methods to fit helical indices in helix_lists call ParametricHelix class
    def fit_all(self):
        """Call fit_helix calls fit in ParametricHelix based on booleans"""
        #Stored in self.paraHelix_, as self.fit in ParametricHelix (ParametricFit_Record)
        
        self.paraHelix_ = []
        
        for x in range(len(self.helix_ca_list)):
            self.paraHelix_.append(self.fit_helix(x))
        return self.paraHelix_
    
    
    def fit_helix(self, target_helix_index):
        """Fits Helix via ParametricHelix class, called by fit_all, fit based on boolean inputs"""
        
        ph = ParametricHelix(f'fit {target_helix_index}')
        
        if len(self.helix_ca_list[target_helix_index]) < self.minLen:
            warnings.warn(f'{self.file_name[:-4]}: helix {target_helix_index} below minimum length to fit.')
            return ph

        
        ph.fit_str_helix_axis(self.helix_ca_list[target_helix_index], method ='least_sq')
   
        return ph
        
#---Methods to build full atom helical coordinates from Parameters   , wrapper for ParametricHelix   
    
    def build_single_helix_npose_(self, index):
        """Return numpy array of full atom (backbones) coordinates for the specified helix"""
       
        return self.paraHelix_[index].helix_parameters.to_npose()
    
    def build_nposes_(self):
        """Builds full atom (backbone) coordinates for all hlelices """
        
        self.npose_list_= []
        
        for count in range(len(self.paraHelix_)):
            self.npose_list_.append(self.build_single_helix_npose_(count))
            
        return self.npose_list_
    
    def get_endpoints_(self):
        """Caculates list of helical endpoints from helix parameters"""
        
        self.endpoints_list_ = []
        
        for x in self.paraHelix_:
            self.endpoints_list_.extend(x.helix_parameters.convert_to_endpoints())
            
        return self.endpoints_list_
        
#--- Methods that make poses and dump them based on the fitted parameters
    def get_CA_list_fit_(self):
        
        outList = []
        
        for x in self.paraHelix_:
            ca_list, axis = x.Ca_coords_from_stored_params()
            outList.append(ca_list)
            
        return outList

    def dump_comparison_poses_(self,outDirectory="",name=None, endpoints=False):
        """Creates pdb from fits and origination file together for comparison and the fits .pdb alone.
        I am unsure how to load coordinates directly into the cmd so I need save to disk.
        """
        if name:
            self.name = name
        
        cmd.delete("all") #clear current protein in pymol
        cmd.load(f'{self.input_direc}/{self.file_name}')
        
        self.dump_fits_(outDirec=outDirectory,name=name)
        cmd.alter("(chain A)","chain='B'")
        cmd.load(f'{outDirectory}{self.name}.pdb')
        
        if endpoints:
            self.dump_endpoints(outDirec=outDirectory)
            cmd.alter("(chain A)","chain='C'")
            cmd.load(f'{outDirectory}/{self.name}_endpoints.pdb')
        
        cmd.save(f'{outDirectory}/{self.name}_compare.pdb')
        
    def dump_fits_(self,outDirec="",name=None):
        """Makes pose from Parameters then to file"""
        #both puts axis endpoints and reconstructed helices
        #axis only no pose just points
        
        self.npose_list_ = self.build_nposes_()
        counter=1
        
        if name:
            self.name = name
        
        self.npose_ = np.array(np.empty((0,4), np.float32))
        
        for x in self.npose_list_:
            self.npose_ = np.vstack((self.npose_,x))
            
        nu.dump_npdb(self.npose_,f'{outDirec}/{self.name}.pdb')
        
    def dump_endpoints(self,outDirec="",name=None):
        
        if name:
            self.name=name
        
        self.get_endpoints_()
        
        HelicalProtein.makePointPDB(self.endpoints_list_,f'{self.name}_endpoints.pdb',outDirec=outDirec)

            
#-----------Methods to export fit data as csv lines,
    def export_single_helix_fit_(self,index,labelPull=False,decPoint=2):
        """Exports helix parameters as a single line from HelixParameters Object. Plus rmsd
        
        Of the form parameter_{helixnumber} for the labels, in order of helix number NtoC
        Labels returns the label as another line
        
        HelixEndsPoints( Axis only) of the form x1,y1,z1,x2,y2,z2 Plus rmsd
        """
        
        #from HelicalParameters object 
        #'omega1','phi1','d','r1','translate_x','translate_y','translate_z'
        #,'rotate_phi','rotate_theta','rotate_psi','length','rmsd'
        
        if labelPull:
            labels, outString = self.paraHelix_[index].helix_parameters.export_helix(labels=labelPull,decPoint=decPoint)
            return labels, outString
        else:
            outString = self.paraHelix_[index].helix_parameters.export_helix(decPoint=decPoint)
            return outString
        
    
    def export_to_dict(self):
        """Exports helical parameters in dictionary format"""
        
        outDict = dict()
        
        for x in range(len(self.paraHelix_)):
            
            params = self.paraHelix_[x].helix_parameters.get_dict()
            for key,val in params.items():
                outDict[self.concatNum(key,x+1)] = val
                
                
        return outDict
            
    
    def export_fits_(self, label=False, preventError=True):
        """CSV String Single line per Protein Helix Parameters and RMSD.
        Label=True also returns the labels as additional output (Labels, ExportedParameters)
        Uses export single helix fit. Prevent errors prevents output if error_thrown is true."
        """
        
        outString = f'{self.name},'

        for count in range(len(self.paraHelix_)):
                outString += f'{self.export_single_helix_fit_(count)},'
        
        if label:
            return self.getLabel_(), outString[:-1]
        elif preventError and self.error_thrown:
            return ''
        else:
            return outString[:-1]
        
    def getLabel_(self):
        
        labelAdd = 'name,'

        for helixNum in range(1,len(self.paraHelix_)+1):
            labels, outString = self.paraHelix_[0].helix_parameters.export_helix(labels=True)
            labelAdd = f'{labelAdd}{self.concatNum(labels,helixNum)},'
        
        return labelAdd[:-1]
#-------------Random Methods to determine    
        
    def get_hp_object_(self,index):
        return self.paraHelix_[index].get_helix_parameters()

    def get_rmsd_(self):
        """Return Max and Mean of the RMSD of each helix"""
        
        rmsd_list = []
        
        for index in range(len(self.paraHelix_)):
            rmsd_list.append(self.paraHelix_[index].helix_parameters.rmsd())
            
        
        return max(rmsd_list), np.mean(rmsd_list)
    
    
    @staticmethod
    def makePointPDB(coordinates,name,outDirec=""):
        """Take a vector of x,y,z vectors coords and writes pdb file with those points"""
        resid=1
        atomnumb=2
        with open(f'{outDirec}/{name}', 'w') as f:
            for i in coordinates:
                f.write('ATOM{:>7s}  CA  ALA A{:>4s}     {:>7.3f} {:>7.3f} {:>7.3f}  1.00  0.00           C\n'.format(str(atomnumb),str(resid),float(i[0]),float(i[1]),float(i[2])))
                resid=resid+1
                atomnumb=atomnumb+10
            f.write('END')
        return 1
        
    
    @staticmethod
    def concatNum(strIn,num):
        sVec=strIn.split(',')
        for count in range(len(sVec)):
            sVec[count] = f'{sVec[count]}_{num}'
        str1=","
        return str1.join(sVec)
    
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description =" 4 Helix data data/4H_dataset/models/  a few failures expected")
    parser.add_argument("inDirec", help="Directory with Helical Protein.")
    parser.add_argument("outFile", help="Output File Name")
    args = parser.parse_args()

    args.outFile = f'{args.outFile}.csv' #add file extension

    fileList = os.listdir(args.inDirec)
    h1 = HelicalProtein(fileList[0],direc=args.inDirec,name=fileList[0][:-4],expected_helices=4)
    h1.fit_all()

    with open(f'{args.outFile}','w') as f:
        f.write(h1.getLabel_())
        f.write('\n')

    for i,c in enumerate(fileList):

        h1 = HelicalProtein(c,direc=args.inDirec,name=c[:-4],expected_helices=4)
        h1.fit_all()

        fitString = h1.export_fits_()
        if fitString:
            with open(f'{args.outFile}','a') as f:
                f.write(f'{fitString}\n')

        if i%1000 ==0:
            print(f'{i} fits done')
        




