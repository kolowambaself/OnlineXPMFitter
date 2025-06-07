import numpy as np
from scipy.special import wofz
from lmfit.models import SkewedVoigtModel
from lmfit.models import ExponentialGaussianModel
from lmfit import Model
from lmfit import Minimizer, minimize, fit_report, conf_interval, printfuncs
from scipy import integrate
from scipy.special import erfc
from uncertainties.core import wrap
import os
from pathlib import WindowsPath, Path
import pandas as pd
import matplotlib.pyplot as plt   # needed for plotting
from time import sleep            # needed for delays
from ctypes import *
import sys

lifetime = 1000.0

def biGaus_skew(x,cat,sig_c,tau_c,sig_a,tau_a) :
  tc = 10.0
  ta = 81.9
  i_c = cat*(erfc((tc-x)/sig_c))*np.exp(-(x-tc)/tau_c)
  adjusted_an = np.exp(-(ta-tc)/lifetime)*cat
  #i_a = (sig_c/sig_a)*adjusted_an*(erfc((ta-x)/sig_a))*np.exp(-(x-ta)/tau_a)
  #i_a = (sig_c/sig_a)*adjusted_an*2.0/(1.0+np.exp((ta-x)/sig_a))*np.exp(-(x-ta)/tau_a)
  i_a = -((sig_c+tau_c)/(sig_a+tau_a))*erfc((ta-x)/sig_a)*np.exp(-(x-ta)/(tau_a))*adjusted_an
  return  i_c + i_a

def doubleBiGaus(x, cat):
    tc = 10.0
    ta = 81.9
    sig_c = 1.0
    sig_a = 1.0
    i_c = cat*np.exp(-((x-tc)**2)/(2*sig_c**2))
    adjusted_an = np.exp(-(ta-tc)/lifetime)*cat 
    i_a = (sig_c/sig_a)*adjusted_an*np.exp(-((x-ta)**2)/(2*sig_a**2))
    return i_c - i_a

sqrt2 = np.sqrt(2.0)
sqrt2pi = np.sqrt(2.0*np.pi)
def expGaus_skewVoigt(x,cat,sig_c,tau_c,sig_a,gam_a,skew) :
  tc = 10.0
  ta = 81.9
  i_c = cat*(erfc((tc-x)/sig_c))*np.exp(-(x-tc)/tau_c)
  adjusted_an = np.exp(-(ta-tc)/lifetime)*cat

  z_a = (x - ta + gam_a*1j)/(sig_a*sqrt2)
  realw_a = -adjusted_an*np.real(wofz(z_a))/(sqrt2pi*sig_a)
  i_a = (sig_c)/(sig_a)*(realw_a)*(erfc(skew*(ta-x)/(sig_a*sqrt2)))
  return  i_c + i_a # + 0.028515897

def expGaus_skew(x,cat,sig_c,tau_c,sig_a,tau_a) :
  tc = 10.0
  ta = 81.9
  i_c = cat*(erfc((tc-x)/sig_c))*np.exp(-(x-tc)/tau_c)
  adjusted_an = np.exp(-(ta-tc)/lifetime)*cat
  i_a = -(np.sqrt(sig_c**2+tau_c**2)/np.sqrt(sig_a**2+tau_a**2))*(erfc((ta-x)/sig_a))*(np.exp(-(x-ta)/(tau_a)))*adjusted_an
  return  i_c + i_a

"""-----------------------------------------------------------------------"""
def skewVoigtC_skewVoigtA_yesGamma (x, cat, sig_c, gam_c, skew_c, an, sig_a, skew_a,offst):
    CF = 1.0  # feedback capacitance (not really 1, but it doesn't really matter anyway since it will divide out,
              # it's just here to turn voltage into charge in the equation)
    ta = 81.9 # where t = 1 on the anode
    tc = 10.0 # where t = 1 on the cathode
    sqrt2 = np.sqrt(2.0)
    sqrt2pi = np.sqrt(2.0*np.pi)
    # anode
    z_a = (x - ta + sig_a*1j)/(sig_a*sqrt2) # first part of the Voigt dist
    adjusted_an = np.exp(-(ta-tc)/lifetime)*an 
    realw_a = adjusted_an*np.real(wofz(z_a))/(sqrt2pi*sig_a) # real component of a Faddeeva func of A 
    i_a = (realw_a)*(2.0/(1.0+np.exp((ta-x)/skew_a))) # multiply everything by the Fermi function
    anode = -(1.0/CF)*i_a

    # cathode - this is just the anode code again 
    z_c = (x - tc + gam_c*1j)/(sig_c*sqrt2)
    realw_c = -cat*np.real(wofz(z_c))/(sqrt2pi*sig_c)
    i_c = (realw_c)*(2.0/(1.0+np.exp((tc-x)/skew_c)))
    cathode = -(1.0/CF)*i_c

    return cathode + anode + offst

#wavmodel = Model(biGaus_skew,nan_policy='raise')
#wavparams = wavmodel.make_params()
#wavparams['cat'].value = 1.0   # formerly qc
#wavparams['sig_c'].value = 2.3331900249976414
#wavparams['tau_c'].value = 1.9260917008369134
#wavparams['sig_a'].value = 1.5068445074484516
#wavparams['tau_a'].value = 1.3511128944360826

#wavmodel = Model(doubleBiGaus,nan_policy='raise')
#wavparams = wavmodel.make_params()
#wavmodel = Model(skewVoigtC_skewVoigtA_yesGamma,nan_policy='raise')
#wavparams = wavmodel.make_params()
#wavparams['cat'].value = 1.0   # formerly qc

#wavmodel = Model(expGaus_skewVoigt,nan_policy='raise')
#wavparams = wavmodel.make_params()
#wavparams['cat'].value = 1.0   # formerly qc
#wavparams['sig_c'].value =  2.4426740601859485
#wavparams['tau_c'].value = 2.0639455769341524
#wavparams['sig_a'].value = 0.5376770542018842
#wavparams['gam_a'].value = 1.0603610428272454
#wavparams['skew'].value =  0.05928022074975299

wavmodel = Model(expGaus_skew,nan_policy='raise')
wavparams = wavmodel.make_params()
wavparams['cat'].value = 1.0
wavparams['sig_c'].value = 2.323785373027948
wavparams['tau_c'].value = 1.9216895263092932
wavparams['sig_a'].value =  1.5121733234495
wavparams['tau_a'].value = 1.35520629870204

t = np.linspace(0.0,163.79,16380)
t_ad2 = np.linspace(0.0,163.79e-6,16380)
print('ground-truth e- lifetime [us]',lifetime)

fp_ = '/home/kolo/vfp25/waveforms/control9r_python.wf.9'
if len(sys.argv) > 1 :
    fp_ = sys.argv[1]

rawdf = pd.read_csv(fp_,header=None)

Re = 5.0e4 #XPM effective DC resistance
CF = 10.0e-12 #UA1 preamp feedback cap
RF = (395.4e-6)/CF #UA1 feedback resistance
v_of_t = wavmodel.eval(wavparams,x=t)
#plt.plot(t,v_of_t)
t_raw = np.array(rawdf[0])
v_raw = np.array(rawdf[1])
dt = t_raw[1] - t_raw[0]
v_in = (Re*CF)*np.exp(-t_raw*1.0e-6/(RF*CF))*np.gradient( np.exp(t_raw*1.0e-6/(RF*CF))*v_raw )/(dt*1.0e-6)
plt.plot(t_raw,v_in,'-')
plt.grid(True)
plt.xlabel('Time [$\mu$s]')
#plt.ylabel('Signal [AU]')
plt.show()

