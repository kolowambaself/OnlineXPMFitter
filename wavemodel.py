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
import ROOT
import sys

sr2 = np.sqrt(2.0)
CF = 10.0e-12 #UA1 preamp feedback cap
RF = (395.3e-6)/CF #UA1 feedback resistance
DELTAT = 4.000000000000625e-08

def fitter_func(x, cat, an, tcrise, tarise, offst,thold ):
    z = np.array(x)
    x_beg = z[z<10.0]
    x_mid = z[(z>=10.0)*(z<81.9)]
    x_end = z[z>=81.9]
    y_beg = 0.5*cat*erfc(-(x_beg-10.0)/tcrise) - 0.5*an*erfc(-(x_beg-81.9)/tarise)
    y_mid = 0.5*cat*erfc(-(x_mid-10.0)/tcrise)*np.exp(-(x_mid-10.0)/thold) - 0.5*an*erfc(-(x_mid-81.9)/tarise)
    y_end = 0.5*cat*erfc(-(x_end-10.0)/tcrise)*np.exp(-(x_end-10.0)/thold) - 0.5*an*erfc(-(x_end-81.9)/tarise)*np.exp(-(x_end-81.9)/thold)
    y = np.concatenate((y_beg,y_mid,y_end),axis=None)
    y = y + offst
    return y

def standard_fitter( t , v ) :
    p_i = [49.98262, 46.10659, 10.0, 1.0, 2.9, 81.9, 395.3, 0.8, 0.9, 43.619015]
    wavmodel = Model(fitter_func,nan_policy='raise')
    wavparams = wavmodel.make_params()
    wavparams['cat'].value = p_i[0]
    wavparams['cat'].vary = True
    wavparams['an'].value = p_i[1]
    wavparams['an'].vary = True
    wavparams['thold'].value = p_i[6]
    wavparams['thold'].vary = False
    wavparams['tcrise'].value = p_i[3]
    wavparams['tcrise'].vary = False
    wavparams['tarise'].value = p_i[4]
    wavparams['tarise'].vary = False
    wavparams['offst'].value = p_i[9]
    wavparams['offst'].vary = True
    fitterfit = wavmodel.fit( v, x=t , params = wavparams )
    bestparams = fitterfit.best_values
    
    tau_e = (p_i[5] - p_i[2])/np.log( bestparams['cat']/bestparams['an'] )
    return [bestparams['offst'], tau_e , bestparams['cat']]



def tripleEMG(x,cat,sig_c,tau_c,sig_a,tau_a,sig_b,tau_b,tau_e,u) :
    tc = 10.0
    ta = 81.9
    C = (cat/1000.0)*(CF/1.0e-6) #Note that parameter C is in C/s, but cat is in mA⋅us/F. Note also that CF=10 pF
    i_c = (C/(2*tau_c))*np.exp(-sig_c*sig_c/(2*tau_c*tau_c))*np.exp((tc-x)/tau_c)*erfc((tc-x)/(sr2*sig_c))

    i_a = (C*(1-u)/(2*tau_a))*np.exp(-sig_a*sig_a/(2*tau_a*tau_a))*np.exp((ta-x)/tau_a)*erfc((ta-x)/(sr2*sig_a))*np.exp(-(ta-tc)/tau_e)

    i_a = i_a + (C*u/(2*tau_b))*np.exp(-sig_b*sig_b/(2*tau_b*tau_b))*np.exp((ta-x)/tau_b)*erfc((ta-x)/(sr2*sig_b))*np.exp(-(ta-tc)/tau_e)

    return i_c - i_a

wavmodel = Model(tripleEMG,nan_policy='raise')
wavparams = wavmodel.make_params()
wavparams['sig_c'].value = 1.7906811646890577
wavparams['sig_c'].vary = True
wavparams['tau_c'].value = 1.9877563125009832
wavparams['tau_c'].vary = True
wavparams['sig_a'].value = 1.7345324740561452
wavparams['sig_a'].vary = True
wavparams['tau_a'].value = 3.41643895917533
wavparams['tau_a'].vary = True
wavparams['sig_b'].value = 0.8326697719514247
wavparams['sig_b'].vary =  True
wavparams['tau_b'].value = 0.6196258987634229
wavparams['tau_b'].vary = True
wavparams['u'].value = 0.6174349790337955 #fractional admixture of the 'b' anode pulse
wavparams['u'].vary = True #fractional admixture of the 'b' anode pulse


fp_ = '/home/kolo/vfp25/waveforms/control9r_python.wf.40'
if len(sys.argv) > 1 :
    fp_ = sys.argv[1]

rawdf = pd.read_csv(fp_,header=None)

t_raw = np.array(rawdf[0])
v_raw = np.array(rawdf[1])
dt = t_raw[1] - t_raw[0]
i_in = (CF/1000.0)*np.exp(-t_raw*1.0e-6/(RF*CF))*np.gradient( np.exp(t_raw*1.0e-6/(RF*CF))*v_raw )/(dt*1.0e-6) #i_in is in amps
plt.plot(t_raw,i_in,'-') 

hist2 = ROOT.TH2F('hist2','',int(len(t_raw)/10), t_raw[0], t_raw[-1], 80 , -2.0e-7 , 2.0e-7 )
for t,iv in zip( t_raw, i_in ) :
    hist2.Fill(t,iv)
myprof = hist2.ProfileX('myprof')

xbinned = []
ybinned = []
e = []
ex = []
for bin in range(1,myprof.GetNbinsX()+1) :
   # print(bin,myprof.GetBinContent(bin),myprof.GetBinEntries(bin))
   #if( myprof.GetBinEntries(bin)<=3 ): continue
   xbinned.append(myprof.GetBinCenter(bin))
   ybinned.append(myprof.GetBinContent(bin))
   e.append(myprof.GetBinError(bin))
   ex.append(myprof.GetBinWidth(bin)/2.0)
       
plt.errorbar(xbinned,ybinned,e,fmt='.')

wavparams['tau_e'].value = standard_fitter( t_raw , v_raw )[1] 
wavparams['tau_e'].vary = False
wavparams['cat'].value = standard_fitter( t_raw , v_raw )[2] # overall normalization
wavparams['cat'].vary = False   # overall normalization
wavefit = wavmodel.fit( i_in, x=t_raw , params = wavparams)#, max_nfev=1)
#wavefit = wavmodel.fit( ybinned , x=xbinned , params = wavparams)#, max_nfev=1)

bestparams = wavefit.best_values

wavparams['cat'].value = bestparams['cat']   # overall normalization
wavparams['sig_c'].value = bestparams['sig_c']
wavparams['tau_c'].value = bestparams['tau_c']
wavparams['sig_a'].value = bestparams['sig_a']
wavparams['tau_a'].value = bestparams['tau_a']
wavparams['sig_b'].value = bestparams['tau_b']
wavparams['tau_b'].value = bestparams['tau_b']
wavparams['u'].value = bestparams['u']
#print(bestparams)




ioftprime = wavmodel.eval(wavparams,x=t_raw)



print( fp_, bestparams['cat'], bestparams['sig_c'], bestparams['tau_c'], bestparams['sig_a'],bestparams['tau_a'],bestparams['sig_b'],bestparams['tau_b'], bestparams['u'],bestparams['tau_e'], np.sqrt(wavefit.redchi)*1.0e6 )
plt.plot(t_raw , ioftprime )

integrand = ioftprime*np.exp(t_raw*1.0e-6/(RF*CF))

vout = np.exp(-t_raw*1.0e-6/(RF*CF))*integrate.cumulative_trapezoid(integrand, t_raw*1.0e-6, dx=dt*1.0e-6, initial=0)/CF
vout = vout*1000.0 #convert to millivolts

plt.plot(t_raw,vout+standard_fitter( t_raw , v_raw )[0])
plt.plot(t_raw, v_raw)
plt.grid(True)
plt.xlabel('Time [$\mu$s]')
#plt.ylabel('Signal [mA]')
#plt.show()

