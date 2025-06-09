import numpy as np
import ROOT
import sys
import matplotlib.pyplot as plt   # needed for plotting


#print( fp_, ['cat'], ['sig_c'],['tau_c'],['sig_a'],['tau_a'],['sig_b'],['tau_b'],['u'],['tau_e'], np.sqrt(wavefit.redchi)*1.0e6 )

def vtoa( buf, entries ):
  retarr = []
  for idx in range(0,entries) :
    retarr.append( buf[idx] )
  return retarr

tree = ROOT.TTree('waveformparams','')
tree.SetMarkerColor(ROOT.kRed)
tree.SetMarkerStyle(ROOT.kFullDotMedium)
tree.ReadFile('pars.txt','cat:sig_c:tau_c:sig_a:tau_a:sig_b:tau_b:u:tau_e:rmsr',' ')


