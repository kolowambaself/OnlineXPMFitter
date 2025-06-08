import numpy as np
import ROOT
import sys
import matplotlib.pyplot as plt   # needed for plotting



def vtoa( buf, entries ):
  retarr = []
  for idx in range(0,entries) :
    retarr.append( buf[idx] )
  return retarr

tree = ROOT.TTree('waveformparams','')
tree.SetMarkerColor(ROOT.kRed)
tree.SetMarkerStyle(ROOT.kFullDotMedium)
tree.ReadFile('pars.txt','cat_ap:C:sig_c:sig_a:tau_a:tau_b:u:tau_e:redchi',' ')

tree.Draw('redchi:tau_e','','goff')
redchi = vtoa( tree.GetV1(), tree.GetSelectedRows() )
tau_e = vtoa( tree.GetV2(), tree.GetSelectedRows() )

tree.Draw('redchi:cat_ap','','goff')
redchi = vtoa( tree.GetV1(), tree.GetSelectedRows() )
cat_ap = vtoa( tree.GetV2(), tree.GetSelectedRows() )


plt.plot( cat_ap , redchi , '.' )

plt.xlabel('cat_ap [mV]')
plt.ylabel('redchi [mA]^2')
plt.show()

