from tkinter import *
import socket, threading
import time
import datetime
import sys
import os
import random
from WF_SDK import device, scope, wavegen  # import instruments
from ctypes import *
import numpy as np
from lmfit import Model
from scipy.special import erfc

if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")
sr2 = np.sqrt(2.0)
CF = 10.0e-12 #UA1 preamp feedback cap
RF = (395.3e-6)/CF #UA1 feedback resistance

class fake(Frame) :

    def tripleEMG(x,cat,sig_c,tau_c,sig_a,tau_a,sig_b,tau_b,tau_e,u) :
        tc = 10.0
        ta = 81.9
        C = (cat/1000.0)*(CF/1.0e-6) #Note that parameter C is in C/s, but cat is in mA⋅us/F. Note also that CF=10 pF
        i_c = (C/(2*tau_c))*np.exp(-sig_c*sig_c/(2*tau_c*tau_c))*np.exp((tc-x)/tau_c)*erfc((tc-x)/(sr2*sig_c))
        i_a = (C*(1-u)/(2*tau_a))*np.exp(-sig_a*sig_a/(2*tau_a*tau_a))*np.exp((ta-x)/tau_a)*erfc((ta-x)/(sr2*sig_a))*np.exp(-(ta-tc)/tau_e)
        i_a = i_a + (C*u/(2*tau_b))*np.exp(-sig_b*sig_b/(2*tau_b*tau_b))*np.exp((ta-x)/tau_b)*erfc((ta-x)/(sr2*sig_b))*np.exp(-(ta-tc)/tau_e)
        return i_c - i_a

    def laser(self) :
        try:
            self.device_data = device.open('Analog Discovery 2')     #if the AD2 is connected
            self.is_connected = True
            print('AD2 connected')
        except:     # if it isn't, we want to ignore the code that handles it. i would make this nicer if i could find the wf_sdk docs
            self.is_connected = False
            print('AD2 not found')
        if self.is_connected == True :
            try:
                device_ct = c_int()
                dwf.FDwfEnum( scope.constants.enumfilterAll , byref(device_ct) )

                # connect to the device
                hdwf0 = c_int()
                hdwf1 = c_int()
                #dwf.FDwfDeviceOpen(c_int(1),byref(hdwf1))

                device_name0 = create_string_buffer(32)
                device_name1 = create_string_buffer(32)
                dwf.FDwfEnumDeviceName( 0 , byref(device_name0) )
                dwf.FDwfEnumDeviceName( 1 , byref(device_name1) )
                self.device_data = device.data()
                if device_name0.value == 'Analog Discovery 3' :
                    dwf.FDwfDeviceOpen(c_int(0),byref(hdwf0))
                    self.device_data.handle = hdwf0.value
                else :
                    dwf.FDwfDeviceOpen(c_int(1),byref(hdwf1))
                    self.device_data.handle = hdwf1.value

                self.device_data.name = device_name0.value
                self.device_data = device.__get_info__(self.device_data)

                """-----------------------------------"""
                # handle devices without analog I/O channels
                if self.device_data.name != "Digital Discovery":

                    # initialize the scope with default settings
                    scope.open(self.device_data)

                    # generate a 10KHz sine signal with 2V amplitude on channel 1
                    wavegen.enable(self.device_data, channel=1)

                    # set up triggering on scope channel 1
                    scope.trigger(self.device_data, enable=True, source=scope.constants.trigsrcDetectorAnalogIn, channel=2, level=0.1)
                    wavegen.dwf.FDwfAnalogInConfigure(self.device_data.handle, c_int(1), c_int(False), c_int(True))

                    wavegen.dwf.FDwfAnalogOutNodeEnableSet(self.device_data.handle, c_int(0), scope.constants.AnalogOutNodeCarrier, c_bool(True))
                    wavegen.dwf.FDwfAnalogOutTriggerSourceSet(self.device_data.handle,c_int(-1),scope.constants.trigsrcDetectorAnalogIn)
                    wavegen.dwf.FDwfDeviceTriggerSet(self.device_data.handle,c_int(-1),scope.constants.trigsrcDetectorAnalogIn)
                    wavegen.dwf.FDwfAnalogOutRunSet(self.device_data.handle, c_int(0) , c_double(163.80e-6))
                    wavegen.dwf.FDwfAnalogOutRepeatSet(self.device_data.handle, c_int(0), c_int(1))
                    #wavegen.dwf.FDwfAnalogOutIdleSet(device_data.handle, c_int(-1), scope.constants.DwfAnalogOutIdleInitial)
                    self.ad3()
            except Exception as exc :
                print(exc)
                return

    def ad3(self) :
        try :
            wavmodel = Model(self.tripleEMG,nan_policy='raise')
            wavparams = wavmodel.make_params()
            cat = float(self.catText().get('1.0','end-1c'))
            tau_e = float(self.tau_eText().get('1.0','end-1c'))
            if cat < 45 :
                sig_c = 1.7841530799865724
                tau_c = 1.9858030259609223
                sig_a = 1.8957586348056794
                tau_a = 3.783218514919281
                sig_b = 0.8382677763700486
                tau_b = 0.6204022705554962
                u = 0.6140601396560669
            elif cat >= 45 and cat < 55 :
                sig_c = 1.8104135632514953
                tau_c = 2.007856070995331
                sig_a = 1.9639676411946614
                tau_a = 4.217148415247599
                sig_b = 0.8468018611272176
                tau_b = 0.7376075367132823 
                u = 0.6173524677753448
            else :
                sig_c = 1.9419307231903076 
                tau_c = 2.2630012035369873
                sig_a = 1.7463257789611817
                tau_a = 4.616640996932984
                sig_b = 0.9523972332477569
                tau_b = 0.8427122056484222 
                u = 0.5668412625789643

            wavparams['sig_c'].value = sig_c
            wavparams['tau_c'].value = tau_c
            wavparams['sig_a'].value = sig_a
            wavparams['tau_a'].value = tau_a
            wavparams['sig_b'].value = sig_b
            wavparams['tau_b'].value = tau_b
            wavparams['u'].value = u #fractional admixture of the 'b' anode pulse
            wavparams['tau_e'].value = tau_e
            wavparams['cat'].value = cat # in 'XPM units' (mA⋅us/F)

            t = np.linspace(0.0,163.79,16380)
            t_ad2 = np.linspace(0.0,163.79e-6,16380)

            v_of_t = 5.0e6*wavmodel.eval(wavparams,x=t) #5M resistor converts current to voltage
            st = c_int(0)
            mydata = (c_double * len(v_of_t))()
            for i in range(0, len(mydata)) :
                mydata[i] = c_double(v_of_t[i])
            wavegen.dwf.FDwfAnalogOutNodeFunctionSet(self.device_data.handle, c_int(0), scope.constants.AnalogOutNodeCarrier, scope.constants.funcCustom )
            wavegen.dwf.FDwfAnalogOutNodeDataSet(self.device_data.handle, c_int(0), scope.constants.AnalogOutNodeCarrier, mydata, c_int(len(v_of_t)) )
            wavegen.dwf.FDwfAnalogOutNodeFrequencySet(self.device_data.handle, c_int(0), scope.constants.AnalogOutNodeCarrier, c_double(6.105006105e3))
            wavegen.dwf.FDwfAnalogOutNodeAmplitudeSet(self.device_data.handle, c_int(0), scope.constants.AnalogOutNodeCarrier, c_double(0.2175))
            try:
                pathExists = (WindowsPath.home() / '.shutterclosed').exists()
            except:
                pathExists = Path('/tmp/.shutterclosed').exists()

            if pathExists == True: #shutter is closed
                wavegen.dwf.FDwfAnalogOutConfigure(device_data.handle, c_int(0), c_int(0))
            else :
                wavegen.dwf.FDwfAnalogOutConfigure(device_data.handle, c_int(0), c_int(1))
        except Exception as exc :
            print(exc)
            return
        after(250,self.ad3)

    def on_closing(self) :
        self.run = False
        # reset the scope
        if self.is_connected :
            scope.close(self.device_data)
            # reset the wavegen
            wavegen.close(self.device_data)

        os._exit(0)

    def __init__(self):
        self.window = Tk()
        self.window.geometry('240x120')
        self.window.title( 'XPM HW sim' )
        self.is_connected = False
        self.tau_eText = Text( height=1, width = 18 )
        self.tau_eText.insert(END,' ')
        self.tau_eText.grid(row=1,column=1,sticky=W)
        self.tau_eLabel = Label(height=1, width = 18)
        self.tau_eLabel.config(text='e- lifetime (us)')
        self.tau_eLabel.grid(row=2, column=1, sticky=W)

        self.catText = Text( height=1, width = 18 )
        self.catText.insert(END,' ')
        self.catText.grid(row=3,column=1,sticky=W)
        self.catLabel = Label(height=1, width = 18)
        self.catLabel.config(text='cathode signal (mV)')
        self.catLabel.grid(row=4, column=1, sticky=W)

        self.button1 = Button( self.window, text='GO!', command = self.laser , height=4)
        self.button1.grid(column=2, row=1,rowspan=4)

        self.window.protocol( 'WM_DELETE_WINDOW', self.on_closing )
        self.window.mainloop()
        
appl = fake()
