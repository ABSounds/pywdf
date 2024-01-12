#  As described in : 
#  https://dafx2020.mdw.ac.at/proceedings/papers/DAFx20in22_paper_39.pdf

import sys, os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from core.wdf import *
from core.rtype import *
from core.circuit import Circuit


class RCA_MK2_SEF(Circuit):
    
    def __init__(self, sample_rate: int, highpass_cutoff: float, lowpass_cutoff: float) -> None:

        # mod switches engaged by default

        self.lowpass_mod = True
        self.highpass_mod = True

        self.k = 560

        # measured C & L vals for each HP & LP knob position

        self.HP_vals = {
            0 : {'C' : 9999999, 'L' : 9999999},
            175 : {'C' : 1.6e-6, 'L' : 255.6e-3},
            248 : {'C' : 1.15e-6, 'L' : 176.9e-3},
            352 : {'C' : .8e-6, 'L' : 126.4e-3},
            497 : {'C' : .57e-6, 'L' : 90.11e-3},
            699 : {'C' : .4e-6, 'L' : 63.9e-3},
            1002 : {'C' : .272e-6, 'L' : 44.56e-3},
            1411 : {'C' : .2e-6, 'L' : 31.79e-3 },
            2024 : {'C' : .15e-6, 'L' : 21.77e-3},
            2847 : {'C' : .1e-6, 'L' : 15.63e-3 },
            3994 : {'C' : .069e-6, 'L' : 11.18e-3}
        }

        self.LP_vals = {
            175 : {'C' : 3.22e-6, 'L' : 511.1e-3 },
            245 : {'C' : 2.3e-6, 'L' :  365.2e-3 },
            350 : {'C' : 1.6e-6, 'L' : 255.6e-3 },
            499 : {'C' : 1.15e-6, 'L' : 178.6e-3 },
            703 : {'C' : .8e-6, 'L' : 126.4e-3 },
            996 : {'C' : .57e-6, 'L' : 90.02e-3 },
            1408 : {'C' : .4e-6, 'L' : 63.54e-3},
            1989 : {'C' : .272e-6, 'L' : 45.08e-3},
            2803 : {'C' : .2e-6, 'L' : 32.13e-3 },
            3992 : {'C' : .15e-6, 'L' : 22.38e-3 },
            999999 : {'C' : 1e-15, 'L' : 1e-15}
        }

        self.fs = sample_rate

        self.Z_input = 560
        self.Z_output = 560

        self.highpass_cutoff = highpass_cutoff
        self.lowpass_cutoff = lowpass_cutoff

        self.C_HP = 1e-6
        self.L_HP = 1e-3
        self.C_LP = 1e-6
        self.L_LP = 1e-3

        self.Rt = Resistor(self.Z_output)
        self.L_LPm2 = Inductor(self.L_LP, self.fs)

        self.S8 = SeriesAdaptor(self.L_LPm2, self.Rt)
        self.C_LPm1 = Capacitor(self.C_LP, self.fs)

        self.P4 = ParallelAdaptor(self.C_LPm1, self.S8)
        self.L_LPm1 = Inductor(self.L_LP, self.fs)

        self.S7 = SeriesAdaptor(self.L_LPm1, self.P4)
        self.L_LP2 = Inductor(self.L_LP, self.fs)

        self.S6 = SeriesAdaptor(self.L_LP2, self.S7)
        self.C_LP1 = Capacitor(self.C_LP, self.fs)

        self.P3 = ParallelAdaptor(self.C_LP1, self.S6)
        self.L_LP1 = Inductor(self.L_LP, self.fs)

        self.S5 = SeriesAdaptor(self.L_LP1, self.P3)
        self.C_HP2 = Capacitor(self.C_HP, self.fs)

        self.S4 = SeriesAdaptor(self.C_HP2, self.S5)
        self.L_HP1 = Inductor(self.L_HP, self.fs)

        self.P2 = ParallelAdaptor(self.L_HP1, self.S4)
        self.C_HP1 = Capacitor(self.C_HP, self.fs)

        self.S3 = SeriesAdaptor(self.C_HP1, self.P2)
        self.C_HPm2 = Capacitor(self.C_HP, self.fs)
        
        self.S2 = SeriesAdaptor(self.C_HPm2, self.S3)
        self.L_HPm = Inductor(self.L_LP, self.fs)

        self.P1 = ParallelAdaptor(self.L_HPm, self.S2)
        self.C_HPm1 = Capacitor(self.C_HP, self.fs)

        self.S1 = SeriesAdaptor(self.C_HPm1, self.P1)
        self.Rin = Resistor(self.Z_input)

        self.S0 = SeriesAdaptor(self.Rin, self.S1)
        self.Vin = IdealVoltageSource(self.S0)

        self.set_HP_components()
        self.set_LP_components()

        super().__init__(self.Vin, self.Vin, self.Rt)


    def set_HP_components(self):
        wc = self.highpass_cutoff * 2. * np.pi
        self.C_HP = np.sqrt(2) / (self.k * wc)
        self.L_HP = self.k / (2. * np.sqrt(2) * wc)

        self.C_HP1.set_capacitance(self.C_HP)
        self.C_HP2.set_capacitance(self.C_HP)
        self.L_HP1.set_inductance(self.L_HP)
        
        if self.highpass_mod == False:
            wc = 1e-8
            self.C_HP = np.sqrt(2) / (self.k * wc)
            self.L_HP = self.k / (2 * np.sqrt(2) * wc)

        self.C_HPm1.set_capacitance(self.C_HP)
        self.C_HPm2.set_capacitance(self.C_HP)
        self.L_HPm.set_inductance(self.C_HP)
        

    def set_LP_components(self):
        wc = self.lowpass_cutoff * 2 * np.pi
        self.C_LP = (2 * np.sqrt(2)) / (self.k * wc)
        self.L_LP = (np.sqrt(2) * self.k) / wc      

        self.C_HP1.set_capacitance(self.C_LP)
        self.C_HP2.set_capacitance(self.C_LP)
        self.L_HP1.set_inductance(self.L_LP)
         
        if self.set_lowpass_mod == False:
            wc = 1e8
            self.C_LP = (2 * np.sqrt(2)) / (self.k * wc)
            self.L_LP = (np.sqrt(2) * self.k) / wc

        self.C_HPm1.set_capacitance(self.C_LP)
        self.C_HPm2.set_capacitance(self.C_LP)
        self.L_HPm.set_inductance(self.L_LP)

    def process_sample(self, sample: float) -> float:
        gain_db = 6 # factor to compensate for gain loss
        k = 10 ** (gain_db / 20)
        return k * super().process_sample(sample)

    def set_highpass_cutoff(self, new_cutoff):
        self.highpass_cutoff = new_cutoff
        self.set_HP_components()

    def set_lowpass_cutoff(self, new_cutoff):
        self.lowpass_cutoff = new_cutoff
        self.set_LP_components()

    def set_highpass_mod(self, mod):
        if self.highpass_mod != mod:
            self.highpass_mod = mod
            self.set_HP_components()

    def set_lowpass_mod(self, mod):
        if self.set_lowpass_mod != mod:
            self.lowpass_mod = mod
            self.set_LP_components()

    def set_Z_input(self, new_Z):
        if self.Z_input != new_Z:
            self.Z_input = new_Z
            self.Rin.set_resistance(new_Z)

    def set_Z_output(self, new_Z):
        if self.Z_output != new_Z:
            self.Z_output = new_Z
            self.Rt.set_resistance(new_Z)

    def get_closest(self, d, search_key):
        if d.get(search_key):
            return search_key, d[search_key]
        key = min(d.keys(), key=lambda key: abs(key - search_key))
        return key, d[key]


if __name__ == '__main__':

    mk2 = RCA_MK2_SEF(44100, 20, 20e3)
    mk2.set_highpass_mod(0)
    mk2.set_lowpass_mod(0)
    vals = range(0, 8000, 1000)
    mk2.plot_freqz_list(vals, mk2.set_highpass_cutoff, 'hp cutoff')
    
    mk2.set_highpass_cutoff(0)
    vals = range(1000,8000,1000)
    mk2.plot_freqz_list(vals, mk2.set_lowpass_cutoff, 'lp cutoff')
