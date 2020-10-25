from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.Library import SpiceLibrary
import matplotlib.pyplot as plt
import numpy as np

#import PySpice
#print(PySpice.__file__)


class opamp(Circuit):
    def __init__(self, name):
        Circuit.__init__(self, name)
        libraries_path = 'C:\\Users\\micha\\PycharmProjects\\circuit_model\\component_libraries\\transistor'
        spice_library = SpiceLibrary(libraries_path)
        self.include(spice_library['ptm65nm_nmos'])
        self.include(spice_library['ptm65nm_pmos'])

        # input pair
        self.MOSFET(1, "vd_l", "inp", "vtail", self.gnd, model='ptm65nm_nmos', width=5e-6, length=0.65e-6)
        self.MOSFET(2, "vd_r", "inn", "vtail", self.gnd, model='ptm65nm_nmos', width=5e-6, length=0.65e-6)

        # active load
        self.MOSFET(3, "vd_l", "vd_l", "vdd", "vdd", model='ptm65nm_pmos', width=2.5e-6, length=0.65e-6)
        self.MOSFET(4, "vd_r", "vd_l", "vdd", "vdd", model='ptm65nm_pmos', width=2.5e-6, length=0.65e-6)

        # tail current
        self.MOSFET(5, "vtail", "vbias", self.gnd, self.gnd, model='ptm65nm_nmos', width=5e-6, length=1e-6)
        # input current mirror
        self.MOSFET(6, "vbias", "vbias", self.gnd, self.gnd, model='ptm65nm_nmos', width=5e-6, length=1e-6)

        # sources
        self.I(1, "vdd", "vbias", 1e-6)

        # load
        self.R('load', 'vd_r', self.gnd, 1e9)
        self.C('load', 'vd_r', self.gnd, 5e-12)


def run_dc(circuit, vdd, inp, inn):
    circuit.V(1, "vdd", circuit.gnd, vdd)
    circuit.V(2, "inp", circuit.gnd, "dc %f ac 1" % inp)
    circuit.V(4, "inn", circuit.gnd, inn)
    simulator = circuit.simulator()
    analysis_dc = simulator.operating_point()
    for node in analysis_dc.nodes.values():
        print('{}: {:5.2f} V'.format(str(node), float(node)))


def run_ac(circuit, vdd, inp, inn):
    circuit.V(1, "vdd", circuit.gnd, vdd)
    circuit.V(2, "inp", circuit.gnd, "dc %f ac 1" % inp)
    circuit.V(4, "inn", circuit.gnd, inn)
    simulator = circuit.simulator()
    analysis_ac = simulator.ac(start_frequency=1, stop_frequency=10e6, number_of_points=10, variation='dec')
    gain = np.array(analysis_ac["vd_r"])

    axe = plt.subplot(211)
    axe.grid(True)
    axe.set_xlabel("Frequency [Hz]")
    axe.set_ylabel("dB gain.")
    axe.semilogx(analysis_ac.frequency, 20*np.log10(np.abs(gain)))

    axe = plt.subplot(212)
    axe.grid(True)
    axe.set_xlabel("Frequency [Hz]")
    axe.set_ylabel("Phase.")
    axe.semilogx(analysis_ac.frequency, np.arctan2(gain.imag, gain.real)/np.pi*180)
    plt.show()


def run_trans_sine(circuit, vdd=1.2, inp=0.6, inn=0.6, amp=0.3, freq=1e3):
    circuit.V(1, "vdd", circuit.gnd, vdd)
    circuit.SinusoidalVoltageSource(2, "inp", circuit.gnd, offset=inp, amplitude=amp, frequency=freq)
    circuit.V(4, "inn", circuit.gnd, inn)
    simulator = circuit.simulator()
    analysis_trans = simulator.transient(step_time=0.5e-6, end_time=10e-3)

    plt.grid(True)
    plt.title('Transient')
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage [V]')
    plt.plot(analysis_trans['inp'])
    plt.plot(analysis_trans['vd_r'])
    plt.legend(('input', 'output'), loc=(.05, .1))
    plt.ylim(0, vdd)
    plt.show()


def run_trans_pwl(circuit, tend, tstep, vdd=[(0, 1.2)], inp=[(0, 0.6)], inn=[(0, 0.6)], plot=True):
    circuit.PieceWiseLinearVoltageSource(1, "vdd", circuit.gnd, values=vdd)
    circuit.PieceWiseLinearVoltageSource(2, "inp", circuit.gnd, values=inp)
    circuit.PieceWiseLinearVoltageSource(4, "inn", circuit.gnd, values=inn)
    simulator = circuit.simulator()
    analysis_trans = simulator.transient(step_time=tstep, end_time=tend-tstep)

    t_array = analysis_trans._time
    vdd_array = analysis_trans['vdd']
    inp_array = analysis_trans['inp']
    inn_array = analysis_trans['inn']
    out_array = analysis_trans['vd_r']

    if plot:
        axe = plt.subplot(311)
        plt.title('Transient')
        axe.grid(True)
        axe.plot(t_array, vdd_array)
        axe.set_ylim(0, 1.2)
        axe = plt.subplot(312)
        axe.plot(t_array, inp_array)
        axe.plot(t_array, inn_array)
        axe.legend(('inp', 'inn'), loc=(.05, .1))
        axe.set_ylabel('Voltage [V]')
        axe = plt.subplot(313)
        axe.plot(t_array, out_array)
        axe.set_xlabel('Time [s]')
        plt.show()
    return np.array([t_array, vdd_array, inp_array, inn_array, out_array])

#run_dc(circuit, vdd=1.2, inp=0.6, inn=0.6)
#run_ac(circuit, vdd=1.2, inp=0.6, inn=0.6)
#run_trans_sine(circuit, vdd=1.2, inp=0.6, inn=0.6, amp=0.3, freq=3e3)
#run_trans_pwl(circuit, tend = 1e-3, tstep=1e-6, vdd=[(0, 1.2), (1e-3, 0.4)], inp=[(0, 1.2), (1e-3, 0.4)], inn=[(0, 1.2), (1e-3, 0.4)])


def pwl_gen(vmin, vmax, tmax, tstep, seed):
    pwl = []
    np.random.seed(seed)
    for t in np.arange(start=0, stop=tmax, step=tstep):
        print(t)
        pwl = pwl + [(t, np.random.randint(low=vmin*100, high=vmax*100)/100)]
    return pwl


tmax = 1e-3
num_step = 100
tstep = tmax / num_step
circuit_test_data = opamp("diff_amp")
vdd = pwl_gen(vmin=0, vmax=1.2, tmax=tmax, tstep=tstep, seed=1)
inp = pwl_gen(vmin=0, vmax=1.2, tmax=tmax, tstep=tstep, seed=2)
inn = pwl_gen(vmin=0, vmax=1.2, tmax=tmax, tstep=tstep, seed=3)
sim_data = run_trans_pwl(circuit_test_data, tend=1e-3, tstep=tstep, vdd=vdd, inp=inp, inn=inn, plot=False)
np.savetxt("sim_data_test.csv", sim_data.T, delimiter=',', header="time, vdd, inp, inn, out")
circuit_train_data = opamp("diff_amp")
vdd = pwl_gen(vmin=0, vmax=1.2, tmax=tmax, tstep=tstep, seed=4)
inp = pwl_gen(vmin=0, vmax=1.2, tmax=tmax, tstep=tstep, seed=5)
inn = pwl_gen(vmin=0, vmax=1.2, tmax=tmax, tstep=tstep, seed=6)
sim_data = run_trans_pwl(circuit_train_data, tend=1e-3, tstep=tstep, vdd=vdd, inp=inp, inn=inn, plot=False)
np.savetxt("sim_data_train.csv", sim_data.T, delimiter=',', header="time, vdd, inp, inn, out")