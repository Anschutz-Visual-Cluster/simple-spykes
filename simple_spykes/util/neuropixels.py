import numpy as np


def get_default_probe_mapping():
    # Code from https://github.com/denmanlab/dlab/blob/9e2cd55f1df68901faca6f702e3f4ede3805d84d/generalephys.py#L59

    option234_xpositions = np.zeros((192, 2))
    option234_ypositions = np.zeros((192, 2))
    option234_positions = np.zeros((384, 2))
    option234_positions[:, 0][::4] = 21
    option234_positions[:, 0][1::4] = 53
    option234_positions[:, 0][2::4] = 5
    option234_positions[:, 0][3::4] = 37
    option234_positions[:, 1] = np.floor(np.linspace(383, 0, 384) / 2) * 20
    # imecp3_image = plt.imread(os.path.join(os.path.dirname(os.path.abspath(maps.__file__)),'imec_p3.png'))

    imec_p2_positions = np.zeros((128, 2))
    imec_p2_positions[:, 0][::2] = 18
    imec_p2_positions[:, 0][1::2] = 48
    imec_p2_positions[:, 1] = np.floor(np.linspace(0, 128, 128) / 2) * 20;
    imec_p2_positions[:, 1][-1] = 1260.

    import numpy as np
    import matplotlib.pyplot as plt

    from probeinterface import Probe
    from probeinterface.plotting import plot_probe

    probe = Probe(ndim=2, si_units='um')
    probe.set_contacts(positions=option234_positions, shapes='circle', shape_params={'radius': 12})
    probe.create_auto_shape()
    plot_probe(probe)
    plt.show()
