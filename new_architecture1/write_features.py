import os
os.environ['OMP_NUM_THREADS'] = '1'
from simulation_loader import SimulationLoader
from multiprocessing import Pool

def f(i):
    d = SimulationLoader(i)
    d.load_features_from_h5()
    d.save_feature_arrays()


with Pool(10) as p:
    p.map(f, list(range(40)))
