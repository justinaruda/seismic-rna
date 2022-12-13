
import pytest
import pandas as pd
import dreem.util as util
import os
from dreem import clustering
from dreem.test import files_generator
from dreem.test.files_generator import test_files_dir, input_dir, prediction_dir, output_dir
import json 

module = 'clustering'

sample_name = 'test_set_1'
mode = 'light' # only uses the first sample

reads_partition = [
    [5000], [10000], [20000],
    [25000, 25000], [20000,80000], [10000,90000],
    [10000,20000,70000], [33333,33333,33334], [20000,40000,40000] 
]

half_sequence_length = [50, 100, 150]

unpaired_bases = {k:[int(k*l) for l in  [0.2, 0.4, 0.6]] for k in half_sequence_length}

shared_bases = {}
for k, v in unpaired_bases.items():
    shared_bases[k] = []
    for l in v:
        shared_bases[k].append([int(u*l) for u in  [0.2, 0.4, 0.6]])
        
sample_profile = {}
for r in reads_partition:
    for ac in half_sequence_length:
        for uc in unpaired_bases[ac]:
            for sc in shared_bases[ac][unpaired_bases[ac].index(uc)]:
                sample_profile['r{}_sl{}_ub{}_sb{}'.format(r, ac, uc, sc)] = {
                    'n_reads': r,
                    'n_AC': ac,
                    'n_unpaired': uc,
                    'n_shared': sc,
                    'path_bv': os.path.join(test_files_dir, 'input', module, sample_name, 'r{}_sl{}_ub{}_sb{}.orc'.format(r, ac, uc, sc)),
                    'path_json': os.path.join(test_files_dir, 'output', module, sample_name, 'r{}_sl{}_ub{}_sb{}'.format(r, ac, uc, sc))
                }

if mode == 'light':
    sample_profile = {list(sample_profile.keys())[15]:sample_profile[list(sample_profile.keys())[15]]}

module_input = os.path.join(input_dir, module)
module_predicted = os.path.join(prediction_dir, module)
module_output =  os.path.join(output_dir, module)

inputs = ['clustering']

def test_make_files():
    os.makedirs(os.path.join(test_files_dir, 'input', module, sample_name), exist_ok=True)
    os.makedirs(os.path.join(test_files_dir, 'predicted_output', module, sample_name), exist_ok=True)
    files_generator.generate_files(sample_profile, module, inputs, [], test_files_dir, sample_name)
    files_generator.assert_files_exist(sample_profile, module, inputs, input_dir, sample_name)
    
def test_run():
    for sample in os.listdir(module_input):
        clustering.run(
            input_dir =module_input,
            out_dir = module_output
            )

