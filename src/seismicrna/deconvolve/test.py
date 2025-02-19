import os
os.chdir("/n/data1/hms/microbiology/rouskin/lab/projects/justin/a2g")
from pathlib import Path
from seismicrna.core.rel import RelPattern
import numpy as np
from seismicrna.deconvolve.main import run
from seismicrna.deconvolve import main
from seismicrna.core.logs import set_config

mirnas = ["hsa-mir-4731"]
#mirnas = ["hsa-let-7a-3", "hsa-mir-365a", "hsa-mir-577", "hsa-mir-27b", "hsa-mir-4502", "hsa-mir-605"]
deconvolve_reports = list()
edited_reports = list()
background_reports = list()
patterns = list()
positions_list = list()
for mirna in mirnas:
    ag_pattern = RelPattern.from_counts(discount=["ac","at","gt", "gc", "ga", "ta", "tg", "tc", "ca", "ct", "cg"])
    deconvolve_report = Path(f'/n/data1/hms/microbiology/rouskin/lab/projects/justin/ivt_edit_nextseq/out/AD/mask/{mirna}')
    edited_report = Path(f'/n/data1/hms/microbiology/rouskin/lab/projects/justin/ivt_edit_nextseq/out/A/mask/{mirna}')
    background_report = Path(f'/n/data1/hms/microbiology/rouskin/lab/projects/justin/ivt_edit_nextseq/out/D/mask/{mirna}')
    deconvolve_reports.append(deconvolve_report)
    edited_reports.append(edited_report)
    background_reports.append(background_report)
    positions = []
    positions = tuple([tuple(pos) for pos in positions])
    positions_list.append(positions)
    patterns.append(ag_pattern)
set_config(exit_on_error=True)

deconvolve_reports = tuple(deconvolve_reports)
edited_reports = tuple(edited_reports)
background_reports = tuple(background_reports)
patterns = tuple(patterns)
positions_list = tuple(positions_list)

run(deconvolve_reports,
    edited_reports,
    background_reports,
    positions_list,
    patterns,
    conf_thresh=0.8,
    combinations=[2],
    norm_edits=False,
    strict=False,
    n_procs=20,
    deconvolve_pos_table=True,
    deconvolve_abundance_table=True,
    brotli_level=1,
    force=True)
