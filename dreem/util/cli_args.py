import dreem.util.util as util
import click
import os
from click_option_group import optgroup

# Input files
OUT_DIR = os.getcwd()
SAMPLE = 'unnamed_sample'
FASTA = ""
FASTQ1 = ""
FASTQ2 = ""
INTERLEAVED = False
INPUT_DIR = '.'
SAMPLES = None
CLUSTERING_FILE = None
LIBRARY = 'library.csv'

out_dir = optgroup.option('--out_dir', '-o', default=OUT_DIR, type=click.Path(exists=True), help='Where to output files')
sample = optgroup.option('--sample', '-s', default=SAMPLE, type=click.Path(exists=True), help='Name of the sequence alignment map file(s) folder')
fasta = optgroup.option('--fasta', '-fa', default=FASTA, type=click.Path(exists=True), help='Path to the fasta file')
fastq1 = optgroup.option('--fastq1', '-fq1', default=FASTQ1, help='Paths to the fastq1 file (forward primer). Enter multiple times for multiple files', type=click.Path(exists=True))
fastq2 = optgroup.option('--fastq2', '-fq2', default=FASTQ2, help='Paths to the fastq2 file (reverse primer). Enter multiple times for multiple files', type=click.Path(exists=True))
input_dir = optgroup.option('--input_dir', '-id', default=INPUT_DIR, type=click.Path(exists=True), help='Sequence alignment map files folder(s) generated by alignment')
samples = optgroup.option('--samples', '-s', default=SAMPLES, type=click.Path(exists=True), help='Path to the samples.csv file')
clustering_file = optgroup.option('--clusters', '-cl', default=CLUSTERING_FILE, type=click.Path(exists=True), help='Path to the clustering.json file')
library = optgroup.option('--library', '-l', default=LIBRARY, type=click.Path(exists=True), help='Path to the library.csv file')
interleaved = optgroup.option('--interleaved', '-i', default=INTERLEAVED, type=bool, help='Fastq files are interleaved')

# Construct selection
COORDS = []
PRIMERS = []
FILL = False
PARALLEL = 'auto'

coords = optgroup.option('--coords', '-c', type=(str, int, int), multiple=True, help="coordinates for reference: '-c ref-name first last'", default=COORDS)
primers = optgroup.option('--primers', '-p', type=(str, str, str), multiple=True, help="primers for reference: '-p ref-name fwd rev'", default=PRIMERS)
fill = optgroup.option('--fill/--no-fill', type=bool, default=FILL, help="Fill in coordinates of reference sequences for which neither coordinates nor primers were given (default: no).")
parallel = optgroup.option('--parallel', '-P', type=click.Choice(["profiles", "reads", "off", "auto"], case_sensitive=False), default=PARALLEL, help="Parallelize the processing of mutational PROFILES or READS within each profile, turn parallelization OFF, or AUTO matically choose the parallelization method (default: auto).")


# Demultiplexing
DEMULTIPLEXING = False
BARCODE_START = None
BARCODE_LENGTH = None
MAX_BARCODE_MISMATCHES = 1

demultiplexing = optgroup.option('--demultiplexing', '-dx', type=bool, help='Use demultiplexing', default=DEMULTIPLEXING)
barcode_start = optgroup.option('--barcode_start', '-bs', type=int, help='Start position of the barcode in the read', default=BARCODE_START)
barcode_length = optgroup.option('--barcode_length', '-bl', type=int, help='Length of the barcode', default=BARCODE_LENGTH)
max_barcode_mismatches = optgroup.option('--max_barcode_mismatches', '-mb', type=int, help='Maximum number of mutations on the barcode', default=MAX_BARCODE_MISMATCHES)


# Cutadapt parameters
DEFAULT_MIN_BASE_QUALITY = 25
DEFAULT_ILLUMINA_ADAPTER = "AGATCGGAAGAGC"
DEFAULT_MIN_OVERLAP = 6
DEFAULT_MAX_ERROR = 0.1
DEFAULT_INDELS = True
DEFAULT_NEXTSEQ = True
DEFAULT_DISCARD_TRIMMED = False
DEFAULT_DISCARD_UNTRIMMED = False
DEFAULT_MIN_LENGTH = 50


# Bowtie 2 parameters
DEFAULT_LOCAL = True
DEFAULT_UNALIGNED = False
DEFAULT_DISCORDANT = False
DEFAULT_MIXED = False
DEFAULT_DOVETAIL = False
DEFAULT_CONTAIN = True
DEFAULT_FRAG_LEN_MIN = 0
DEFAULT_FRAG_LEN_MAX = 300  # maximum length of a 150 x 150 read
DEFAULT_N_CEILING = "L,0,0.05"
DEFAULT_SEED_INTERVAL = "L,1,0.1"
DEFAULT_GAP_BAR = 4
DEFAULT_SEED_SIZE = 20
DEFAULT_EXTENSIONS = 5
DEFAULT_RESEED = 1
DEFAULT_PADDING = 4
DEFAULT_ALIGN_THREADS = os.cpu_count()
MATCH_BONUS = "1"
MISMATCH_PENALTY = "1,1"
N_PENALTY = "0"
REF_GAP_PENALTY = "0,1"
READ_GAP_PENALTY = "0,1"
IGNORE_QUALS = True


# Clustering
CLUSTERING = False
MAX_CLUSTERS = 3
SIGNAL_THRESH = 0.005
INFO_THRESH = None
INCLUDE_G_U = False
INCLUDE_DEL = False
MIN_READS = 1000
CONVERGENCE_CUTOFF = 0.5
MIN_ITER = 100
NUM_RUNS = 10
N_CPUS = 2

clustering = optgroup.option('--clustering', '-cl', type=bool, help='Use clustering', default=CLUSTERING)
max_clusters = optgroup.option('--max_clusters', '-mc', type=int, help='Maximum number of clusters', default=MAX_CLUSTERS)
min_iter = optgroup.option('--min_iter', '-mi', type=int, help='Minimal number of EM iterations', default=MIN_ITER)
signal_thresh = optgroup.option('--signal_thresh', '-st', type=float, help='Signal threshold', default=SIGNAL_THRESH)
info_thresh = optgroup.option('--info_thresh', '-it', type=float, help='Information threshold', default=INFO_THRESH)
include_g_u = optgroup.option('--include_g_u', '-igu', type=bool, help='Include G and U', default=INCLUDE_G_U)
include_del = optgroup.option('--include_del', '-id', type=bool, help='Include deletions', default=INCLUDE_DEL)
min_reads = optgroup.option('--min_reads', '-mr', type=int, help='Minimum number of reads', default=MIN_READS)
convergence_cutoff = optgroup.option('--convergence_cutoff', '-cc', type=float, help='Convergence cutoff', default=CONVERGENCE_CUTOFF)
num_runs = optgroup.option('--num_runs', '-nr', type=int, help='Number of runs', default=NUM_RUNS)
n_cpus = optgroup.option('--n_cpus', '-cpu', type=int, help='Number of CPUs', default=N_CPUS)

# Aggregation
RNASTRUCTURE_PATH = None
RNASTRUCTURE_TEMPERATURE = False
RNASTRUCTURE_FOLD_ARGS = None
RNASTRUCTURE_DMS = False
RNASTRUCTURE_DMS_MIN_UNPAIRED_VALUE = 0.04
RNASTRUCTURE_DMS_MAX_PAIRED_VALUE = 0.01
RNASTRUCTURE_PARTITION = False
RNASTRUCTURE_PROBABILITY = False
POISSON = True

rnastructure_path = optgroup.option('--rnastructure_path', '-rs', type=click.Path(exists=True), help='Path to RNAstructure, to predict structure and free energy', default=RNASTRUCTURE_PATH)
rnastructure_temperature = optgroup.option('--rnastructure_temperature', '-rst', type=bool, help='Use sample.csv temperature values for RNAstructure', default=RNASTRUCTURE_TEMPERATURE)
rnastructure_fold_args = optgroup.option('--rnastructure_fold_args', '-rsa', type=str, help='optgroup.options to pass to RNAstructure fold', default=RNASTRUCTURE_FOLD_ARGS)
rnastructure_dms = optgroup.option('--rnastructure_dms', '-rsd', type=bool, help='Use the DMS signal to make predictions with RNAstructure', default=   RNASTRUCTURE_DMS)
rnastructure_dms_min_unpaired_value = optgroup.option('--rnastructure_dms_min_unpaired_value', '-rsdmin', type=int, help='Minimum unpaired value for using the dms signal as an input for RNAstructure', default=RNASTRUCTURE_DMS_MIN_UNPAIRED_VALUE)
rnastructure_dms_max_paired_value = optgroup.option('--rnastructure_dms_max_paired_value', '-rsdmax', type=int, help='Maximum paired value for using the dms signal as an input for RNAstructure', default=RNASTRUCTURE_DMS_MAX_PAIRED_VALUE)
rnastructure_partition = optgroup.option('--rnastructure_partition', '-rspa', type=bool, help='Use RNAstructure partition function to predict free energy', default=RNASTRUCTURE_PARTITION)
rnastructure_probability = optgroup.option('--rnastructure_probability', '-rspr', type=bool, help='Use RNAstructure partition function to predict per-base mutation probability', default=RNASTRUCTURE_PROBABILITY)
poisson = optgroup.option('--poisson', '-po', type=bool, help='Predict Poisson confidence intervals', default=POISSON)

# Misc
VERBOSE = False
verbose = optgroup.option('--verbose', '-v', type=bool, help='Verbose output', default=VERBOSE)

