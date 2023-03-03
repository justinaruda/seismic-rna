# DREEM Aggregate Module
Contributor: Yves Martin

## Purpose
The bitvectors are turned into a json file that will contain the key information for the bitvector, plus additional information such as per-sample content, per-reference content, RNAstructure prediction.

- If library.csv is an input, splits the reads into sections
- If clustering.json is an input, splits the reads into clusters
- Aggregates the bitvector by counting for each reference
  - number of mutations per residue across reads
  - histogram of number of mutations per read
  - base coverages per residue across reads
  - insertions per residue across reads
  - deletions per residue across reads
  - ...
- Adds the content of `samples.csv`.
- Adds the content of `library.csv`.
- Adds RNAstructure prediction.

## Interface

### Input Files
- [≥1] `/{input_dir}`. A directory containing mutation vector stored in Apache ORC format.
```bash
{input_dir}:= output/vectoring/{sample_1}/
  |- report.txt
  |- reference_1/
     |- section_1.orc
     |- section_2.orc
     |- ...
  |- reference_2.orc
    |- ...
  |- ...
{input_dir}:= output/vectoring/{sample_2}/
  |- ..
```
- [≤1] `clustering.json`. JSON file containing the clustering likelihood for each read of each bitvector.
- [=1] `samples.csv`. CSV file containing per-sample data.
- [=1] `library.csv`. CSV file containing per-reference data.

### Output Files
```
/{out_dir}
  |- {sample_1}.json
  |- {sample_2}.json
```

### Command-line usage

```dreem-aggregate -id [/path/to/{sample_k}] —fa {reference}.fasta  --out_dir [path] —per_mp_file [True/False]```

- ```dreem-aggregate```: Wrapper for ```run``` function in ```dreem/aggregate/run.py```. 

- [≥1] ```-id / --input_dir```: Sequence alignment map files folder(s) generated by ```alignment```.
- [=1] ```--out_dir / -o```: name of the output directory.
- [≤1] ```--clusters / -cl```: `clustering.json`: Path to the clustering.json file
- [=1] ```--samples / -s`: `samples.csv``: Path to the samples.csv file
- [≤1] ```--library / -l`: `library.csv```: Path to the library.csv file
- [=1] ```--rnastructure_temperature / -rst```: Use sample.csv temperature values for RNAstructure or not.
- [=1] ```--rnastructure_fold_args / -rsa```: Arguments to pass to RNAstructure fold
- [=1] ```--rnastructure_dms / -rsd```: Use the DMS signal to make predictions with RNAstructure
- [=1] ```--rnastructure_dms_min_unpaired_value / -rsdmin```: Minimum unpaired value for using the dms signal as an input for RNAstructure
- [=1] ```--rnastructure_fold_args / -rsa```: Maximum paired value for using the dms signal as an input for RNAstructure
- [=1] ```--rnastructure_partition / -rspa```: Use RNAstructure partition function to predict free energy
- [=1] ```--rnastructure_probability / -rspr```: Use RNAstructure partition function to predict per-base mutation probability

## Output format by source module
### Alignment

| attribute | type | description | comment |
| --- | --- | --- | --- |
| num_aligned | int | Number of reads for this mutation profile |  |
| num_aligned | int | Number of aligned reads used for the next steps |  |

### Vectoring

| attribute | type | description | comment |
| --- | --- | --- | --- |

### Clustering

| attribute | type | description | comment |
| --- | --- | --- | --- |
| cluster | str | alternative mutational profiles given by DREEM | default: ‘pop_avg’ |
| cluster_weight | float | Weigth for the EM cluster |  |
| cluster_quality | float | Our quality metric for clsutering quality | float 0 → 1 |
| cluster_reads_used | int | Number of used reads |  |
| cluster_unique_reads | int | Number of unique reads |  |
| cluster_reads_del_all | int | Number of reads removed in total |  |
| cluster_reads_del_too_many_muts | int | Number of reads removed because of too many mutations |  |
| cluster_reads_del_too_few_info_bits | int | Number of reads removed because of too few informative bits |  |
| cluster_reads_del_mut_close_by | int | Number of reads removed because of mutations close by |  |
| cluster_reads_del_no_info_mut | int | Number of reads removed because of no info around mutations |  |

### Aggregate

| attribute | type | description | comment |
| --- | --- | --- | --- |
| sequence | str | nucleotides sequence | uses A, C, G, T |
| sub_hist | str(list(int)) | Count of mutations per read | useful? |
| sub_N | str(list(int)) | Per-residue count of mutations | 0-indexed |
| cov | str(list(int)) | Per-residue count of covered bases |  |
| del | str(list(int)) | Per-residue count of deleted bases | useful? |
| ins | str(list(int)) | Per-residue count of inserted bases | useful? |
| sub_A | str(list(int)) | Per-residue count of mutations to a A base |  |
| sub_C | str(list(int)) | Per-residue count of mutations to a C base |  |
| sub_G | str(list(int)) | Per-residue count of mutations to a G base |  |
| sub_T | str(list(int)) | Per-residue count of mutations to a T base |  |
| sub_rate | str(list(float)) | Per-residue count of mutation divided by the count of valid reads | sub_N/info, shall we use cov instead? |
| min_cov | int | min(info) (or cov?) | to adapt to per-section and per-cluster samples |
| skips_short_reads | int | number of reads that we don’t use because they are too short. | useful? |
| skips_too_many_muts | int | number of reads that that we don’t use because they have so many mutations, and therefore we have low confidence. | useful? |
| skips_low_mapq | int | number of reads that that we don’t use because the map score is too low (default is below 15) | useful? |

## Formats

### The standard format for `samples.csv` is:

| attribute | type | description | comment |
| --- | --- | --- | --- |
| sample | str | fastq file prefix | same as the containing folders names |
| user | str | Who did the experiment |  |
| date | str | Date of the experiment |  |
| exp_env | str | Experimental environment, “in_vivo” or “in_vitro” | Can only be one of the two options “in_vivo” and “in_vitro” |
| temperature_k | float | Temperature in Kelvin |  |
| inc_time_tot_secs | float | Total incubation time in seconds |  |
| buffer | str | Exact buffer including Mg, eg 300mM Sodium Cacodylate , 3mM Mg | Only if exp_env == “in_vitro” |
| cell_line | str | Cell line | Only if exp_env == “in_vivo” |

### The standard format for `library.csv` is:

| attribute | type | description | comment |
| --- | --- | --- | --- |
| reference | str | fasta file references names | cannot be int |
| section | str | Optional name for the section (i.e a subsequence in the sequence). In not and section start/end isn’t the sequence’s boundaries, name will be “section_start”-”section_stop”. Else, default is “full”. | If no “section free” row for a reference, the full reference won’t be included |
| section_start | int | 0-index start index of the section |  |
| section_end | int | 0-index end index of the section |  |
| barcode | str | barcode sequence |  |
| barcode_start | int | 0-index start index of the barcode |  |
| barcode_end | int | 0-index end index of the barcode |  |
| [attribute] | [type] | a per-reference attribute | can be family or flank for example |

### The output from RNAstructure are:

| attribute | type | description | comment |
| --- | --- | --- | --- |
| deltaG | float | minimum energy for the sequence |  |
| deltaG_T | float | minimum energy for the sequence using temperature | optional (flag —rnastructure_temperature) |
| deltaG_DMS | float | minimum energy for the sequence and DMS signal | optional (flag —rnastructure_dms) |
| deltaG_T_DMS | float | minimum energy for the sequence using temperature and DMS signal | optional (flag —rnastructure_temperature -rnastructure_dms) |
| structure | str | minimum energy structure for the sequence |  |
| structure_T | str | structure for the sequence using temperature as an input | optional (flag —rnastructure_temperature) |
| structure_DMS | str | energy structure for the sequence using DMS as an input | optional (flag —rnastructure_dms) |
| structure_T_DMS | str | energy structure for the sequence using temperature and DMS as an input | optional (flag —rnastructure_temperature -rnastructure_dms) |
| deltaG_ens | float | average energy of the partition function for this sequence | optional (flag —rnastructure_partition) |
| mut_probability | str(list(float)) | base-pairing prediction for each residue using the partition function | optional (flag —rnastructure_probability) |