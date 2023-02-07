from dreem.align import align
from dreem.util.dflt import PHRED_ENCODING


def run(top_dir: str, fasta: str,
        fastqs: str, fastqi: str, fastq1: str, fastq2: str,
        fastqs_dir: str, fastqi_dir: str, fastq12_dir: str,
        phred_enc: int, **kwargs):
    """
    Run the alignment module.

    Align the reads to the set of reference sequences and output one BAM file
    for each sample aligned to each reference in the directory 'output'.
    Temporary intermediary files are written in the directory 'temp' and then
    deleted after they are no longer needed.

    {top_dir}/
        output/
            {sample_1}/
                -| {ref_1}.bam
                -| {ref_2}.bam
                ...
            {sample_2}/
            ...
        temp/
            {step_1}/
                {sample_1}/
                    -| {ref_1}.file
                    -| {ref_2}.file
                    ...
                {sample_2}/
                ...
            {step_2}/
            ...
        ...

    Parameters
    ----------
    top_dir: str
        Path to the top-level folder into which all final outputs and temporary
        files are written. Must already exist.
    fasta: str
        Path to the reference FASTA file.
    fastqs: str †
        Path to a FASTQ file of single-end reads, or '' if none.
    fastqi: str †
        Path to a FASTQ file of interleaved, paired-end reads, or '' if none.
    fastq1: str †
        Path to a FASTQ file of 1st mates of paired-end reads, or '' if none.
        If given, fastq2 must also be given, and its sample name must match.
    fastq2: str †
        Path to a FASTQ file of 2nd mates of paired-end reads, or '' if none.
        If given, fastq1 must also be given, and its sample name must match.
    fastqs_dir: str ‡
        Path to a directory containing demultiplexed FASTQ files of single-end
        end reads from one sample, or '' if none.
    fastqi_dir: str ‡
        Path to a directory containing demultiplexed FASTQ files of interleaved,
        paired-end reads, or '' if none.
    fastq12_dir: str ‡
        Path to a directory containing, for each  FASTQ files of both the 1st and 2nd
        mates of paired-end reads, or '' if none.
    phred_enc: int
        The ASCII encoding offset of the Phred scores. For example, if
        ```phred_enc = 33```, then a Phred score of 30 will be encoded as the
        ASCII character corresponding to 30 + 33 = 63, which is '?'; and in a
        FASTQ or SAM file, the character 'F' (ASCII value 70) denotes a Phred
        score of 70 - 33 = 37. This encoding of 33 is used by most modern
        Illumina sequencers and is the default.
    **kwargs
        Additional keyword arguments to pass to the alignment function.

    NOTES
    † The file name (minus the extension) will be used as the sample name.
    ‡ The file name (minus the extension) will be used as the reference name,
      and the name of the parent directory will be used as the sample name.
    """
    demult_args = fastqs_dir, fastqi_dir, fastq12_dir
    demultiplexed = any(map(bool, demult_args))
    non_demult_args = fastqs, fastqi, fastq1, fastq2
    non_demultiplexed = any(map(bool, non_demult_args))
    if demultiplexed and non_demultiplexed:
        raise ValueError("Both non-demultiplexed FASTQ files "
                         "and demultiplexed FASTQ directories were given.")
    if demultiplexed:
        align.each_ref(top_dir, fasta, fastqs_dir, fastqi_dir, fastq12_dir,
                       phred_enc, **kwargs)
    elif non_demultiplexed:
        align.all_refs(top_dir, fasta, fastqs, fastqi, fastq1, fastq2,
                       phred_enc, **kwargs)
    else:
        raise ValueError("No FASTQ input files were given")
