import numpy as np

from ..array import ensure_order, ensure_same_length
from ..types import fit_uint_type

# Indexes of read end coordinates.
END5_COORD = "5' End"
END3_COORD = "3' End"
END_COORDS = END5_COORD, END3_COORD


def count_reads_segments(seg_ends: np.ndarray,
                         what: str = "seg_ends") -> tuple[int, int]:
    if not isinstance(seg_ends, np.ndarray):
        raise TypeError(
            f"{what} must be ndarray, but got {type(seg_ends).__name__}"
        )
    if seg_ends.ndim != 2:
        raise ValueError(
            f"{what} must have 2 dimensions, but got {seg_ends.ndim}"
        )
    num_reads, num_segs = seg_ends.shape
    if num_segs == 0:
        raise ValueError(f"{what} has 0 segments")
    return num_reads, num_segs


def match_reads_segments(seg_end5s: np.ndarray, seg_end3s: np.ndarray):
    """ Number of segments for the given end coordinates. """
    num_reads_5, num_segs_5 = count_reads_segments(seg_end5s, "seg_end5s")
    num_reads_3, num_segs_3 = count_reads_segments(seg_end3s, "seg_end3s")
    if num_reads_5 != num_reads_3:
        raise ValueError("Numbers of 5' and 3' reads must equal, "
                         f"but got {num_reads_5} ≠ {num_reads_3}")
    if num_segs_5 != num_segs_3:
        raise ValueError("Numbers of 5' and 3' segments must equal, "
                         f"but got {num_segs_5} ≠ {num_segs_3}")
    return num_reads_5, num_segs_5


def mask_segment_ends(seg_end5s: np.ndarray, seg_end3s: np.ndarray):
    """ Mask segments with no coverage (5' end > 3' end). """
    match_reads_segments(seg_end5s, seg_end3s)
    no_coverage = seg_end5s > seg_end3s
    if np.any(no_coverage):
        seg_end5s = np.ma.masked_array(seg_end5s, no_coverage)
        seg_end3s = np.ma.masked_array(seg_end3s, no_coverage)
    return seg_end5s, seg_end3s


def merge_segment_ends(seg_end5s: np.ndarray,
                       seg_end3s: np.ndarray,
                       fill_value: int | None = None):
    """ """
    match_reads_segments(seg_end5s, seg_end3s)
    if np.ma.isarray(seg_end5s) or np.ma.isarray(seg_end3s):
        seg_ends = np.ma.hstack([seg_end5s, seg_end3s])
        if fill_value is not None:
            seg_ends.fill_value = 0
    else:
        seg_ends = np.hstack([seg_end5s, seg_end3s])
    return seg_ends


def _check_no_coverage_reads(seg_ends: np.ndarray, what: str = "seg_ends"):
    num_reads, num_segs = count_reads_segments(seg_ends, what)
    if np.ma.isarray(seg_ends):
        # Check if any reads have no coverage (are completely masked).
        no_coverage = seg_ends.mask.all(axis=1)
        if np.any(no_coverage):
            raise ValueError(
                f"Got {np.count_nonzero(no_coverage)} read(s) with no coverage "
            )
    return num_reads, num_segs


def find_read_end5s(seg_end5s: np.ndarray):
    """ Find the 5' end of each read. """
    num_reads, num_segs = _check_no_coverage_reads(seg_end5s, "seg_end5s")
    return np.asarray(seg_end5s.min(axis=1)
                      if num_segs > 1
                      else seg_end5s[:, 0])


def find_read_end3s(seg_end3s: np.ndarray):
    """ Find the 3' end of each read. """
    num_reads, num_segs = _check_no_coverage_reads(seg_end3s, "seg_end3s")
    return np.asarray(seg_end3s.max(axis=1)
                      if num_segs > 1
                      else seg_end3s[:, 0])


def merge_read_ends(read_end5s: np.ndarray, read_end3s: np.ndarray):
    """ Return the 5' and 3' ends as one 2D array. """
    ensure_same_length(read_end5s, read_end3s, "read_end5s", "read_end3s")
    return np.column_stack([read_end5s, read_end3s])


def sanitize_segment_ends(seg_end5s: np.ndarray,
                          seg_end3s: np.ndarray,
                          min_pos: int,
                          max_pos: int,
                          check_values: bool = True):
    """ Sanitize end coordinates.

    Parameters
    ----------
    seg_end5s: np.ndarray
        5' end coordinate of each segment in each read.
    seg_end3s: np.ndarray
        3' end coordinate of each segment in each read.
    min_pos: int
        Minimum allowed value of a position.
    max_pos: int
        Maximum allowed value of a position.
    check_values: bool = True
        Whether to check the bounds of the values, which is the most
        expensive operation in this function. Can be set to False if the
        only desired effect is to ensure the output is a positive, even
        number of arrays in the proper data type.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Sanitized end coordinates: encoded in the most efficient data
        type, and if `check_values` is True then all between `min_pos`
        and `max_pos` (inclusive).
    """
    num_reads, num_segs = match_reads_segments(seg_end5s, seg_end3s)
    if check_values:
        for i in range(num_segs):
            # All coordinates must be ≥ 1.
            ensure_order(seg_end5s[:, i],
                         np.broadcast_to(min_pos, num_reads),
                         f"segment {i} 5' coordinates",
                         f"minimum position ({min_pos})",
                         gt_eq=True)
            ensure_order(seg_end3s[:, i],
                         np.broadcast_to(min_pos, num_reads),
                         f"segment {i} 3' coordinates",
                         f"minimum position ({min_pos})",
                         gt_eq=True)
            # All coordinates must be ≤ max position
            ensure_order(seg_end5s[:, i],
                         np.broadcast_to(max_pos, num_reads),
                         f"segment {i} 5' coordinates",
                         f"maximum position ({max_pos})")
            ensure_order(seg_end3s[:, i],
                         np.broadcast_to(max_pos, num_reads),
                         f"segment {i} 3' coordinates",
                         f"maximum position ({max_pos})")
    # Convert all ends into arrays of the most efficient data type.
    # Conversion must be done after checking the values against max_pos.
    # Otherwise, values greater than the data type can accomodate would
    # be converted to their modulo relative to the maximum value of the
    # data type, which would mean checking the wrong values.
    dtype = fit_uint_type(max_pos)
    return np.asanyarray(seg_end5s, dtype), np.asanyarray(seg_end3s, dtype)


def sort_segment_ends(seg_end5s: np.ndarray,
                      seg_end3s: np.ndarray,
                      one_indexed: bool = True,
                      fill_mask: bool = False):
    """ Sort the segment end coordinates and label the 3' end of each
    contiguous set of segments.

    Parameters
    ----------
    seg_end5s: np.ndarray
        5' end of each segment in each read; may be masked.
    seg_end3s: np.ndarray
        3' end of each segment in each read; may be masked.
    one_indexed: bool = True
        In the return array, make the 5' ends 1-indexed; if False, then
        they will be 0-indexed.
    fill_mask: bool = False
        If `seg_end5s` or `seg_end3s` is a masked array, then return a
        regular array with all masked coordinates set to 0 (or 1 for
        5' ends if `one_indexed` is True) rather than a masked array.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - Sorted 5' and 3' coordinates of the segments in each read
        - Labels of whether each coordinate is a 5' end of a segment
        - Labels of whether each coordinate is a 3' end of a contiguous
          segment
    """
    _, num_segs = match_reads_segments(seg_end5s, seg_end3s)
    # Make the 5' ends 0-indexed and join the coordinates horizontally.
    seg_ends = merge_segment_ends(seg_end5s - 1, seg_end3s)
    # Sort the coordinates in ascending order within each read.
    kwargs = dict(fill_value=0) if np.ma.isarray(seg_ends) else dict()
    sort_order = seg_ends.argsort(axis=1, **kwargs)
    if np.ma.isarray(seg_ends) and fill_mask:
        # Fill the masked values with zero and remove the mask.
        mask = seg_ends.mask
        seg_ends = seg_ends.data
        seg_ends[mask] = 0
    seg_ends_sorted = np.take_along_axis(seg_ends, sort_order, axis=1)
    # Label the sorted coordinates that correspond to segment 5' ends.
    is_seg_end5 = sort_order < num_segs
    # Label the 3' end of each contiguous set of segments, which occurs
    # when the cumulative numbers of 5' and 3' segment ends are equal.
    is_contig_end3 = np.logical_not(np.cumsum(np.where(is_seg_end5, 1, -1),
                                              axis=1))

    if np.ma.isarray(seg_ends_sorted):
        # Mask is_seg_end5 and is_contig_end3 like seg_ends_sorted.
        is_seg_end5 = np.ma.masked_array(is_seg_end5,
                                         seg_ends_sorted.mask)
        is_contig_end3 = np.ma.masked_array(is_contig_end3,
                                            seg_ends_sorted.mask)
    if one_indexed:
        # Make the 5' coordinates 1-indexed again.
        seg_ends_sorted[np.nonzero(is_seg_end5)] += 1
    return seg_ends_sorted, is_seg_end5, is_contig_end3


def find_contiguous_reads(seg_end5s: np.ndarray, seg_end3s: np.ndarray):
    """ Whether the segments of each read are contiguous. """
    num_reads, num_segs = match_reads_segments(seg_end5s, seg_end3s)
    if num_segs == 1:
        # Single-segment reads must be contiguous.
        return np.ones(num_reads, dtype=bool)
    # For contiguous reads, only the last coordinate (when sorted) will
    # be the end of a contiguous segment.
    _, _, is_contig_end3 = sort_segment_ends(seg_end5s, seg_end3s)
    return np.logical_not(np.count_nonzero(is_contig_end3[:, :-1], axis=1))

########################################################################
#                                                                      #
# © Copyright 2024, the Rouskin Lab.                                   #
#                                                                      #
# This file is part of SEISMIC-RNA.                                    #
#                                                                      #
# SEISMIC-RNA is free software; you can redistribute it and/or modify  #
# it under the terms of the GNU General Public License as published by #
# the Free Software Foundation; either version 3 of the License, or    #
# (at your option) any later version.                                  #
#                                                                      #
# SEISMIC-RNA is distributed in the hope that it will be useful, but   #
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANT- #
# ABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General     #
# Public License for more details.                                     #
#                                                                      #
# You should have received a copy of the GNU General Public License    #
# along with SEISMIC-RNA; if not, see <https://www.gnu.org/licenses>.  #
#                                                                      #
########################################################################
