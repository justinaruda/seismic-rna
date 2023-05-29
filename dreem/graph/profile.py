from click import command
import os
from pathlib import Path

from ..core import docdef
from ..core.cli import opt_table, opt_max_procs, opt_parallel
from ..core.parallel import dispatch
from ..table.io import load


params = [
    opt_table,
    opt_max_procs,
    opt_parallel,
]


@command(os.path.splitext(__name__)[-1], params=params)
def cli(*args, **kwargs):
    """ Draw a bar graph of a positional attribute. """
    return run(*args, **kwargs)


@docdef.auto()
def run(table: tuple[str, ...],
        yaxis: tuple[str, ...], *,
        max_procs: int,
        parallel: bool) -> list[Path]:
    """ Run the graph profile module. """
