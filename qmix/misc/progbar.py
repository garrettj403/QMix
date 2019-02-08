""" This sub-module contains a function for printing a progress bar to the 
terminal.

Taken from:

   http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console

"""

import sys


def progress_bar(iteration, total, prefix='Progress:', suffix='Complete',
                 decimals=2, bar_length=20):
    """Generate a progress bar.

    Args:
        iteration (int): current iteration
        total (int): total iterations

    Keyword Args:
        prefix (str): prefix string
        suffix (str): suffix string
        decimals (int): number of decimals in percent complete
        bar_length (int): character length of bar

    """

    filled_length = int(round(bar_length * iteration / float(total)))
    pcent = round(100.00 * (iteration / float(total)), decimals)
    pbar = '-' * filled_length + ' ' * (bar_length - filled_length)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, pbar, pcent, '%', suffix)),
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()
