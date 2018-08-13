"""Print a progress bar to the console.

From:

   http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console

"""

import sys


def progress_bar(iteration, total, prefix='Progress:', suffix='Complete',
                 decimals=2, bar_length=20):
    """Create progress bar.

    Args:
        iteration (int): current iteration
        total (int): total iterations
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


# # Run -----------------------------------------------------------------------
#
# def _main():  # pragma: no cover
#
#     from time import sleep
#
#     print "\nTesting progbar.py ...\n"
#
#     items = list(range(0, 100))
#     i = 0
#     l = len(items)
#
#     # Initial call to print 0% progress
#     progress_bar(i, l, prefix='Progress:', suffix='Complete', bar_length=50)
#     for _ in items:
#         sleep(0.1)
#         i += 1
#         progress_bar(i, l, prefix='Progress:', suffix='Complete',
#                      bar_length=50)
#     print ""
#
#
# if __name__ == "__main__":  # pragma: no cover
#     _main()
