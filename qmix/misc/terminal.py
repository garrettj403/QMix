""" This sub-module contains functions for printing numbers and text to the 
terminal.

"""

import os


def print_intro():  # pragma: no cover
    """ Print a quick intro to the terminal.

    """

    os.system('clear')
    print("\nWelcome to")
    art = """
      ,ad8888ba,    88b           d88  88
     d8"'    `"8b   888b         d888  ""
    d8'        `8b  88`8b       d8'88
    88          88  88 `8b     d8' 88  88  8b,     ,d8
    88          88  88  `8b   d8'  88  88   `Y8, ,8P'
    Y8,    "88,,8P  88   `8b d8'   88  88     )888(
     Y8a.    Y88P   88    `888'    88  88   ,d8" "8b,
      `"Y8888Y"Y8a  88     `8'     88  88  8P'     `Y8
    """
    intro = """
            *** SIS Mixer Analysis Software ***
    """
    cprint(art, color='MAGENTA')
    print(intro)


def cprint(text, color='HEADER'):  # pragma: no cover
    """ Print colored text to the terminal.

    Args:
        text (str): Text to print
        color (str): Color/style to print in

    """

    if color is None:
        print(text)
    else:
        try:
            tcolour = _terminal_colors[color.upper()]
            print((tcolour + text + _terminal_colors['ENDC']))
        except KeyError:
            print("\'color\' must be one of:")
            print((sorted(_terminal_colors.keys())))
            print(text, color)
            return


# Titles ---------------------------------------------------------------------

def title(title_string, color=None, total_len=60):  # pragma: no cover
    """ Print a nice title to the terminal.

    Args:
        title_string (str): title to print
        color (str): Color to print in
        total_len (int): Total length of title string (including stars)

    """

    minus_title = total_len - len(title_string) - 2
    if minus_title % 2 == 0:
        left = minus_title // 2
        right = minus_title // 2
    else:
        left = minus_title // 2
        right = minus_title // 2 + 1

    title_string = " " + title_string + " "
    title_string = "*" * left + title_string + "*" * right + "\n"

    cprint(title_string, color)


def header(header_string, color=None):  # pragma: no cover
    """ Print a nice header to the terminal.

    Args:
        header_string (str): Header title to print
        color (str): Color to print in

    """

    cprint("\t" + header_string + ":", color)
    cprint("\t" + "-" * 50)


# Print numbers to the terminal in a nice way --------------------------------

def pvalf(name, val, units='', comment='', color=None):  # pragma: no cover
    """ Print name, value as float, and units to terminal.

    Args:
        name (str): variable name
        val (float): variable value
        units (str): variable units (optional)
        comment (str): comment (optional)

    """

    if units != '':
        units = '\t[' + units + ']'
    if comment != '':
        comment = "  # {0}".format(comment)

    if isinstance(val, complex):
        re = val.real
        im = val.imag
        if val.imag < 0:
            str_tmp = "\t{0:15s} = {1:7.3f} - j{2:7.3f}{3:15s}{4}"
        else:
            str_tmp = "\t{0:15s} = {1:7.3f} + j{2:7.3f}{3:15s}{4}"
        cprint(str_tmp.format(name, re, abs(im), units, comment), color)

    else:
        str_tmp = "\t{0:15s} = {1:7.3f}\t{2:15s}{3}"
        cprint(str_tmp.format(name, val, units, comment), color)


def pvale(name, val, units='', comment='', color=None):  # pragma: no cover
    """ Print name, value in scientific notation and units to terminal.

    Args:
        name (str): variable name
        val (float): variable value
        units (str): variable units (optional)
        comment (str): comment (optional)

    """

    if units != '':
        if isinstance(val, complex):
            units = '\t[' + units + ']'
        else:
            units = '\t\t[' + units + ']'
    if comment != '':
        comment = "\t\t# {0}".format(comment)

    str_tmp = "\t{0:15s} = {1:7.1e}{2:15s}{3}"
    cprint(str_tmp.format(name, val, units, comment), color)


# Color dictionary for the terminal ------------------------------------------

_terminal_colors = {
    # colours
    'CYAN': '\033[36m',
    'MAGENTA': '\033[35m',
    'PINK': '\033[95m',
    'BLUE': '\033[94m',
    'GREEN': '\033[92m',
    'YELLOW': '\033[93m',
    'RED': '\033[91m',
    # structured colours
    'HEADER': '\033[95m',  # pink
    'HEADER1': '\033[95m',  # pink
    'HEADER2': '\033[36m',  # cyan
    'HEADER3': '\033[92m',  # green
    'OKBLUE': '\033[94m',  # blue
    'OKGREEN': '\033[92m',  # green
    'WARNING': '\033[93m',  # yellow
    'FAIL': '\033[91m',  # red
    # other
    'BOLD': '\033[1m',
    'UNDERLINE': '\033[4m',
    'INVERSE': '\033[7m',
    'BLINK': '\033[5m',
    # end of color
    'ENDC': '\033[0m'
}


# Print complex --------------------------------------------------------------

def printc(complex_number):  # pragma: no cover
    """Print a complex number to the terminal.

    Args:
        complex_number (complex): number to print

    """

    if complex_number.imag >= 0:
        sign = '+'
    else:
        sign = '-'

    return str("{:+6.2f} {} j{:5.2f}".format(complex_number.real, sign,
                                             complex_number.imag))
