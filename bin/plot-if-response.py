#!/usr/bin/env python3
"""Plot IF response."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from qmix.exp.if_response import if_response
from qmix.exp.parameters import params
from qmix.mathfn.filters import gauss_conv

# Grab arguments
parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, nargs='+', help="IF power vs. IF frequency from hot/cold blackbody loads. Data should have 3 columns")
parser.add_argument('-th', '--thot', type=float, help='hot blackbody load temperature')
parser.add_argument('-tc', '--tcold', type=float, help='cold blackbody load temperature')
parser.add_argument('-d', '--delimiter', type=str, help='delimiter used in data file', default=',')
parser.add_argument('-x', '--xmax', type=float, help='max. IF frequency to plot')
parser.add_argument('-y', '--ymax', type=float, help='max. noise temperature to plot')
parser.add_argument('--height', type=float, help='figure height', default=5)
parser.add_argument('--width', type=float, help='figure width', default=8)
parser.add_argument('-s', '--save', type=str, help='save figure (using this file name)')
parser.add_argument('-t', '--title', type=str, help='figure title')
parser.add_argument('-v', '--verbose', action="store_true", help="print info to terminal")
parser.add_argument('--filter', type=float, help='width of Gaussian filter', default=None)
args = parser.parse_args()

if args.verbose:
    print('plot-if-spectrum.py')
    print(' - Files: ', args.file)

# Unpack keyword arguments
if args.verbose:
    print(' - Unpacking arguments')
if args.thot is not None:
    params['t_hot'] = args.thot
if args.tcold is not None:
    params['t_cold'] = args.tcold

# Plot data
if args.verbose:
    print(' - Generating figure')
fig, ax = plt.subplots(figsize=(args.width, args.height))

for filename in args.file:
    if args.verbose:
        print("   - Importing: " + filename)
    # Import and plot noise temperature
    data = if_response(filename, ifresp_delimiter=args.delimiter)
    freq, tn = data[0], data[1]
    # Filter
    if args.filter is not None:
        tn = gauss_conv(tn, args.filter)
    plt.plot(freq, tn, label=filename)

# Figure properties
if args.verbose:
    print(' - Customizing figure')
plt.xlabel(r'Frequency (GHz)')
plt.ylabel(r'Noise Temperature (K)')
if args.xmax is not None:
    plt.xlim([0, args.xmax])
if args.ymax is not None:
    plt.ylim([0, args.ymax])
else:
    plt.ylim([0, 500])
plt.legend(loc=0)
if args.title is not None:
    ax.set_title(args.title)

# Save or plot
if args.save is not None:
    if args.verbose:
        print(' - Saving figure')
    plt.savefig(args.save, dpi=500)
else:
    if args.verbose:
        print(' - Displaying figure')
    plt.show()
