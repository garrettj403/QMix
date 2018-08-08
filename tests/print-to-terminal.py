from qmix.misc.terminal import *

print_intro()
title('Title')
title('Very Very Long Title')
header('Header')
print("")
header('Colorful Header', color='CYAN')
pvale('c', 3e8, 'm/s', 'speed of light')
pvalf('pi', 3.14159, 'rad', color='RED')
pvalf('Zin', 300 + 1j*5, 'ohms')
pvalf('Zin', 342.15673 - 1j*543.123124, 'ohms')
print("")
