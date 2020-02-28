import qmix

from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

# Setup ----------------------------------------------------------------------

resp = qmix.respfn.RespFnPolynomial(50, verbose=False)

num_b = (10, 5, 5, 5)

# 1 tone ---------------------------------------------------------------------

cct = qmix.circuit.EmbeddingCircuit(1, 1)
cct.vph[1] = 0.3

vj = cct.initialize_vj()
vj[1,1,:] = 0.3

with PyCallGraph(output=GraphvizOutput(output_file='results/pycall-qtcurrent-1tone.png')):
    iac = qmix.qtcurrent.qtcurrent(vj, cct, resp, 0.3, num_b=num_b, verbose=False)

# 2 tone ---------------------------------------------------------------------

cct = qmix.circuit.EmbeddingCircuit(2, 1)
cct.vph[1] = 0.30
cct.vph[2] = 0.33

vj = cct.initialize_vj()
vj[1,1,:] = 0.3
vj[2,1,:] = 0.1

with PyCallGraph(output=GraphvizOutput(output_file='results/pycall-qtcurrent-2tone.png')):
    iac = qmix.qtcurrent.qtcurrent(vj, cct, resp, 0.3, num_b=num_b, verbose=False)

# 3 tone ---------------------------------------------------------------------

cct = qmix.circuit.EmbeddingCircuit(3, 1)
cct.vph[1] = 0.30
cct.vph[2] = 0.33
cct.vph[3] = 0.27

vj = cct.initialize_vj()
vj[1,1,:] = 0.3
vj[2,1,:] = 0.1
vj[3,1,:] = 0.1

with PyCallGraph(output=GraphvizOutput(output_file='results/pycall-qtcurrent-3tone.png')):
    iac = qmix.qtcurrent.qtcurrent(vj, cct, resp, 0.3, num_b=num_b, verbose=False)

# 4 tone ---------------------------------------------------------------------

cct = qmix.circuit.EmbeddingCircuit(4, 1)
cct.vph[1] = 0.30
cct.vph[2] = 0.33
cct.vph[3] = 0.27
cct.vph[4] = 0.03

vj = cct.initialize_vj()
vj[1,1,:] = 0.3
vj[2,1,:] = 0.1
vj[3,1,:] = 0.1
vj[4,1,:] = 0.0

with PyCallGraph(output=GraphvizOutput(output_file='results/pycall-qtcurrent-4tone.png')):
    iac = qmix.qtcurrent.qtcurrent(vj, cct, resp, 0.3, num_b=num_b, verbose=False)
