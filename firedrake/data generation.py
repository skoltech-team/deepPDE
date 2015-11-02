import h5py
import numpy as np
import scipy as sp
from scipy import misc
from firedrake import *
from nifty import *

N_samples = 20000
N_mesh = 28

data = np.empty((N_samples, 1, N_mesh, N_mesh), dtype=np.float32)
label = np.empty((N_samples, 1), dtype=np.float32)

class RandomG(Expression):
    def __init__(self, num_k, n=N_mesh, mink=0.01, maxk=0.1):
		Expression.__init__(self)
		x_space = rg_space(N_mesh, naxes=2)
		k_space = x_space.get_codomain()
		a = field(x_space, target=k_space, random="syn", spec=lambda k: (1 / (k + 1)) ** 5)
		self._data = (maxk - mink) / (a.max() - a.min()) * a.val + (a.max() * mink - a.min() * maxk) / (a.max() - a.min())
		data[num_k, 0, :, :] = (self._data - mink) / (maxk - mink)
    def eval(self, value, X):
        value[:] = self._data[self._data.shape[0]*X[0] - 0.5, self._data.shape[1]*X[1] - 0.5]

mesh = UnitSquareMesh(N_mesh, N_mesh)

V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)

k = Function(V)
f = Function(V)
f.interpolate(Expression("1.0"))

s = Function(V)

bcs = [DirichletBC(V, 0, i+1) for i in range(4)]

L = f * v * dx

for j in range(N_samples):
	k.interpolate(RandomG(j))

	a = dot(grad(v), k*grad(u)) * dx

	solve(a==L, s, solver_parameters={'ksp_type': 'cg'}, bcs=bcs)
	label[j] = s.vector().array().max()

with h5py.File("train.h5", "w") as f:
	print data[:np.int32(N_samples*0.8), :, :, :].shape[0]
	f["data"] = data[:np.int32(N_samples*0.8), :, :, :]
	f["label"] = label[:np.int32(N_samples*0.8), :]

with h5py.File("test.h5", "w") as f:
	print data[np.int32(N_samples*0.8):, :, :, :].shape[0]
	f["data"] = data[np.int32(N_samples*0.8):, :, :, :]
	f["label"] = label[np.int32(N_samples*0.8):, :]