from firedrake import *
import scipy as sp
from scipy import misc
from nifty import *

N_pics = 6
N_mesh = 512

class RandomG(Expression):
    def __init__(self, i_pic, mode, n=N_mesh, mink=0.01, maxk=0.1):
		Expression.__init__(self)
		x_space = rg_space(N_mesh, naxes=2)
		k_space = x_space.get_codomain()
		a = field(x_space, target=k_space, random="syn", spec=lambda k: (1 / (k + 1)) ** 5)
		self._data = (maxk - mink) / (a.max() - a.min()) * a.val + (a.max() * mink - a.min() * maxk) / (a.max() - a.min())
		sp.misc.imsave("DEEP/{}/{}.jpg".format(mode, i_pic), 255*(self._data - mink)/(maxk - mink))
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

for j in range(N_pics):
	a = dot(grad(v), k*grad(u)) * dx
	k.interpolate(RandomG(j+1, "pics/{}".format(j+1)))
	solve(a==L, s, solver_parameters={'ksp_type': 'cg'}, bcs=bcs)

	File("DEEP/pics/{}/ufield.pvd".format(j+1)) << s
	File("DEEP/pics/{}/kfield.pvd".format(j+1)) << k