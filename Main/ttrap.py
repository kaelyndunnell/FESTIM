from fenics import *
from dolfin import *
import numpy as np
import csv
import sys
import os
import argparse


def save_as():
    valid = False
    while valid is False:
        print("Save as (.csv):")
        filedesorption = input()
        if filedesorption == '':
            filedesorption = "desorption.csv"
        if filedesorption.endswith('.csv'):
            valid = True
            try:
                with open(filedesorption, 'r') as f:
                    print('This file already exists.'
                          ' Do you want to replace it ? (y/n)')
                choice = input()
                if choice == "n" or choice == "N":
                    valid = False
                elif choice != "y" and choice != "Y":
                    valid = False
            except:
                valid = True
        else:
            print("Please enter a file ending with the extension .csv")
            valid = False
    return filedesorption


def export_TDS(filedesorption):
    busy = True
    while busy is True:
        try:
            with open(filedesorption, "w+") as output:
                busy = False
                writer = csv.writer(output, lineterminator='\n')
                writer.writerows(['dTt'])
                for val in desorption:
                    writer.writerows([val])
        except:
            print("The file " + filedesorption + " is currently busy."
                  "Please close the application then press any key")
            input()
    return

implantation_time = 400.0
resting_time = 50
ramp = 8
delta_TDS = 500
r = 0
flux = 2.5e19
density = 6.3e28
n_trap_1 = 1.3e-3  # trap 1 density
n_trap_2 = 4e-4  # trap 2 density
n_trap_3a_max = 1e-1
n_trap_3b_max = 1e-2
rate_3a = 6e-4
rate_3b = 2e-4
xp = 1e-6

E1 = 0.87  # in eV trap 1 activation energy
E2 = 1.0  # in eV activation energy
E3 = 1.5  # in eV activation energy
alpha = Constant(1.1e-10)  # lattice constant ()
beta = Constant(6)  # number of solute sites per atom (6 for W)
v_0 = 1e13  # frequency factor s-1
k_B = 8.6e-5

TDS_time = int(delta_TDS / ramp) + 1
# final time
Time = implantation_time+resting_time+TDS_time
# number of time steps
num_steps = 2*int(implantation_time+resting_time+TDS_time)
k = Time / num_steps  # time step size
dt = Constant(k)
t = 0  # Initialising time to 0s
size = 20e-6
nb_cells_in = 20
mesh = IntervalMesh(nb_cells_in, 0, size)
nb_cells_ref = 1000
refinement_point = 3e-6
print("Mesh size before local refinement is " + str(len(mesh.cells())))
while len(mesh.cells()) < nb_cells_in + nb_cells_ref:
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
    cell_markers.set_all(False)
    for cell in cells(mesh):
        if cell.midpoint().x() < refinement_point:
            cell_markers[cell] = True
    mesh = refine(mesh, cell_markers)
print("Mesh size after local refinement is " + str(len(mesh.cells())))
nb_cells_in = len(mesh.cells())
nb_cells_ref = 100
refinement_point = 10e-9
print("Mesh size before local refinement is " + str(len(mesh.cells())))
while len(mesh.cells()) < nb_cells_in + nb_cells_ref:
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
    cell_markers.set_all(False)
    for cell in cells(mesh):
        if cell.midpoint().x() < refinement_point:
            cell_markers[cell] = True
    mesh = refine(mesh, cell_markers)
print("Mesh size after local refinement is " + str(len(mesh.cells())))


# Define function space for system of concentrations
P1 = FiniteElement('P', interval, 1)
element = MixedElement([P1, P1, P1, P1])
V = FunctionSpace(mesh, element)
W = FunctionSpace(mesh, 'P', 1)
# BCs
print('Defining boundary conditions')


def inside(x, on_boundary):
    return on_boundary and (near(x[0], 0))


def outside(x, on_boundary):
    return on_boundary and (near(x[0], size))
# #Tritium concentration
inside_bc_c = Expression(('0', '0', '0', '0'), t=0, degree=1)
bci_c = DirichletBC(V, inside_bc_c, inside)
bco_c = DirichletBC(V, inside_bc_c, outside)
bcs = [bci_c, bco_c]


# Define test functions
v_1, v_2, v_3, v_4 = TestFunctions(V)
v_trap_3 = TestFunction(W)

u = Function(V)
u_n = Function(V)
n_trap_3 = TrialFunction(W)  # trap 3 density

# Split system functions to access components
u_1, u_2, u_3, u_4 = split(u)

print('Defining initial values')
ini_u = Expression(("x[0]<1e-6 ? 0 : 0", "0", "0", "0"), degree=1)
u_n = interpolate(ini_u, V)
u_n1, u_n2, u_n3, u_n4 = split(u_n)

ini_n_trap_3 = Expression("0", degree=1)
n_trap_3_n = interpolate(ini_n_trap_3, W)
n_trap_3_ = Function(W)
print('Defining source terms')
f = Expression('1/(2.5e-9*pow(2*3.14,0.5))*  \
               exp(-0.5*pow(((x[0]-4.5e-9)/2.5e-9), 2))',
               degree=2)  # This is the tritium volumetric source term
teta = Expression('x[0] < xp ? 1/xp : 0',
                  xp=xp, degree=1)
flux_ = Expression('t <= implantation_time ? flux/density : 0',
                   t=0, implantation_time=implantation_time,
                   flux=flux, density=density, degree=1)
# Define expressions used in variational forms
print('Defining variational problem')


temp = Expression('t <= (implantation_time+resting_time) ? \
                  300 : 300+ramp*(t-(implantation_time+resting_time))',
                  implantation_time=implantation_time,
                  resting_time=resting_time,
                  ramp=ramp,
                  t=0, degree=2)


def T_var(t):
    if t < implantation_time:
        return 300
    elif t < implantation_time+resting_time:
        return 300
    else:
        return 300+ramp*(t-(implantation_time+resting_time))
T = T_var(t)


def calculate_D(T, subdomain):
    return 4.1e-7*exp(-0.39/(k_B*T))
D = calculate_D(T_var(0), 0)

# Define variational problem
transient_trap1 = ((u_2 - u_n2) / dt)*v_2*dx
trapping_trap1 = - D/alpha/alpha/beta*u_1*(n_trap_1 - u_2)*v_2*dx
detrapping_trap1 = v_0*exp(-E1/k_B/temp)*u_2*v_2*dx
transient_trap2 = ((u_3 - u_n3) / dt)*v_3*dx
trapping_trap2 = - D/alpha/alpha/beta*u_1*(n_trap_2 - u_3)*v_3*dx
detrapping_trap2 = v_0*exp(-E2/k_B/temp)*u_3*v_3*dx
transient_trap3 = ((u_4 - u_n4) / dt)*v_4*dx
trapping_trap3 = - D/alpha/alpha/beta*u_1*(n_trap_3_ - u_4)*v_4*dx
detrapping_trap3 = v_0*exp(-E3/k_B/temp)*u_4*v_4*dx

transient_sol = ((u_1 - u_n1) / dt)*v_1*dx
diff_sol = D*dot(grad(u_1), grad(v_1))*dx
source_sol = - (1-r)*flux_*f*v_1*dx
trapping1_sol = ((u_2 - u_n2) / dt)*v_1*dx
trapping2_sol = ((u_3 - u_n3) / dt)*v_1*dx
trapping3_sol = ((u_4 - u_n4) / dt)*v_1*dx
F = transient_sol + source_sol + diff_sol
F += trapping1_sol + trapping2_sol + trapping3_sol
F += transient_trap1 + trapping_trap1 + detrapping_trap1
F += transient_trap2 + trapping_trap2 + detrapping_trap2
F += transient_trap3 + trapping_trap3 + detrapping_trap3


F_n3 = ((n_trap_3 - n_trap_3_n)/dt)*v_trap_3*dx
F_n3 += -(1-r)*flux_*((1 - n_trap_3_n/n_trap_3a_max)*rate_3a*f + (1 - n_trap_3_n/n_trap_3b_max)*rate_3b*teta)*v_trap_3 * dx


vtkfile_u_1 = File('Solution/c_sol.pvd')
vtkfile_u_2 = File('Solution/c_trap1.pvd')
vtkfile_u_3 = File('Solution/c_trap2.pvd')
vtkfile_u_4 = File('Solution/c_trap3.pvd')
filedesorption = save_as()

#  Time-stepping
print('Time stepping...')

desorption = list()
total_n = 0

set_log_level(30)  # Set the log level to WARNING
# set_log_level(20) # Set the log level to INFO

for n in range(num_steps):
    # Update current time
    t += k
    T = T_var(t)
    flux_.t += k
    D = calculate_D(T_var(t), 0)
    print(str(round(t/Time*100, 2)) + ' %        ' + str(round(t, 1)) + ' s',
          end="\r")
    # Solve variational problem for time step
    solve(F == 0, u, bcs,
          solver_parameters={"newton_solver": {"absolute_tolerance": 1e-19}})

    solve(lhs(F_n3) == rhs(F_n3), n_trap_3_, [])
    _u_1, _u_2, _u_3, _u_4 = u.split()

    # Save solution to file (.vtu)
    vtkfile_u_1 << (_u_1, t)
    vtkfile_u_2 << (_u_2, t)
    vtkfile_u_3 << (_u_3, t)
    vtkfile_u_4 << (_u_4, t)

    total_trap1 = assemble(_u_2*density*dx)
    total_trap2 = assemble(_u_3*density*dx)
    total_trap3 = assemble(_u_4*density*dx)
    total_trap = total_trap1 + total_trap2 + total_trap3
    total_sol = assemble(_u_1*density*dx)
    total = total_trap + total_sol
    desorption_rate = [-(total-total_n)/k, T_var(t-k), t]
    total_n = total

    if t > implantation_time+resting_time:
        desorption.append(desorption_rate)
        print("Total of D = "+str(total))
        print("Desorption rate = " + str(desorption_rate))

    # Update previous solutions
    u_n.assign(u)
    n_trap_3_n.assign(n_trap_3_)

export_TDS(filedesorption)
