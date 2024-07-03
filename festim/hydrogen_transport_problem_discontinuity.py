import dolfinx
from dolfinx import fem
from dolfinx.nls.petsc import NewtonSolver
import basix
import basix.ufl
import ufl
from mpi4py import MPI
import numpy as np
import tqdm.auto as tqdm
import festim as F


class HydrogenTransportProblemDiscontinuity:
    """
    Hydrogen Transport Problem.

    Args:
        mesh (festim.Mesh): the mesh of the model
        subdomains (list of festim.Subdomain): the subdomains of the model
        species (list of festim.Species): the species of the model
        reactions (list of festim.Reaction): the reactions of the model
        temperature (float, int, fem.Constant, fem.Function or callable): the
            temperature of the model (K)
        sources (list of festim.Source): the hydrogen sources of the model
        initial_conditions (list of festim.InitialCondition): the initial conditions
            of the model
        boundary_conditions (list of festim.BoundaryCondition): the boundary
            conditions of the model
        solver_parameters (dict): the solver parameters of the model
        exports (list of festim.Export): the exports of the model
        traps (list of F.Trap): the traps of the model

    Attributes:
        mesh (festim.Mesh): the mesh of the model
        subdomains (list of festim.Subdomain): the subdomains of the model
        species (list of festim.Species): the species of the model
        reactions (list of festim.Reaction): the reactions of the model
        temperature (float, int, fem.Constant, fem.Function or callable): the
            temperature of the model (K)
        sources (list of festim.Source): the hydrogen sources of the model
        initial_conditions (list of festim.InitialCondition): the initial conditions
            of the model
        boundary_conditions (list of festim.BoundaryCondition): the boundary
            conditions of the model
        solver_parameters (dict): the solver parameters of the model
        exports (list of festim.Export): the exports of the model
        traps (list of F.Trap): the traps of the model
        dx (dolfinx.fem.dx): the volume measure of the model
        ds (dolfinx.fem.ds): the surface measure of the model
        function_space (dolfinx.fem.FunctionSpaceBase): the function space of the
            model
        facet_meshtags (dolfinx.mesh.MeshTags): the facet meshtags of the model
        volume_meshtags (dolfinx.mesh.MeshTags): the volume meshtags of the
            model
        formulation (ufl.form.Form): the formulation of the model
        solver (dolfinx.nls.newton.NewtonSolver): the solver of the model
        multispecies (bool): True if the model has more than one species.
        temperature_fenics (fem.Constant or fem.Function): the
            temperature of the model as a fenics object (fem.Constant or
            fem.Function).
        temperature_expr (fem.Expression): the expression of the temperature
            that is used to update the temperature_fenics
        temperature_time_dependent (bool): True if the temperature is time
            dependent
        V_DG_0 (dolfinx.fem.FunctionSpaceBase): A DG function space of degree 0
            over domain
        V_DG_1 (dolfinx.fem.FunctionSpaceBase): A DG function space of degree 1
            over domain
        volume_subdomains (list of festim.VolumeSubdomain): the volume subdomains
            of the model
        surface_subdomains (list of festim.SurfaceSubdomain): the surface subdomains
            of the model


    Usage:
        >>> import festim as F
        >>> my_model = F.HydrogenTransportProblem()
        >>> my_model.mesh = F.Mesh(...)
        >>> my_model.subdomains = [F.Subdomain(...)]
        >>> my_model.species = [F.Species(name="H"), F.Species(name="Trap")]
        >>> my_model.temperature = 500
        >>> my_model.sources = [F.ParticleSource(...)]
        >>> my_model.boundary_conditions = [F.BoundaryCondition(...)]
        >>> my_model.initialise()

        or

        >>> my_model = F.HydrogenTransportProblem(
        ...     mesh=F.Mesh(...),
        ...     subdomains=[F.Subdomain(...)],
        ...     species=[F.Species(name="H"), F.Species(name="Trap")],
        ... )
        >>> my_model.initialise()

    """

    def __init__(
        self,
        mesh=None,
        subdomains=None,
        species=None,
        reactions=None,
        temperature=None,
        sources=None,
        initial_conditions=None,
        boundary_conditions=None,
        settings=None,
        exports=None,
        traps=None,
    ):
        self.mesh = mesh
        self.temperature = temperature
        self.settings = settings

        # for arguments to initliase as empty list
        # if arg not None, assign arg, else assign empty list
        self.subdomains = subdomains or []
        self.species = species or []
        self.reactions = reactions or []
        self.sources = sources or []
        self.initial_conditions = initial_conditions or []
        self.boundary_conditions = boundary_conditions or []
        self.exports = exports or []
        self.traps = traps or []

        self.dx = None
        self.ds = None
        self.function_space = None
        self.facet_meshtags = None
        self.volume_meshtags = None
        self.formulation = None
        self.bc_forms = []
        self.temperature_fenics = None

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        if value is None:
            self._temperature = value
        elif isinstance(value, (float, int, fem.Constant, fem.Function)):
            self._temperature = value
        elif callable(value):
            self._temperature = value
        else:
            raise TypeError(
                f"Value must be a float, int, fem.Constant, fem.Function, or callable"
            )

    @property
    def temperature_fenics(self):
        return self._temperature_fenics

    @temperature_fenics.setter
    def temperature_fenics(self, value):
        if value is None:
            self._temperature_fenics = value
            return
        elif not isinstance(
            value,
            (fem.Constant, fem.Function),
        ):
            raise TypeError(f"Value must be a fem.Constant or fem.Function")
        self._temperature_fenics = value

    @property
    def temperature_time_dependent(self):
        if self.temperature is None:
            return False
        if isinstance(self.temperature, fem.Constant):
            return False
        if callable(self.temperature):
            arguments = self.temperature.__code__.co_varnames
            return "t" in arguments
        else:
            return False

    @property
    def multispecies(self):
        return len(self.species) > 1

    @property
    def volume_subdomains(self):
        return [s for s in self.subdomains if isinstance(s, F.VolumeSubdomain)]

    @property
    def surface_subdomains(self):
        return [s for s in self.subdomains if isinstance(s, F.SurfaceSubdomain)]

    @property
    def species(self):
        return self._species

    @species.setter
    def species(self, value):
        # check that all species are of type festim.Species
        for spe in value:
            if not isinstance(spe, F.Species):
                raise TypeError(
                    f"elements of species must be of type festim.Species not {type(spe)}"
                )
        self._species = value

    @property
    def facet_meshtags(self):
        return self._facet_meshtags

    @facet_meshtags.setter
    def facet_meshtags(self, value):
        if value is None:
            self._facet_meshtags = value
        elif isinstance(value, dolfinx.mesh.MeshTags):
            self._facet_meshtags = value
        else:
            raise TypeError(f"value must be of type dolfinx.mesh.MeshTags")

    @property
    def volume_meshtags(self):
        return self._volume_meshtags

    @volume_meshtags.setter
    def volume_meshtags(self, value):
        if value is None:
            self._volume_meshtags = value
        elif isinstance(value, dolfinx.mesh.MeshTags):
            self._volume_meshtags = value
        else:
            raise TypeError(f"value must be of type dolfinx.mesh.MeshTags")

    def initialise(self):
        self.define_meshtags_and_measures()
        self.create_submeshes()
        self.create_entity_maps()
        self.define_function_spaces()
        self.assign_functions_to_species()

        self.t = fem.Constant(self.mesh.mesh, 0.0)
        if self.settings.transient:
            # TODO should raise error if no stepsize is provided
            # TODO Should this be an attribute of festim.Stepsize?
            self.dt = F.as_fenics_constant(
                self.settings.stepsize.initial_value, self.mesh.mesh
            )

        self.define_temperature()
        self.define_boundary_conditions()
        self.create_initial_conditions()
        self.create_formulation()
        self.create_solver()

    def define_temperature(self):
        self.temperature = 500

    def create_entity_maps(self):
        # We need to modify the cell maps, as for `dS` integrals of interfaces between submeshes, there is no entity to map to.
        # We use the entity on the same side to fix this (as all restrictions are one-sided)

        self.entity_maps = {}

        num_facets_local = (
            self.mesh.mesh.topology.index_map(self.mesh.fdim).size_local + self.mesh.mesh.topology.index_map(self.mesh.fdim).num_ghosts
        )

        for subdomain in self.volume_subdomains:
            parent_to_sub = np.full(num_facets_local, -1, dtype=np.int32)
            parent_to_sub[subdomain.submesh_to_mesh] = np.arange(len(subdomain.submesh_to_mesh), dtype=np.int32)

            # Transfer meshtags to submesh
            ft, facet_to_parent = F.helpers.transfer_meshtags_to_submesh(
                self.mesh.mesh, self.facet_meshtags, subdomain.submesh, subdomain.v_map, subdomain.submesh_to_mesh
            )


            parent_to_facet = np.full(num_facets_local, -1)
            parent_to_facet[facet_to_parent] = np.arange(len(facet_to_parent), dtype=np.int32)


            # Hack, as we use one-sided restrictions, pad dS integral with the same entity from the same cell on both sides
            # TODO do it for each interface here we assume the interface is tagged with 5
            f_to_c = self.mesh.mesh.topology.connectivity(self.mesh.fdim, self.mesh.vdim)
            for facet in self.facet_meshtags.find(5):
                cells = f_to_c.links(facet)
                assert len(cells) == 2
                map = parent_to_sub[cells]
                parent_to_sub[cells] = max(map)

            self.entity_maps[subdomain.submesh] = parent_to_sub

        

    def define_function_spaces(self):
        """Creates the function space of the model, creates a mixed element if
        model is multispecies. Creates the main solution and previous solution
        function u and u_n. Create global DG function spaces of degree 0 and 1
        for the global diffusion coefficient"""
        self.function_spaces = {}
        self.u = {}
        self.u_n = {}

        for subdomain in self.volume_subdomains:
            element_CG = basix.ufl.element(
                basix.ElementFamily.P,
                subdomain.submesh.basix_cell(),
                1,
                basix.LagrangeVariant.equispaced,
            )
            elements = []
            for spe in self.species:
                if isinstance(spe, F.Species):
                    elements.append(element_CG)
            mixed_element = basix.ufl.mixed_element(elements)
            V = fem.functionspace(subdomain.submesh, mixed_element)
            self.function_spaces[subdomain] = V
            self.u[subdomain] = fem.Function(V)
            self.u_n[subdomain] = fem.Function(V)


    def assign_functions_to_species(self):
        """Creates the solution, prev solution, test function and
        post-processing solution for each species, if model is multispecies,
        created a collapsed function space for each species"""

        for subdomain in self.volume_subdomains:
            sub_solutions = list(ufl.split(self.u[subdomain]))
            sub_prev_solution = list(ufl.split(self.u_n[subdomain]))
            sub_test_functions = list(ufl.TestFunctions(self.function_spaces[subdomain]))
            for idx, spe in enumerate(self.species):
                spe.subdomain_to_solution[subdomain] = sub_solutions[idx]
                spe.subdomain_to_prev_solution[subdomain] = sub_prev_solution[idx]
                spe.subdomain_to_test_function[subdomain] = sub_test_functions[idx]
                spe.subdomain_to_post_processing_solution[subdomain] = self.u[subdomain].sub(idx)

    def create_submeshes(self):
        for subdomain in self.volume_subdomains:
            # TODO document these new attributes
            subdomain.submesh, subdomain.submesh_to_mesh, subdomain.v_map = dolfinx.mesh.create_submesh(
                self.mesh.mesh, self.mesh.vdim, self.volume_meshtags.find(subdomain.id)
            )[0:3]
            ct_r = dolfinx.mesh.meshtags(
                    subdomain.submesh,
                    subdomain.submesh.topology.dim,
                    subdomain.submesh_to_mesh,
                    np.full_like(subdomain.submesh_to_mesh, 1, dtype=np.int32),
            )
            subdomain.dx = ufl.Measure("dx", domain=subdomain.submesh, subdomain_data=ct_r, subdomain_id=1)

    def define_meshtags_and_measures(self):
        """Defines the facet and volume meshtags of the model which are used
        to define the measures fo the model, dx and ds"""
        if isinstance(self.mesh, F.MeshFromXDMF):
            self.facet_meshtags = self.mesh.define_surface_meshtags()
            self.volume_meshtags = self.mesh.define_volume_meshtags()

        elif (
            isinstance(self.mesh, F.Mesh)
            and self.facet_meshtags is None
            and self.volume_meshtags is None
        ):
            self.facet_meshtags, self.volume_meshtags = self.mesh.define_meshtags(
                surface_subdomains=self.surface_subdomains,
                volume_subdomains=self.volume_subdomains,
            )

        # check volume ids are unique
        vol_ids = [vol.id for vol in self.volume_subdomains]
        if len(vol_ids) != len(np.unique(vol_ids)):
            raise ValueError("Volume ids are not unique")

        # define measures
        self.ds = ufl.Measure(
            "ds", domain=self.mesh.mesh, subdomain_data=self.facet_meshtags
        )
        self.dx = ufl.Measure(
            "dx", domain=self.mesh.mesh, subdomain_data=self.volume_meshtags
        )


    def define_boundary_conditions(self):
        pass


    def create_source_values_fenics(self):
        """For each source create the value_fenics"""
        for source in self.sources:
            # create value_fenics for all F.ParticleSource objects
            if isinstance(source, F.ParticleSource):

                source.create_value_fenics(
                    mesh=self.mesh.mesh,
                    temperature=self.temperature_fenics,
                    t=self.t,
                )

    def create_flux_values_fenics(self):
        """For each particle flux create the value_fenics"""
        for bc in self.boundary_conditions:
            # create value_fenics for all F.ParticleFluxBC objects
            if isinstance(bc, F.ParticleFluxBC):

                bc.create_value_fenics(
                    mesh=self.mesh.mesh,
                    temperature=self.temperature_fenics,
                    t=self.t,
                )

    def create_initial_conditions(self):
        """For each initial condition, create the value_fenics and assign it to
        the previous solution of the condition's species"""

        if len(self.initial_conditions) > 0 and not self.settings.transient:
            raise ValueError(
                "Initial conditions can only be defined for transient simulations"
            )

        for condition in self.initial_conditions:
            raise NotImplementedError()


    def create_bulk_formulation(self, subdomain: F.VolumeSubdomain):
        # add diffusion and time derivative for each species
        form = 0
        dx = subdomain.dx
        for spe in self.species:

            u = spe.subdomain_to_solution[subdomain]
            u_n = spe.subdomain_to_prev_solution[subdomain]
            v = spe.subdomain_to_test_function[subdomain]
            D = 1  # FIXME
            # D = vol.material.get_diffusion_coefficient(
            #     self.mesh.mesh, self.temperature_fenics, spe
            # )
            if spe.mobile:
                form += ufl.dot(D * ufl.grad(u), ufl.grad(v)) * dx

            if self.settings.transient:
                form += ((u - u_n) / self.dt) * v * dx

        for reaction in self.reactions:
            if reaction.volume != subdomain:
                continue
            for reactant in reaction.reactant:
                if isinstance(reactant, F.Species):
                    form += (
                        reaction.reaction_term(self.temperature_fenics)
                        * reactant.subdomain_to_test_function[subdomain]
                        * dx
                    )

            # product
            if isinstance(reaction.product, list):
                products = reaction.product
            else:
                products = [reaction.product]
            for product in products:
                form += (
                    -reaction.reaction_term(self.temperature_fenics)
                    * product.test_function
                    * dx
                )

        return form

    def create_formulation(self):
        subdomain_to_formulation = {}
        for subdomain in self.volume_subdomains:
            subdomain_to_formulation[subdomain] = self.create_bulk_formulation(subdomain)

        
        # for interface in self.interfaces:
        #     subdomain_to_formulation[interface.subdomain1] += "something"
        #     subdomain_to_formulation[interface.subdomain2] += "something"

        formulations = [subdomain_to_formulation[subdomain] for subdomain in self.volume_subdomains]

        # jacobian
        self.jacobian = []
        for sub_F in formulations:
            self.jacobian.append([])
            for subdomain in self.volume_subdomains:
                jac = ufl.derivative(sub_F, self.u[subdomain])
                Jac = dolfinx.fem.form(jac, entity_maps=self.entity_maps)
                self.jacobian[-1].append(Jac)

        self.formulation = [dolfinx.fem.form(form, entity_maps=self.entity_maps) for form in formulations]

    def create_solver(self):
        """Creates the solver of the model"""
        raise NotImplementedError()

    def run(self):
        """Runs the model"""

        if self.settings.transient:
            # Solve transient
            self.progress = tqdm.tqdm(
                desc="Solving H transport problem",
                total=self.settings.final_time,
                unit_scale=True,
            )
            while self.t.value < self.settings.final_time:
                self.iterate()
            self.progress.refresh()  # refresh progress bar to show 100%
        else:
            # Solve steady-state
            self.solver.solve(self.u)
            self.post_processing()

    def iterate(self):
        """Iterates the model for a given time step"""
        self.progress.update(
            min(self.dt.value, abs(self.settings.final_time - self.t.value))
        )
        self.t.value += self.dt.value

        self.update_time_dependent_values()

        # solve main problem
        nb_its, converged = self.solver.solve(self.u)

        # post processing
        self.post_processing()

        # update previous solution
        self.u_n.x.array[:] = self.u.x.array[:]

        # adapt stepsize
        if self.settings.stepsize.adaptive:
            new_stepsize = self.settings.stepsize.modify_value(
                value=self.dt.value, nb_iterations=nb_its, t=self.t.value
            )
            self.dt.value = new_stepsize

    def update_time_dependent_values(self):
        t = float(self.t)
        if self.temperature_time_dependent:
            if isinstance(self.temperature_fenics, fem.Constant):
                self.temperature_fenics.value = self.temperature(t=t)
            elif isinstance(self.temperature_fenics, fem.Function):
                self.temperature_fenics.interpolate(self.temperature_expr)
        for bc in self.boundary_conditions:
            if bc.time_dependent:
                bc.update(t=t)
            elif self.temperature_time_dependent and bc.temperature_dependent:
                bc.update(t=t)

        for source in self.sources:
            if source.time_dependent:
                source.update(t=t)
            elif self.temperature_time_dependent and source.temperature_dependent:
                source.update(t=t)

    def post_processing(self):
        """Post processes the model"""

        pass
