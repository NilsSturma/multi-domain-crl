import itertools as itr

# === IMPORTS: THIRD-PARTY ===
import numpy as np
import networkx as nx

# === IMPORTS: LOCAL ===
from pyscipopt import Model, quicksum, multidict
import gurobipy as gp


class IntegerProgram:
    def __init__(
        self,
        env2dim: dict,
        weights: dict,
        linear_constraint = False,
        symmetry_breaking = True,
        solver = "scip"
    ) -> None:
        self.env2dim = env2dim
        self.weights = weights
        self.linear_constraint = linear_constraint
        self.symmetry_breaking = symmetry_breaking
        self.solver = solver

    def create_model_scip(
        self,
        linear_constraint=False,
        symmetry_breaking=True
    ):
        model = Model("minimum")
        p = min(self.env2dim.values())

        # === CREATE THE DECISION VARIABLES
        indicators = dict()
        for k in range(p):
            for e, dim in self.env2dim.items():
                for j_e in range(dim):
                    indicators[(k, e, j_e)] = model.addVar(vtype="B", name=f"A_{e}{j_e}^{k}")
        
        # === CREATE THE CONSTRAINTS
        # each node belongs to at most one cluster
        for e, dim in self.env2dim.items():
            for j_e in range(dim):
                inds = [indicators[(k, e, j_e)] for k in range(p)]
                model.addCons(quicksum(inds) <= 1, f"Node_{e}{j_e}")

        # each cluster has one node from each environment
        for k in range(p):
            for e, dim in self.env2dim.items():
                inds = [indicators[(k, e, j_e)] for j_e in range(dim)]
                model.addCons(quicksum(inds) == 1, f"Cluster_{e}{k}")

        # === CREATE THE OBJECTIVE
        weight_terms = []
        weight_total = 0
        for k in range(p):
            for e, f in itr.combinations(self.env2dim, 2):
                for j_e in range(self.env2dim[e]):
                    for j_f in range(self.env2dim[f]):
                        ind_e = indicators[(k, e, j_e)]
                        ind_f = indicators[(k, f, j_f)]
                        joint_ind = model.addVar(vtype="B", name=f"A_{e}{j_e},{f}{j_f}^{k}")
                        if linear_constraint:
                            model.addCons(ind_e + ind_f - joint_ind <= 1)
                            model.addCons(ind_e - joint_ind >= 0)
                            model.addCons(ind_f - joint_ind >= 0)
                        else:
                            model.addCons(ind_e * ind_f == joint_ind)
                        weight = self.weights[(e, j_e), (f, j_f)]
                        weight_total += weight
                        weight_terms.append(joint_ind * weight)

        if symmetry_breaking:
            env_fixed = next(e for e, dim in self.env2dim.items() if dim == p)
            for k in range(p):
                ind = indicators[(k, env_fixed, k)]
                model.addCons(ind == 1)

        model.setObjective(quicksum(weight_terms), "minimize")

        return model, indicators

    def create_model_gurobi(
        self,
        linear_constraint=False,
        symmetry_breaking=True
    ):
        model = gp.Model("minimum")
        p = min(self.env2dim.values())

        # === CREATE THE DECISION VARIABLES
        indicators = dict()
        for k in range(p):
            for e, dim in self.env2dim.items():
                for j_e in range(dim):
                    indicators[(k, e, j_e)] = model.addVar(vtype="B", name=f"A_{e}{j_e}^{k}")
        
        # === CREATE THE CONSTRAINTS
        # each node belongs to at most one cluster
        for e, dim in self.env2dim.items():
            for j_e in range(dim):
                inds = [indicators[(k, e, j_e)] for k in range(p)]
                model.addConstr(sum(inds) <= 1, f"Node_{e}{j_e}")

        # each cluster has one node from each environment
        for k in range(p):
            for e, dim in self.env2dim.items():
                inds = [indicators[(k, e, j_e)] for j_e in range(dim)]
                model.addConstr(sum(inds) == 1, f"Cluster_{e}{k}")

        # === CREATE THE OBJECTIVE
        weight_terms = []
        weight_total = 0
        for k in range(p):
            for e, f in itr.combinations(self.env2dim, 2):
                for j_e in range(self.env2dim[e]):
                    for j_f in range(self.env2dim[f]):
                        ind_e = indicators[(k, e, j_e)]
                        ind_f = indicators[(k, f, j_f)]
                        joint_ind = model.addVar(vtype="B", name=f"A_{e}{j_e},{f}{j_f}^{k}")
                        if linear_constraint:
                            model.addConstr(ind_e + ind_f - joint_ind <= 1)
                            model.addConstr(ind_e - joint_ind >= 0)
                            model.addConstr(ind_f - joint_ind >= 0)
                        else:
                            model.addConstr(ind_e * ind_f == joint_ind)
                        weight = self.weights[(e, j_e), (f, j_f)]
                        weight_total += weight
                        weight_terms.append(joint_ind * weight)

        if symmetry_breaking:
            env_fixed = next(e for e, dim in self.env2dim.items() if dim == p)
            for k in range(p):
                ind = indicators[(k, env_fixed, k)]
                model.addConstr(ind == 1)

        model.setObjective(sum(weight_terms), gp.GRB.MINIMIZE)

        return model, indicators

    def scip_solution2clusters(self, sol, indicators):
        p = min(self.env2dim.values())
        g = nx.Graph()

        for k in range(p):
            for e, dim in self.env2dim.items():
                for j_e in range(dim):
                    indicator = indicators[(k, e, j_e)]
                    if np.isclose(sol[indicator], 1):
                        g.add_edge((e, j_e), k)

        estimated_clusters = list(nx.connected_components(g))
        
        return [{elm for elm in c if isinstance(elm, tuple)} for c in estimated_clusters]

    def gurobi_solution2clusters(self, indicators):
        p = min(self.env2dim.values())
        g = nx.Graph()

        for k in range(p):
            for e, dim in self.env2dim.items():
                for j_e in range(dim):
                    indicator = indicators[(k, e, j_e)]
                    if np.isclose(indicator.X, 1):
                        g.add_edge((e, j_e), k)

        estimated_clusters = list(nx.connected_components(g))
        
        return [{elm for elm in c if isinstance(elm, tuple)} for c in estimated_clusters]

    def solve_scip(self):
        model, indicators = self.create_model_scip()
        model.optimize()
        solution = model.getBestSol()
        estimated_clusters = self.scip_solution2clusters(solution, indicators)
        return estimated_clusters

    def solve_gurobi(self):
        model, indicators = self.create_model_gurobi()
        model.optimize()
        estimated_clusters = self.gurobi_solution2clusters(indicators)
        return estimated_clusters

    def solve(self):
        if self.solver == "scip":
            return self.solve_scip()
        elif self.solver == "gurobi":
            return self.solve_gurobi()

