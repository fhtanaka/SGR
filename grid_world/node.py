from sgr.sgr import SGR
from typing import Dict
import itertools
from neat.six_util import iteritems



class Node:

    idCounter = itertools.count().__next__
    def __init__(
        self, 
        population: SGR,
        task: str,
        n_steps: int,
        connections: Dict[str, int] = None
        ) -> None:

        self.id = self.idCounter()
        self.sgr_pop = population
        self.connections: Dict[str, int] = {} if connections == None else connections
        self.task = task
        self.n_steps = n_steps

    def add_conn(self, node_id, weight=1):
        self.connections[node_id] = weight
    
    def remove_conn(self, node_id):
        if node_id in self.connections:
            del self.connections[node_id]

    def eval_population_parallel(
        self,
        n_cpus,
        env_name,
        ):

        genomes = list(iteritems(self.sgr_pop.pop.population))
        self.sgr_pop.fit_func(
            genomes,
            self.sgr_pop.neat_config,
            self.task,
            self.n_steps,
            n_cpus,
        ) 

    def eval_population_linear(
        self,
        ):
        genomes = list(iteritems(self.sgr_pop.pop.population))

        for g_id, g in genomes:
            fit, _ = self.sgr_pop.single_genome_fit(
                g,
                self.n_steps,
                self.task,
            )
            g.fitness = fit
            if fit > self.sgr_pop.best_fit:
                self.sgr_pop.best_fit = fit
                self.sgr_pop.stagnation = 0
                self.sgr_pop.best_genome = g

        surviving_genomes = {g_id: g for g_id, g in genomes if g.fitness is not None and g.fitness > -1000}
        self.sgr_pop.population = surviving_genomes
        self.sgr_pop.check_stagnation_and_save_interval()
        

        