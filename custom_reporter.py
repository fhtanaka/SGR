from __future__ import division, print_function

import time

from neat.math_util import mean, stdev
from neat.six_util import itervalues, iterkeys
from neat.reporting import ReporterSet
import neat

class CustomReporter():
    """Uses `print` to output information about the run; an example reporter class."""
    def __init__(self, show_species_detail, txt_file, csv_file):
        self.show_species_detail = show_species_detail
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0
        self.txt = open(txt_file, "w+")
        self.csv = open(csv_file, "w+")
        self.csv.write("pop_size;best_fit;num_species\n")

    def start_generation(self, generation):
        self.generation = generation
        text = ""
        text += '\n ****** Running generation {0} ****** \n'.format(generation) + "\n"
        self.txt.write(text)
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        ng = len(population)
        ns = len(species_set.species)
        text = ""
        if self.show_species_detail:
            text += 'Population of {0:d} members in {1:d} species:'.format(ng, ns) + "\n"
            sids = list(iterkeys(species_set.species))
            sids.sort()
            text += "   ID   age  size  fitness  adj fit  stag" + "\n"
            text += "  ====  ===  ====  =======  =======  ====" + "\n"
            for sid in sids:
                s = species_set.species[sid]
                a = self.generation - s.created
                n = len(s.members)
                f = "--" if s.fitness is None else "{:.1f}".format(s.fitness)
                af = "--" if s.adjusted_fitness is None else "{:.3f}".format(s.adjusted_fitness)
                st = self.generation - s.last_improved
                text += "  {: >4}  {: >3}  {: >4}  {: >7}  {: >7}  {: >4}".format(sid, a, n, f, af, st) + "\n"
        else:
            text += 'Population of {0:d} members in {1:d} species'.format(ng, ns) + "\n"

        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)
        text += 'Total extinctions: {0:d}'.format(self.num_extinctions) + "\n"
        if len(self.generation_times) > 1:
            text += "Generation time: {0:.3f} sec ({1:.3f} average)".format(elapsed, average) + "\n"
        else:
            text += "Generation time: {0:.3f} sec".format(elapsed) + "\n"

        self.txt.write(text)


    def post_evaluate(self, config, population, species, best_genome):
        # pylint: disable=no-self-use
        fitnesses = [c.fitness for c in itervalues(population)]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        best_species_id = species.get_species_id(best_genome.key)
        text = ""
        text += 'Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}'.format(fit_mean, fit_std) + "\n"
        text += 'Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}'.format(best_genome.fitness, best_genome.size(), best_species_id, best_genome.key) + "\n"
        self.txt.write(text)
        self.csv.write(f"{len(population)};{best_genome.fitness};{len(species.species)}\n")

    def complete_extinction(self):
        self.num_extinctions += 1
        text = 'All species extinct.' + "\n"
        self.txt.write(text)

    def found_solution(self, config, generation, best):
        text = '\nBest individual in generation {0} meets fitness threshold - complexity: {1!r}'.format(self.generation, best.size()) + "\n"
        self.txt.write(text)

    def species_stagnant(self, sid, species):
        text = ""
        if self.show_species_detail:
            text += "\nSpecies {0} with {1} members is stagnated: removing it".format(sid, len(species.members)) + "\n"
            self.txt.write(text)

    def info(self, msg):
        self.txt.write(msg+"\n")

def remove_reporters(pop: neat.Population):
    pop.reporters = ReporterSet()
