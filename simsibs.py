import random
from math import log
from multiprocessing import Manager, Process

random.seed("Alice and Bob")

IRRELEVANT = ""

#
# Interaction Models
#


def gen_zipflist(n):
    weightlist = [n / (i + 1) for i in range(n)]
    return weightlist


def gen_inverselist(n):
    weightlist = list(range(n))
    weightlist.reverse()
    return weightlist


def gen_reversezipflist(n):
    weightlist = gen_zipflist(n)
    weightlist.reverse()
    return weightlist


def gen_reverseinverselist(n):
    weightlist = list(range(n))
    return weightlist


def gen_uniformlist(n):
    weightlist = [1] * n
    return weightlist


#
# Lexicon Models
#


def gen_relevants_basic(ratio, numlemmas):
    return [int(ratio * x) for x in range(0, int(numlemmas / ratio))]


def gen_init_variants_basic(ratio, relevants):
    return [
        relevants[i]
        for i in [int(ratio * x) for x in range(0, int(len(relevants) / ratio))]
    ]


def gen_init_variants_absolute(irregnum, relevants):
    ratio = len(relevants) / irregnum
    return [
        relevants[i]
        for i in [int(ratio * x) for x in range(0, int(len(relevants) / ratio))]
    ]


def gen_init_variants_uniformrandom(irregnum, relevants):
    return random.sample(relevants, irregnum)


#
# Helper functions
#


def sample_n(samplelist, n):
    sampleds = []
    for i in range(0, n):
        samplei = random.randint(0, len(samplelist) - 1)
        sampleds.append(samplelist[samplei])
    return sampleds


def applytp(variants, goodvar, evar):
    e = variants.count(evar)
    N = variants.count(goodvar) + e
    if N < 2:
        return False
    return e < N / log(N)


def learn(relevants, variants):
    tp0 = applytp(variants, 0, 1)
    tp1 = applytp(variants, 1, 0)
    other = -1
    if tp0:
        other = 0
    elif tp1:
        other = 1
    learnedvariants = []
    numextended = 0
    for i, variant in enumerate(variants):
        if variant == 0 or variant == 1:
            learnedvariants.append(variant)
        elif i in relevants:
            learnedvariants.append(other)
            numextended += 1
        else:
            learnedvariants.append(IRRELEVANT)
    return learnedvariants


def get_categoricalvariant(variants):
    return int(sum(variants) / len(variants))


#
# Classes
#


class Lemmas(object):

    def __init__(self, N, relevants):
        self.N = N
        self.samplelist = list(range(N))
        self.weightlist = gen_zipflist(N)
        self.relevants = relevants

    def sample_n(self, n):
        return random.choices(self.samplelist, weights=self.weightlist, k=n)


class Individual(object):

    def __init__(self, lemmas, init_variants):
        self.lemmas = lemmas
        self.variants = init_variants
        self.inputvariants = {}

    def process_input(self, inputseq):
        for lemma, variant in inputseq:
            if lemma not in self.inputvariants:
                self.inputvariants[lemma] = []
            self.inputvariants[lemma].append(variant)

        categoricalvariants = [IRRELEVANT] * self.lemmas.N
        for i in self.lemmas.relevants:
            categoricalvariants[i] = -1
        for lemma, variants in self.inputvariants.items():
            categoricalvariants[lemma] = get_categoricalvariant(variants)
        self.variants = learn(self.lemmas.relevants, categoricalvariants)


class Community(object):

    def __init__(self, K, lemmas, init_variants, gen_interactionfunc):
        self.K = K
        self.lemmas = lemmas
        self.init_variants = set(init_variants)
        self.individuals = self.init_individuals(K, lemmas, init_variants)
        self.interactionfunc = gen_interactionfunc
        self.samplelist = list(range(K))
        self.weightlist = gen_interactionfunc(K)

    def sample_n(self, n):
        return random.choices(self.samplelist, weights=self.weightlist, k=n)

    def init_individuals(self, K, lemmas, init_variants):
        N = lemmas.N
        indivs = []
        for i in range(0, K):
            indivlemmas = [IRRELEVANT] * N
            for n in lemmas.relevants:
                indivlemmas[n] = 0
            for e in init_variants:
                indivlemmas[e] = 1
            indivs.append(Individual(lemmas, indivlemmas))
        return indivs

    def get_input(self, n):
        inputseq = []
        while len(inputseq) < n:
            sampledindivs = self.sample_n(n)
            sampledlemmas = self.lemmas.sample_n(n)
            for i in range(0, n):
                indiv = self.individuals[sampledindivs[i]].variants
                lemma = sampledlemmas[i]
                variant = indiv[lemma]
                if variant == 0 or variant == 1:
                    inputseq.append((lemma, variant))
        return inputseq


#
# Simulation functions
#


def create_individual(community, n):
    new_indiv = Individual(community.lemmas, [0] * community.lemmas.N)
    inputseq = community.get_input(n)
    new_indiv.process_input(inputseq)
    return new_indiv


def update_community(community, num_children, ncontinue, ninit):
    new_indiv = create_individual(community, ninit)  # Birth new child
    for i, child in enumerate(reversed(community.individuals)):
        if i < num_children:  # Update the children in ascending order of age
            inputseq = community.get_input(ncontinue)
            child.process_input(inputseq)
    community.individuals.append(new_indiv)  # Update the population
    community.individuals = community.individuals[1:]


def print_learner(community, indices, names):
    print("                     0  1")
    print("--------------------------")
    for index, name in zip(indices, names):
        variants = community.individuals[index].variants
        print(
            name,
            len(variants) - variants.count(-1) - variants.count(IRRELEVANT),
            variants.count(0),
            variants.count(1),
            [var for var in variants if var != IRRELEVANT],
        )


def iterate(community, num_iters, num_children, ninit, ncontinue):
    for i in range(0, num_iters):
        update_community(community, num_children, ninit, ncontinue)

    index = -1
    variants_mature = community.individuals[0 - num_children - 1].variants
    variants_young = community.individuals[-1].variants
    return community


def write_csv(f, data):
    for line in data:
        f.write(",".join([str(item) for item in line]) + "\n")


#
# Analysis functions
#


def freq_analysis_randomvariant(results, sample_indiv, num_iters):
    tokenfreqs_by_lemma = {}
    variants_by_lemma = {}
    variants_by_varlemma = {}
    variants_by_reglemma = {}
    analyzedlemmas = []
    for community in results:
        for i, individual in enumerate(community.individuals):
            for lemma, variants in individual.inputvariants.items():
                if lemma not in tokenfreqs_by_lemma:
                    tokenfreqs_by_lemma[lemma] = 0
                    variants_by_lemma[lemma] = []
                    variants_by_varlemma[lemma] = []
                    variants_by_reglemma[lemma] = []
                if i == sample_indiv:  # only the last developing
                    tokenfreqs_by_lemma[lemma] += len(variants)
                    variants_by_lemma[lemma].append(get_categoricalvariant(variants))
                    if lemma in community.init_variants:
                        variants_by_varlemma[lemma].append(
                            get_categoricalvariant(variants)
                        )
                    else:
                        variants_by_reglemma[lemma].append(
                            get_categoricalvariant(variants)
                        )

    rank = 0
    prevfreq = 999999999
    for lemma, tokenfreq in sorted(
        tokenfreqs_by_lemma.items(), key=lambda x: x[1], reverse=True
    ):
        init_variant = "FALSE"
        denom = len(variants_by_lemma[lemma])
        if not denom:
            denom = 1
        vardenom = len(variants_by_varlemma[lemma])
        if not vardenom:
            vardenom = 1
        regdenom = len(variants_by_reglemma[lemma])
        if not regdenom:
            regdenom = 1
        if lemma in community.init_variants:
            init_variant = "TRUE"
        analyzedlemmas.append(
            (
                community.interactionfunc.__name__,
                len(community.init_variants),
                community.lemmas.N,
                init_variant,
                lemma,
                rank,
                tokenfreq / num_iters,
                variants_by_lemma[lemma].count(1) / denom,
                variants_by_varlemma[lemma].count(1) / vardenom,
                variants_by_reglemma[lemma].count(1) / regdenom,
            )
        )
        if prevfreq > tokenfreq:
            rank += 1
        prevfreq = tokenfreq
    return analyzedlemmas


#
# Run several experimental trials simultaneously
#


def worker(
    seed,
    spent_communities,
    lemmas,
    num_lemmas,
    comm_size,
    num_children,
    ninit,
    ncontinue,
    relevants,
    irregnum,
    num_trials,
    num_iters,
    gen_interactionfunc,
):
    random.seed(seed)
    initvars = gen_init_variants_uniformrandom(
        int(irregnum), relevants
    )  # Irregulars are initially randomsly distributed
    community = Community(
        comm_size, lemmas, initvars, gen_interactionfunc
    )  # Creats a community according to the relevant parameters
    spent_communities.append(
        iterate(
            community,
            num_iters=num_iters,
            num_children=num_children,
            ninit=ninit,
            ncontinue=ncontinue,
        )
    )  # iterates and returns community's final state for analysis


def run_experiment_multiproc(
    num_lemmas,
    comm_size,
    num_children,
    ninit,
    ncontinue,
    relevants,
    irregnum,
    num_trials,
    num_iters,
    gen_interactionfunc,
):
    lemmas = Lemmas(num_lemmas, relevants)
    manager = Manager()
    spent_communities = manager.list([])
    maxp = 10  # NUMBER OF PROCESSES TO RUN IN PARALLEL
    completed = 0
    print("Trial:", end=" ")
    while completed < num_trials:
        ps = []
        for i in range(0, maxp):
            print(completed, end=" ")
            completed += 1
            p = Process(
                target=worker,
                args=(
                    completed,
                    spent_communities,
                    lemmas,
                    num_lemmas,
                    comm_size,
                    num_children,
                    ninit,
                    ncontinue,
                    relevants,
                    irregnum,
                    num_trials,
                    num_iters,
                    gen_interactionfunc,
                ),
            )
            ps.append(p)
            p.start()
        for p in ps:
            p.join()
        print()
    return spent_communities


#
#  EXPERIMENT PARAMETERIZATIONS
#


def exps_basicrandom():
    numlemmas = 100  # Lexicon size
    comm_size = 100  # Community size
    num_children = 3  # Number of learners
    num_iters = comm_size + num_children  # Number of itterations
    ninit = 500  # Number of interactions for a child's first iteration
    ncontinue = 1000  # Number of interactions for all subsequent iterations
    num_trials = 500  # number of Trials

    irregnums = [10, 20]  # Initial number of irregulars
    interactions = [
        gen_uniformlist,
        gen_reversezipflist,
        gen_reverseinverselist,
    ]  # Interaction distributions

    relevants = gen_relevants_basic(
        1, numlemmas
    )  # All items in the lexicon are relevant
    with open("exps_basicrandom.csv", "w") as f:
        f.write(
            "interaction,ratio,paradigm_size,init_variant,lemma,tokenrank,tokenfreq,variant_rate,init_variant_rate,noninit_variant_rate\n"
        )
        for interaction in interactions:
            for irregnum in irregnums:
                init_variants = gen_init_variants_uniformrandom(
                    int(irregnum), relevants
                )  # Irregulars are initially distributed uniformly
                print(len(relevants), len(init_variants))
                relevants = list(set(relevants))

                # Run the experiment for the set number of trials
                results = run_experiment_multiproc(
                    numlemmas,
                    comm_size,
                    num_children,
                    ninit,
                    ncontinue,
                    relevants,
                    irregnum,
                    num_trials,
                    num_iters,
                    interaction,
                )
                # Get and write out summary results for analysis
                analyzedlemmas = freq_analysis_randomvariant(
                    results, comm_size - num_children - 1, num_iters
                )
                write_csv(f, analyzedlemmas)


def exps_icscons():
    numlemmas = 100  # Lexicon size
    comm_size = 100  # Community size
    num_children = 3  # Number of learners
    num_iters = comm_size + num_children  # Number of itterations
    ninit = 500  # Number of interactions for a child's first iteration
    ncontinue = 1000  # Number of interactions for all subsequent iterations
    num_trials = 500  # number of Trials

    numlemmass = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # Lexicon sizes
    interactions = [
        gen_reverseinverselist,
        gen_uniformlist,
        gen_reversezipflist,
    ]  # Iteraction distributions

    with open("exps_icscons.csv", "w") as f:
        f.write(
            "interaction,ratio,paradigm_size,init_variant,lemma,tokenrank,tokenfreq,variant_rate,init_variant_rate,noninit_variant_rate\n"
        )
        for interaction in interactions:
            print(interaction)
            for numlemmas in numlemmass:
                print(numlemmas)
                ninit = numlemmas * 10
                ncontinue = numlemmas * 100
                theta = numlemmas / log(numlemmas)
                irregnums = [0.9 * theta]
                for irregnum in irregnums:
                    print("NUM IRREGS", irregnum)
                    relevants = gen_relevants_basic(
                        1, numlemmas
                    )  # All paradigm items are relevant

                    relevants = list(set(relevants))
                    results = run_experiment_multiproc(
                        numlemmas,
                        comm_size,
                        num_children,
                        ninit,
                        ncontinue,
                        relevants,
                        irregnum,
                        num_trials,
                        num_iters,
                        interaction,
                    )
                    analyzedlemmas = freq_analysis_randomvariant(
                        results, comm_size - num_children - 1, num_iters
                    )
                    write_csv(f, analyzedlemmas)


def main():
    # Lemma Frequency vs Regularization
    exps_basicrandom()
    # Paradigm Size vs Regularization
    exps_icscons()


if __name__ == "__main__":
    main()
