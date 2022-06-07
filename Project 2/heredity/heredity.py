import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]

def get_amount_genes(person, one_gene, two_genes):
    genes = None
    if person in one_gene:
        genes = 1
    elif person in two_genes:
        genes = 2
    else:
        genes = 0
    return genes

def probability_passes_gene(gene_amount):
    if gene_amount == 0:
        return PROBS['mutation']
    elif gene_amount == 2:
        return 1 - PROBS['mutation']
    else:
        return 0.5
    

def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    personal_possibilities = dict.fromkeys(list(people), None)
    for person in people:
        genes = get_amount_genes(person, one_gene, two_genes)
        has_trait = True if person in have_trait else False
        if people[person]['mother'] is None:
            personal_possibilities[person] = PROBS['gene'][genes] * PROBS['trait'][genes][has_trait]
        else:
            mother_genes = get_amount_genes(people[person]['mother'], one_gene, two_genes)
            father_genes = get_amount_genes(people[person]['father'], one_gene, two_genes)
            gets_from_mother = probability_passes_gene(mother_genes)
            gets_from_father = probability_passes_gene(father_genes)
            genes_probability = 0
            if genes == 0:
                genes_probability = (1 - gets_from_mother) * (1 - gets_from_father) #Doesn't get from mother AND doesn't get from father
            elif genes == 2:
                genes_probability = gets_from_mother * gets_from_father #Gets from mother AND gets from father
            else:
                genes_probability = (gets_from_mother * (1 - gets_from_father)) + (gets_from_father * 1 - gets_from_mother) #Gets from mother AND doesn't get from father OR gets from father and doesn't get from mother
            personal_possibilities[person] = genes_probability * PROBS['trait'][genes][has_trait]
    
    joint_prob = 1
    for person in personal_possibilities:
        joint_prob *= personal_possibilities[person]
    return joint_prob
        
        


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        genes = get_amount_genes(person, one_gene, two_genes)
        has_trait = True if person in have_trait else False
        probabilities[person]['gene'][genes] += p
        probabilities[person]['trait'][has_trait] += p


def get_sum_1(collection):
    sumValues = 0
    for key in collection:
        sumValues += collection[key]
    for key in collection:
        collection[key] = collection[key] / sumValues

def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        get_sum_1(probabilities[person]['gene'])
        get_sum_1(probabilities[person]['trait'])


if __name__ == "__main__":
    main()
