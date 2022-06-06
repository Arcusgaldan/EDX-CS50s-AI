import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    page_links = corpus[page]
    prob_linked_page = damping_factor / len(page_links)
    prob_random_page = (1 - damping_factor) / len(corpus)
    distribution = {}
    for given_page in corpus:
        distribution[given_page] = prob_random_page
        if given_page in page_links:
            distribution[given_page] += prob_linked_page
    return distribution
    
        


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    distribution = dict.fromkeys(list(corpus), 0)
    current_page = random.sample(list(corpus), 1)[0]
    distribution[current_page] += 1
    
    for i in range(n-1):
        model = transition_model(corpus, current_page, damping_factor)
        next_page = random.choices(list(model), weights=model.values(), k=1)[0]
        distribution[next_page] += 1
        current_page = next_page
    
    return {k:v/n for k,v in distribution.items()}

def get_linking_pages(corpus, page):
    linking_pages = set()
    for given_page in corpus:
        if len(corpus[given_page]) == 0:
            linking_pages.add(given_page)
        elif page in corpus[given_page]:
            linking_pages.add(given_page)
            
    return linking_pages

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    distribution = dict.fromkeys(list(corpus), 1/len(corpus))
    
    flagNormalized = False
    while not flagNormalized:
        flagNormalized = True
        for page in corpus:
            new_prob = (1 - damping_factor) / len(corpus)
            sum_prob_link = 0
            linking_pages = get_linking_pages(corpus, page)
            if linking_pages is not None:
                for linking_page in linking_pages:
                    sum_prob_link += distribution[linking_page] / len(corpus[linking_page])
            new_prob += damping_factor * sum_prob_link
            if abs(new_prob - distribution[page]) > 0.001:
                flagNormalized = False
            #input()
            distribution[page] = new_prob
    
    return distribution
    


if __name__ == "__main__":
    main()
