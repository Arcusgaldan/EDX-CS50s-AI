import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for var in self.crossword.variables:
            for domain_word in self.domains[var].copy():
                if len(domain_word) != var.length:
                    self.domains[var].remove(domain_word)

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revised = False
        overlap = self.crossword.overlaps[x, y]
        if overlap is None:
            return False
        x_overlap = overlap[0]
        y_overlap = overlap[1]
        for x_domain_word in self.domains[x].copy():
            y_possible_words = [y_domain_word for y_domain_word in self.domains[y] if x_domain_word[x_overlap] == y_domain_word[y_overlap]]
            if len(y_possible_words) == 0:
                self.domains[x].remove(x_domain_word)
                revised = True
        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        if arcs is None:
            arcs = []
            for var in self.crossword.variables:
                for neighbor in list(self.crossword.neighbors(var)):
                    arcs.append((var, neighbor))
        else:
            arcs = list(arcs)
        while len(arcs) > 0:
            arc = arcs[0]
            arcs.remove(arc)
            revised = self.revise(arc[0], arc[1])
            if revised:
                if len(self.domains[arc[0]]) == 0:
                    return False
                for neighbor in self.crossword.neighbors(arc[0]):
                    arcs.append((neighbor, arc[0]))
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        if len(assignment) == len(self.crossword.variables):
            return True
        return False
    
    def assignment_has_duplicates(self, assignment):
        for var in assignment:
            if list(assignment.values()).count(assignment[var]) > 1:
                return True
        return False
    
    def has_empty_domains(self):
        for var in self.domains:
            if len(self.domains[var]) == 0:
                return True
        return False

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        if self.assignment_has_duplicates(assignment):
            return False
        
        old_domains = self.domains.copy()
        for var in assignment:
            self.domains[var] = [assignment[var]]
        
        self.enforce_node_consistency()
        if self.has_empty_domains():
            self.domains = old_domains
            return False
        
        if not self.ac3():
            self.domains = old_domains
            return False
        
        self.domains = old_domains
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        domain_words = self.domains[var]
        sorted_domain_words = []
        non_assigned_neighbors = [neighbor for neighbor in self.crossword.neighbors(var) if neighbor not in assignment]
        for word in domain_words:
            constraints = 0
            for neighbor in non_assigned_neighbors:
                var1_overlap, var2_overlap = self.crossword.overlaps[var, neighbor]
                for neighbor_word in self.domains[neighbor]:
                    if word[var1_overlap] != neighbor_word[var2_overlap]:
                        constraints += 1
            sorted_domain_words.append({'word': word, 'constraints': constraints})
        sorted_domain_words = sorted(sorted_domain_words, key= lambda par: par['constraints'])
        return [par['word'] for par in sorted_domain_words]

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        if self.assignment_complete(assignment):
            return None
        unassigned_variables = [var for var in self.crossword.variables if var not in assignment]
        smallest_domain_length = None
        smallest_domain_vars = []
        for var in unassigned_variables:
            if smallest_domain_length is None:
                smallest_domain_length = len(self.domains[var])
                smallest_domain_vars.append(var)
                continue
            
            if len(self.domains[var]) < smallest_domain_length:
                smallest_domain_length = len(self.domains[var])
                smallest_domain_vars = [var]
            elif len(self.domains[var]) == smallest_domain_length:
                smallest_domain_vars.append(var)
        
        if len(smallest_domain_vars) == 1:
            return smallest_domain_vars[0]
        
        highest_degree_var = None
        for var in smallest_domain_vars:
            if highest_degree_var is None:
                highest_degree_var = var
                continue
            
            if len(self.crossword.neighbors(var)) > len(self.crossword.neighbors(highest_degree_var)):
                highest_degree_var = var
        
        return highest_degree_var
            
    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment
        
        unassigned_var = self.select_unassigned_variable(assignment)
        sorted_domain_words = self.order_domain_values(unassigned_var, assignment)
        for word in sorted_domain_words:
            new_assignment = assignment
            new_assignment[unassigned_var] = word
            if self.consistent(new_assignment):
                old_domains = self.domains.copy()
                self.enforce_node_consistency()
                self.ac3()
                result = self.backtrack(new_assignment)
                if result is not None:
                    return result
                else:
                    self.domains = old_domains
        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
