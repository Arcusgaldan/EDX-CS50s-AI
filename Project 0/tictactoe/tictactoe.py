"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    if terminal(board):
        return None
    ocount = 0
    xcount = 0
    for row in board:
        for slot in row:
            if slot == X:
                xcount += 1
            elif slot == O:
                ocount += 1
    
    if xcount > ocount:
        return O
    else:
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    if terminal(board):
        return None
    actions = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                actions.append((i, j))
    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if terminal(board) or board[action[0]][action[1]] != EMPTY or action[0] > 2 or action[0] < 0 or action[1] > 2 or action[1] < 0:
        raise Exception("Invalid action")
    newBoard = copy.deepcopy(board)
    token = player(board)
    newBoard[action[0]][action[1]] = token
    return newBoard
    

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    winningPositions = [[(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 1), (1, 2)], [(2, 0), (2, 1), (2, 2)], 
                        [(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 1), (2, 1)], [(0, 2), (1, 2), (2, 2)], 
                        [(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)]]
    
    for winningPosition in winningPositions:
        if board[winningPosition[0][0]][winningPosition[0][1]] != EMPTY and board[winningPosition[0][0]][winningPosition[0][1]] == board[winningPosition[1][0]][winningPosition[1][1]] == board[winningPosition[2][0]][winningPosition[2][1]]:
            if board[winningPosition[0][0]][winningPosition[0][1]] == X:
                return X
            else:
                return O
    
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    winningPositions = [[(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 1), (1, 2)], [(2, 0), (2, 1), (2, 2)], 
                        [(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 1), (2, 1)], [(0, 2), (1, 2), (2, 2)], 
                        [(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)]]
    
    flagFull = True
    for row in board:
        for slot in row:
            if slot == EMPTY:
                flagFull = False
                break
    
    if flagFull:
        return True
    
    for winningPosition in winningPositions:
        if board[winningPosition[0][0]][winningPosition[0][1]] != EMPTY and board[winningPosition[0][0]][winningPosition[0][1]] == board[winningPosition[1][0]][winningPosition[1][1]] == board[winningPosition[2][0]][winningPosition[2][1]]:
            return True
    
    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    utility = winner(board)
    if utility == X:
        return 1
    elif utility == O:
        return -1
    else:
        return 0
    
def possibleResults(board):
    possibleActions = actions(board)
    possibleResults = []
    for possibleAction in possibleActions:
        possibleResult = {'board': result(board, possibleAction), 'action': possibleAction}
        possibleResult['utility'] = utility(possibleResult['board'])
        possibleResults.append(possibleResult)
    return possibleResults

def isNewBestUtility(bestUtility, currentUtility, flagMin):
    if flagMin:
        if currentUtility < bestUtility:
            return True
    else:
        if currentUtility > bestUtility:
            return True        
    return False

def recursiveMinimax(board, flagMin, starter=False):
    bestUtility = None
    bestAction = None
    if terminal(board):
        if starter:
            return None, None
        else:
            return utility(board)
    
    for action in actions(board):
        nextUtility = recursiveMinimax(result(board, action), not flagMin)
        if bestUtility is None:
            bestUtility = nextUtility
            bestAction = action
        elif isNewBestUtility(bestUtility, nextUtility, flagMin):
            bestUtility = nextUtility
            bestAction = action
            
    if starter:
        return bestUtility, bestAction
    else:
        return bestUtility
                
    
        


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if player(board) == X:
        flagMin = False
    else:
        flagMin = True
    
    bestUtility, bestAction = recursiveMinimax(board, flagMin, starter=True)
    return bestAction
    