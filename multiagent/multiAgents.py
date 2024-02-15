# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foods = newFood.asList()
        if len(foods) == 0:
            nearest_food = 0
        else:
            nearest_food = 9999
            for food in foods:
                dist = manhattanDistance(newPos, food)
                if dist < nearest_food:
                    nearest_food = dist

        nearest_ghost = 9999
        for ghost in newGhostStates:
            dist = manhattanDistance(ghost.getPosition(), newPos)
            scared = ghost.scaredTimer
            if (scared == 0) and (dist < nearest_ghost):
                nearest_ghost = dist


        if nearest_ghost == 0:
            return successorGameState.getScore() - 1
        elif nearest_food == 0:
            return successorGameState.getScore() + 1

        return successorGameState.getScore() + 10 / nearest_food - 10 / nearest_ghost

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def value(self, gameState, depth, agentNum):
        """if the state is a terminal state: return the state’s Ulity
        if the next agent is MAX: return max - value(state)
        if the next agent is MIN: return min - value(state)"""
        if (depth == 0) or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP
        elif agentNum == 0:
            return self.maxValue(gameState, depth, agentNum)
        else:
            return self.minValue(gameState, depth, agentNum)

    def maxValue(self, gameState, depth, agentNum):
        legalMoves = gameState.getLegalActions(agentNum)
        if agentNum == gameState.getNumAgents() - 1:
            new_depth, new_agentNum = depth - 1, 0
        else:
            new_depth, new_agentNum = depth, agentNum + 1

        v = -float('inf')
        action = Directions.STOP
        for move in legalMoves:
            successor = gameState.generateSuccessor(agentNum, move)
            new_v = self.value(successor, new_depth, new_agentNum)[0]
            if new_v > v:
                v = new_v
                action = move

        return v, action

    def minValue(self, gameState, depth, agentNum):
        legalMoves = gameState.getLegalActions(agentNum)
        if agentNum == gameState.getNumAgents() - 1:
            new_depth, new_agentNum = depth - 1, 0
        else:
            new_depth, new_agentNum = depth, agentNum + 1

        v = float('inf')
        action = Directions.STOP
        for move in legalMoves:
            successor = gameState.generateSuccessor(agentNum, move)
            new_v = self.value(successor, new_depth, new_agentNum)[0]
            if new_v < v:
                v = new_v
                action = move

        return v, action

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        return self.value(gameState, self.depth, 0)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def value(self, gameState, depth, agentNum, alpha, beta):
        """if the state is a terminal state: return the state’s Ulity
        if the next agent is MAX: return max - value(state)
        if the next agent is MIN: return min - value(state)"""
        if (depth == 0) or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP
        elif agentNum == 0:
            return self.maxValue(gameState, depth, agentNum, alpha, beta)
        else:
            return self.minValue(gameState, depth, agentNum, alpha, beta)

    def maxValue(self, gameState, depth, agentNum, alpha, beta):
        legalMoves = gameState.getLegalActions(agentNum)
        if agentNum == gameState.getNumAgents() - 1:
            new_depth, new_agentNum = depth - 1, 0
        else:
            new_depth, new_agentNum = depth, agentNum + 1

        v = -float('inf')
        action = Directions.STOP
        for move in legalMoves:
            successor = gameState.generateSuccessor(agentNum, move)
            new_v = self.value(successor, new_depth, new_agentNum, alpha, beta)[0]
            if new_v > v:
                v = new_v
                action = move
            if new_v > beta:
                return new_v, action
            alpha = max(alpha, new_v)

        return v, action

    def minValue(self, gameState, depth, agentNum, alpha, beta):
        legalMoves = gameState.getLegalActions(agentNum)
        if agentNum == gameState.getNumAgents() - 1:
            new_depth, new_agentNum = depth - 1, 0
        else:
            new_depth, new_agentNum = depth, agentNum + 1

        v = float('inf')
        action = Directions.STOP
        for move in legalMoves:
            successor = gameState.generateSuccessor(agentNum, move)
            new_v = self.value(successor, new_depth, new_agentNum, alpha, beta)[0]
            if new_v < v:
                v = new_v
                action = move
            if new_v < alpha:
                return new_v, action
            beta = min(beta, new_v)

        return v, action
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, self.depth, 0, -float('inf'), float('inf'))[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def value(self, gameState, depth, agentNum):
        if (depth == 0) or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP
        if agentNum == 0:
            return self.maxValue(gameState, depth, agentNum)
        if agentNum > 0:
            return self.expValue(gameState, depth, agentNum)

    def maxValue(self, gameState, depth, agentNum):
        legalMoves = gameState.getLegalActions(agentNum)
        if agentNum == gameState.getNumAgents() - 1:
            new_depth, new_agentNum = depth - 1, 0
        else:
            new_depth, new_agentNum = depth, agentNum + 1

        max_v = -float('inf')
        action = Directions.STOP
        for move in legalMoves:
            successor = gameState.generateSuccessor(agentNum, move)
            curr_val = self.value(successor, new_depth, new_agentNum)[0]
            if curr_val > max_v:
                max_v = curr_val
                action = move
        return max_v, action

    def expValue(self, gameState, depth, agentNum):
        prob = 0
        legalMoves = gameState.getLegalActions(agentNum)
        if agentNum == gameState.getNumAgents() - 1:
            new_depth, new_agentNum = depth - 1, 0
        else:
            new_depth, new_agentNum = depth, agentNum + 1

        exp = 0
        action = Directions.STOP
        for move in legalMoves:
            successor = gameState.generateSuccessor(agentNum, move)
            exp += self.value(successor, new_depth, new_agentNum)[0]
            prob += 1
        action = move
        return exp/prob, action

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, self.depth, 0)[1]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    currentFood = newFood.asList()
    score = float(0)

    #eat ghost or run away?
    dist_to_ghosts = list()
    for ghost in newGhostStates:
        dist = manhattanDistance(ghost.getPosition(), newPos)
        scaredTime = ghost.scaredTimer
        if dist < scaredTime:
            score += 200 * dist
        dist_to_ghosts.append(dist)
    score += min(dist_to_ghosts)
            
    #distance to food?
    dist_to_food = list()
    for food in currentFood:
        dist = manhattanDistance(food, newPos)
        dist_to_food.append(dist)
    if len(dist_to_food) > 0:
        score -= max(dist_to_food)

    return score

# Abbreviation
better = betterEvaluationFunction
