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

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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

    def evaluationFunction(self, currentGameState, action):
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
        newCapsules = successorGameState.getCapsules()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        foodCount = 0
        minFoodDist = float('inf')
        minGhostDist = float('inf')
        timeCount = sum(newScaredTimes)
        foodList = newFood.asList()
        minCapDist = float('inf')
        if (successorGameState.isWin() | successorGameState.isLose()):
            return successorGameState.getScore()
        
        for food in foodList:
            tempFoodDist = abs(newPos[0] - food[0]) + abs(newPos[1] - food[1])
            if (minFoodDist > tempFoodDist):
                minFoodDist = tempFoodDist
            foodCount += 1

        for cap in newCapsules:
            tempCapDist = abs(newPos[0] - cap[0]) + abs(newPos[1] - cap[1])
            if (minCapDist > tempCapDist):
                minCapDist = tempCapDist
            foodCount += 2
        for ghost in newGhostStates:
            ghostPos = ghost.getPosition()
            tempGhostDist = abs(newPos[0] - ghostPos[0]) + abs(newPos[1] - ghostPos[1])
            if (minGhostDist > tempGhostDist):
                minGhostDist = tempGhostDist
        if (minGhostDist < 2):
            return -999999
        return successorGameState.getScore() + timeCount + 1/minGhostDist + 1/minFoodDist + 1/minCapDist - foodCount
        '''
        foodCount = 0
        for i in newFood:
            for j in i:
                if j:
                    foodCount += 1
        maxDist = 0
        for ghost in newGhostStates:
            ghostPos = ghost.getPosition()
            temp = abs(newPos[0] - ghostPos[0]) + abs(newPos[1] - ghostPos[1])
            maxDist += temp
        timeCount = 0
        for time in newScaredTimes:
            timeCount += time
        if (maxDist < 2):
            return -99999
        if (foodCount == 0):
            return 999999
        return timeCount + successorGameState.getScore() - foodCount + 1/maxDist - len(newCapsules)'''

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        value = -float('inf')
        opMove = None

        for action in gameState.getLegalActions():
            state = gameState.generateSuccessor(0, action)
            tempVal = self.valuation(state, 1, self.depth)

            if (tempVal > value):
                value = tempVal
                opMove = action
        return opMove

    def valuation(self, gameState, turn, numMoves):
        numAgents = gameState.getNumAgents()
        if (turn == numAgents):
            numMoves -= 1
            turn = turn % numAgents
        if numMoves == 0:
            return self.evaluationFunction(gameState)
        if gameState.isWin() | gameState.isLose():
            return self.evaluationFunction(gameState)
        if (turn == 0):
            return self.maximizer(gameState, turn, numMoves)
        else:
            return self.minimizer(gameState, turn, numMoves)


    def maximizer(self, state, turn, numMoves):

        value = -float('inf')
        for action in state.getLegalActions():
            successor = state.generateSuccessor(turn, action)
            tempVal = self.valuation(successor, turn + 1, numMoves)
            if (tempVal > value):
                value = tempVal
        return value



    def minimizer(self, state, turn, numMoves):

        value = float('inf')
        for action in state.getLegalActions(turn):
            successor = state.generateSuccessor(turn, action)
            tempVal = self.valuation(successor, turn + 1, numMoves)
            if (tempVal < value):
                value = tempVal
        return value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        value = -float('inf')
        opMove = None
        alpha = -float('inf')
        for action in gameState.getLegalActions():
            state = gameState.generateSuccessor(0, action)
            tempVal = self.findOptimal(state, 1, self.depth, alpha, float('inf'))

            if (tempVal > value):
                value = tempVal
                opMove = action
            if (value > alpha):
                alpha = value

        return opMove
    
    def findOptimal(self, state, turn, depth, alpha, beta):
        numAgents = state.getNumAgents()
        if (turn == numAgents):
            depth -= 1
            turn = turn % numAgents
        if depth == 0:
            return self.evaluationFunction(state)
        if state.isWin() | state.isLose():
            return self.evaluationFunction(state)
        if (turn == 0):
            return self.alphaMax(state, turn, depth, alpha, beta)
        else:
            return self.betaMin(state, turn, depth, alpha, beta)

    def alphaMax(self, state, turn, depth, alpha, beta):
        value = -float('inf')
        for action in state.getLegalActions():
            successor = state.generateSuccessor(turn, action)
            tempVal = self.findOptimal(successor, turn + 1, depth, alpha, beta)
            if (tempVal > value):
                value = tempVal
            if (value > beta):
                return value
            if (value > alpha):
                alpha = value
        return value

    def betaMin(self, state, turn, depth, alpha, beta):
        value = float('inf')
        for action in state.getLegalActions(turn):
            successor = state.generateSuccessor(turn, action)
            tempVal = self.findOptimal(successor, turn + 1, depth, alpha, beta)
            if (tempVal < value):
                value = tempVal
            if (value < alpha):
                return value
            if (value < beta):
                beta = value
        return value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        value = -float('inf')
        opMove = None

        for action in gameState.getLegalActions():
            state = gameState.generateSuccessor(0, action)
            tempVal = self.subOptimal(state, 1, self.depth)
            if (tempVal > value):
                value = tempVal
                opMove = action
        return opMove

    def subOptimal(self, state, turn, depth):
        numAgents = state.getNumAgents()
        if (turn == numAgents):
            depth -= 1
            turn = turn % numAgents
        if depth == 0:
            return self.evaluationFunction(state)
        if state.isWin() | state.isLose():
            return self.evaluationFunction(state)
        if (turn == 0):
            return self.expectiMax(state, turn, depth)
        else:
            return self.expectiMin(state, turn, depth)

    def expectiMax(self, state, turn, depth):
        value = -float('inf')
        for action in state.getLegalActions():
            successor = state.generateSuccessor(turn, action)
            tempVal = self.subOptimal(successor, turn + 1, depth)
            if (tempVal > value):
                value = tempVal
        return value

    def expectiMin(self, state, turn, depth):
        value = 0
        legalActions = state.getLegalActions(turn)
        for action in legalActions:
            successor = state.generateSuccessor(turn, action)
            p = 1/(len(legalActions))
            value += p*self.subOptimal(successor, turn + 1, depth)
        return value

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <I hope this works.>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return 99999
    if currentGameState.isLose():
        return -99999
    pacPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    numFood = 0
    minCapDist = float('inf')
    maxDistance = -float('inf')
    for i in food:
        for j in i:
            if (j == True):
                numFood += 1
    if capsules != None:
        for capPos in capsules:
            temp = abs(pacPos[0] - capPos[0]) + abs(pacPos[1] - capPos[1])
            if (temp > minCapDist):
                minCapDist = temp
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        tempDist = abs(pacPos[0] - ghostPos[0]) + abs(pacPos[1] - ghostPos[1])
        if (tempDist > maxDistance):
            maxDistance = tempDist
        if (maxDistance == 0):
            maxDistance = -float('inf')
    return currentGameState.getScore() + 1/maxDistance + 1/minCapDist - numFood - 2*len(capsules)
    

# Abbreviation
better = betterEvaluationFunction
