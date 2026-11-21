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

        score = successorGameState.getScore()

        """ Them cac tinh nang tinh diem dua tren food, ma, va banh """
        
        # 1. Dua theo food gan nhat
        foodList = newFood.asList()
        if foodList:
            minfoodDist = 9999
            for food in foodList:
                minfoodDist = min(manhattanDistance(newPos,food),minfoodDist)
            score += 10.0 / (minfoodDist+1)
        
        # 2. Dua tren ma
        for ghost, scaredTime in zip(newGhostStates,newScaredTimes):
            ghostPos = ghost.getPosition()

            if ghostPos is None:
                continue

            ghostDist = manhattanDistance(newPos,ghostPos)

            if scaredTime > 0:
                score += 5.0 / (ghostDist+1)

            else:
                if ghostDist <= 1:
                    score -= 300
                else :
                    score -= 5.0 / ghostDist

        # 3. Khuyen khich theo food

        score -= 4 * len(foodList)

        return score

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
        numAgents = gameState.getNumAgents()

        def minimax(state, agentIndex, depth):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)

            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            if agentIndex == 0:
                bestValue = float("-inf")
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = minimax(successor, nextAgent, nextDepth)

                    if value > bestValue:
                        bestValue = value

                return bestValue

            bestValue = float("inf")
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                value = minimax(successor, nextAgent, nextDepth)

                if value < bestValue:
                    bestValue = value

            return bestValue

        legalActions = gameState.getLegalActions(0)

        if not legalActions:
            return Directions.STOP

        bestAction = legalActions[0]
        bestValue = float("-inf")

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)

            value = minimax(
                successor,
                1 % numAgents,
                1 if numAgents == 1 else 0
            )

            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(state, depth, agentIndex):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()
            # Chuyen sang agent tiep theo
            nextAgent = (agentIndex + 1) % numAgents

            # Chi tang depth sau khi tat ca ghosts da di
            nextDepth = depth + 1 if nextAgent == 0 else depth

            # Kiem tra do sau
            if depth == self.depth and agentIndex == 0:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)

            if not legalActions:
                return self.evaluationFunction(state)

            # Pacman: MAX node
            if agentIndex == 0:

                values = []

                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)

                    value = expectimax(
                        successor,
                        nextDepth,
                        nextAgent
                    )

                    values.append(value)

                return max(values)

            # Ghost: CHANCE node
            else:

                values = []

                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)

                    value = expectimax(
                        successor,
                        nextDepth,
                        nextAgent
                    )

                    values.append(value)

                return sum(values) / len(values)

        # Tim action tot nhat cho Pacman
        legalActions = gameState.getLegalActions(0)

        if not legalActions:
            return Directions.STOP

        bestAction = None
        bestValue = float('-inf')

        for action in legalActions:

            successor = gameState.generateSuccessor(0, action)

            value = expectimax(successor, 0, 1)

            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction
            
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    score = currentGameState.getScore()

    # 1. Danh gia Food
    if foodList:
        minFoodDist = min(
            manhattanDistance(pacmanPos, food)
            for food in foodList
        )

        score += 10.0 / (minFoodDist + 1)

        # Phat neu con nhieu Food
        score -= 4 * len(foodList)

    # 2. Danh gia Capsule
    if capsules:
        minCapsuleDist = min(
            manhattanDistance(pacmanPos, capsule)
            for capsule in capsules
        )

        score += 15.0 / (minCapsuleDist + 1)

    # 3. Danh gia Ghost
    for ghost in ghostStates:

        ghostPos = ghost.getPosition()

        if ghostPos is None:
            continue

        ghostDist = manhattanDistance(pacmanPos, ghostPos)

        if ghost.scaredTimer > 0:
            # Ghost dang so: khuyen khich truy duoi
            score += 20.0 / (ghostDist + 1)

        else:
            # Ghost binh thuong: tranh xa
            if ghostDist <= 1:
                score -= 500
            else:
                score -= 10.0 / ghostDist

    return score

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
