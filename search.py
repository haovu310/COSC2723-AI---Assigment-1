# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    from util import Stack  # Import the Stack data structure to use as the frontier (LIFO for DFS)

    # Initialize the frontier with a stack
    frontier = Stack()

    # Get the starting state from the problem
    start_state = problem.getStartState()

    # Push the starting state onto the stack along with an empty path (no actions taken yet)
    frontier.push((start_state, []))

    # Set of states that have been fully explored (visited and expanded)
    explored = set()

    # Set of states currently in the frontier (used to prevent duplicate pushes)
    frontier_states = set([start_state])

    # Begin the DFS search loop
    while not frontier.isEmpty():
        # Pop the most recently added state from the frontier
        state, path = frontier.pop()

        # Remove it from the frontier_states since we're now exploring it
        frontier_states.discard(state)

        # If the current state is the goal, return the path that got us here
        if problem.isGoalState(state):
            return path  # Success: return list of actions (e.g. ['North', 'East', ...])

        # Only expand this node if it hasn't been explored before
        if state not in explored:
            # Mark the state as explored so we don't revisit it
            explored.add(state)

            # Get all successors of the current state
            # Each successor is a (next_state, action, step_cost) tuple
            for successor, action, cost in reversed(problem.getSuccessors(state)):
                # Only consider successors that have not been explored
                # and are not already in the frontier (to avoid redundant work)
                # use the reversed function to ensure we traverse the left-most node first to align with the test case
                if successor not in explored and successor not in frontier_states:
                    # Push the new state and the path taken to reach it
                    frontier.push((successor, path + [action]))
                    # Also track that this state is now in the frontier
                    frontier_states.add(successor)

    # If the loop ends, it means no path to the goal was found
    # Return an empty list to indicate failure (optional in Pacman)
    return []


def breadthFirstSearch(problem):
    from util import Queue  # FIFO queue for BFS

    # Get the initial state of the problem
    start_state = problem.getStartState()

    # Special case: if the start is already the goal, return an empty path
    if problem.isGoalState(start_state):
        return []

    # Initialize the frontier with the starting node and path
    frontier = Queue()
    frontier.push((start_state, []))  # node = (state, path)

    # Set of visited/reached states
    reached = set([start_state])

    # Main BFS loop
    while not frontier.isEmpty():
        # Pop (Dequeue) the oldest node (FIFO)
        state, path = frontier.pop()

        # Expand the node: get its children (successors)
        for successor, action, cost in problem.getSuccessors(state):
            if successor not in reached:
                # Check if successor is goal before enqueuing
                if problem.isGoalState(successor):
                    return path + [action]  # Found goal

                # Mark as reached and enqueue the node
                reached.add(successor)
                frontier.push((successor, path + [action]))

    # No path found to the goal
    return []



def uniformCostSearch(problem):
    from util import PriorityQueue  # Uses priority queue for cost-based expansion

    # Step 1: Get the start state
    start_state = problem.getStartState()

    # Step 2: Special case — start is already the goal
    if problem.isGoalState(start_state):
        return []

    # Step 3: Initialize the frontier with a priority queue
    # Each item: (state, path); priority = path cost so far
    frontier = PriorityQueue()
    frontier.push((start_state, []), 0)  # Priority = 0 for start node

    # Step 4: Track visited states and their best known cost
    visited_cost = {start_state: 0}  # Maps state → cost

    # Step 5: UCS loop
    while not frontier.isEmpty():
        state, path = frontier.pop()

        # Calculate total cost of path to this state
        current_cost = problem.getCostOfActions(path)

        # Step 6: Check for goal
        if problem.isGoalState(state):
            return path  # ✅ Return complete path to goal

        # Step 7: Explore successors
        for successor, action, step_cost in problem.getSuccessors(state):
            new_path = path + [action]
            new_cost = problem.getCostOfActions(new_path)

            # Only add to frontier if:
            # (1) It's a new state OR
            # (2) We found a cheaper path than before
            if successor not in visited_cost or new_cost < visited_cost[successor]:
                visited_cost[successor] = new_cost
                frontier.push((successor, new_path), new_cost)

    # Step 8: Return failure if no solution found
    return []



def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):

    from util import PriorityQueue

    # Step 1: Get the start state
    start_state = problem.getStartState()

    # Step 2: Edge case — start is goal
    if problem.isGoalState(start_state):
        return []

    # Step 3: Initialize frontier with start node; priority = f(n) = 0 + h(start)
    frontier = PriorityQueue()
    frontier.push((start_state, []), heuristic(start_state, problem))

    # Step 4: Track lowest cost to each visited state
    visited_cost = {start_state: 0}  # g(n)

    # Step 5: Main loop
    while not frontier.isEmpty():
        state, path = frontier.pop()
        current_cost = problem.getCostOfActions(path)  # g(n)

        # Step 6: Goal check
        if problem.isGoalState(state):
            return path

        # Step 7: Expand node
        for successor, action, step_cost in problem.getSuccessors(state):
            new_path = path + [action]
            new_cost = problem.getCostOfActions(new_path)  # g(n) to successor

            # Only consider if not visited or found a cheaper path
            if successor not in visited_cost or new_cost < visited_cost[successor]:
                visited_cost[successor] = new_cost

                # Compute f(n) = g(n) + h(n)
                priority = new_cost + heuristic(successor, problem)

                frontier.push((successor, new_path), priority)

    # Step 8: No path found
    return []



#####################################################
# EXTENSIONS TO BASE PROJECT
#####################################################

# Extension Q1e
def iterativeDeepeningSearch(problem):
    """Search the deepest node in an iterative manner."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


#####################################################
# Abbreviations
#####################################################
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
ids = iterativeDeepeningSearch
