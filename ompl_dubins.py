#!/usr/bin/env python

# from os.path import abspath, dirname, join
import sys

try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import tools as ot
except ImportError:
    # if the ompl module is not in the PYTHONPATH add subdirectory "py-bindings" to path.
    sys.path.insert(0,'/home/OMPLPATH/py-bindings')
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import tools as ot

import math
from math import sqrt
import yaml, time, argparse, csv, os
from functools import partial
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt


def getObstacleSet(obstacleMap):
    return pkl.load(open(obstacleMap, 'rb'))


class ValidityChecker(ob.StateValidityChecker):
    def __init__(self, si, obstacleList):
        super(ValidityChecker, self).__init__(si)
        self.obstacles = obstacleList

    # Returns whether the given state's position overlaps the obstacles
    def isValid(self, state):

        x = state.getX()
        y = state.getY()
        statePoint = Point(x,y)

        for obstacle in self.obstacles:
            if statePoint.within(obstacle) or statePoint.intersects(obstacle):
                return False

        return True


    # Returns the distance from the given state's position to the boundary of the obstacle.
    def clearance(self, state):
        if len(self.obstacles) == 0:
            return 1

        # Extract the robot's (x,y) position from its state
        x = state.getX()
        y = state.getY()
        statePoint = Point(x,y)

        for obstacle in self.obstacles:
            # Distance between statePoint and the obstacle exterior
            clearance += obstacle.exterior.distance(statePoint)
            if clearance <= 0:
                return 0

        return clearance


def getPathLengthObjective(si):
    return ob.PathLengthOptimizationObjective(si)


def getThresholdPathLengthObj(si):
    obj = ob.PathLengthOptimizationObjective(si)
    obj.setCostThreshold(ob.Cost(1.51))
    return obj


class ClearanceObjective(ob.StateCostIntegralObjective):
    def __init__(self, si):
        super(ClearanceObjective, self).__init__(si, True)
        self.si_ = si

    # Our requirement is to maximize path clearance from obstacles,
    # but we want to represent the objective as a path cost
    # minimization. Therefore, we set each state's cost to be the
    # reciprocal of its clearance, so that as state clearance
    # increases, the state cost decreases.
    def stateCost(self, s):
        return ob.Cost(1 / (self.si_.getStateValidityChecker().clearance(s) +
                            sys.float_info.min))


def getClearanceObjective(si):
    return ClearanceObjective(si)


def getBalancedObjective1(si):
    lengthObj = ob.PathLengthOptimizationObjective(si)
    clearObj = ClearanceObjective(si)

    opt = ob.MultiOptimizationObjective(si)
    opt.addObjective(lengthObj, 5.0)
    opt.addObjective(clearObj, 1.0)

    return opt


def getPathLengthObjWithCostToGo(si):
    obj = ob.PathLengthOptimizationObjective(si)
    obj.setCostToGoHeuristic(ob.CostToGoHeuristic(ob.goalRegionCostToGo))
    return obj


# Keep these in alphabetical order and all lower case
def allocatePlanner(si, plannerType):
    if plannerType.lower() == "bfmtstar":
        return og.BFMT(si)
    elif plannerType.lower() == "bitstar":
        return og.BITstar(si)
    elif plannerType.lower() == "fmtstar":
        return og.FMT(si)
    elif plannerType.lower() == "informedrrtstar":
        return og.InformedRRTstar(si)
    elif plannerType.lower() == "prmstar":
        return og.PRMstar(si)
    elif plannerType.lower() == "rrtstar":
        return og.RRTstar(si)
    elif plannerType.lower() == "rrt":
        return og.RRT(si)
    elif plannerType.lower() == "sorrtstar":
        return og.SORRTstar(si)
    else:
        ou.OMPL_ERROR("Planner-type is not implemented in allocation function.")


# Keep these in alphabetical order and all lower case
def allocateObjective(si, objectiveType):
    if objectiveType.lower() == "pathclearance":
        return getClearanceObjective(si)
    elif objectiveType.lower() == "pathlength":
        return getPathLengthObjective(si)
    elif objectiveType.lower() == "thresholdpathlength":
        return getThresholdPathLengthObj(si)
    elif objectiveType.lower() == "weightedlengthandclearancecombo":
        return getBalancedObjective1(si)
    else:
        ou.OMPL_ERROR("Optimization-objective is not implemented in allocation function.")

def create_numpy_path(states):
    lines = states.splitlines()
    length = len(lines) - 1
    array = np.zeros((length, 2))

    for i in range(length):
        array[i][0] = float(lines[i].split(" ")[0])
        array[i][1] = float(lines[i].split(" ")[1])
    return array

def plot_path(solution_path, start, goal, dimensions, obstacleList, savePath):

    matrix = solution_path.printAsMatrix()
    path = create_numpy_path(matrix)
    x, y = path.T
    ax = plt.gca()

    ax.plot(start.getX(), start.getY(), color='red', marker='x')
    ax.plot(goal.getX(), goal.getY(), color='red', marker='x')

    ax.plot(x, y, color='green', linewidth=3)
    # ax.plot(x, y, 'go') 
    ax.axis(xmin=dimensions[0], xmax=dimensions[2], ymin=dimensions[1], ymax=dimensions[3])

    for obstacle in obstacleList:
        x, y = obstacle.exterior.xy
        plt.fill(x, y, c="blue")

    try:

        plt.savefig(savePath+'.png')
        plt.clf()
    except:
        plt.show()


def dubplan(runTime, plannerType, objectiveType, fname, radius, obstacleMap):
    # Construct the robot state space in which we're planning. We're
    # planning in [0,1]x[0,1], a subset of R^2.
    # space = ob.RealVectorStateSpace(2)

    space = ob.DubinsStateSpace(radius)
    

    # Set the bounds of space to be in [0,1].
    # space.setBounds(0.0, 1.0)
    bounds = ob.RealVectorBounds(2)

    # Set map dimensions [x_min, y_min, x_max, y_max]
    mapDimensions = [-5, -5, 20, 14]
    bounds.setLow(0, mapDimensions[0])
    bounds.setLow(1, mapDimensions[1])
    bounds.setHigh(0, mapDimensions[2])
    bounds.setHigh(1, mapDimensions[3])

    space.setBounds(bounds)
    
    # Construct a space information instance for this state space
    si = ob.SpaceInformation(space)

    obstacleList = getObstacleSet(obstacleMap) #Can be avoided

    # Set the object used to check which states in the space are valid
    # validityChecker = ValidityChecker(si)
    validityChecker = ValidityChecker(si, obstacleList)
    si.setStateValidityChecker(validityChecker)
    si.setup()

    # Set our robot's starting state 
    start = ob.State(space)
    start().setX(0)
    start().setY(4.5)
    # start().setYaw(math.pi/2)
    start().setYaw(0)


    # Set our robot's goal state 
    goal = ob.State(space)
    goal().setX(16)
    goal().setY(4.5)
    # goal().setYaw(math.pi/2)
    goal().setYaw(0)

    # '''
    # Create a problem instance
    pdef = ob.ProblemDefinition(si)

    # Set the start and goal states
    # pdef.setStartAndGoalStates(start, goal)
    pdef.setStartAndGoalStates(start(), goal())

    # Create the optimization objective specified by our command-line argument.
    # This helper function is simply a switch statement.
    pdef.setOptimizationObjective(allocateObjective(si, objectiveType))

    
    # Construct the optimal planner specified by our command line argument.
    # This helper function is simply a switch statement.
    optimizingPlanner = allocatePlanner(si, plannerType)

    # Set the problem instance for our planner to solve
    optimizingPlanner.setProblemDefinition(pdef)
    optimizingPlanner.setup()

    # attempt to solve the planning problem in the given runtime
    solved = optimizingPlanner.solve(runTime)

    if solved:
        solutionPath = pdef.getSolutionPath()
        # valid = solutionPath.checkMotion(start(), goal())
        valid = solutionPath.check()
        # print(valid)


        # Output the length of the path found
        print('{0} found solution of path length {1:.4f} with an optimization ' \
            'objective value of {2:.4f}'.format( \
            optimizingPlanner.getName(), \
            solutionPath.length(), \
            solutionPath.cost(pdef.getOptimizationObjective()).value()))

        # If a filename was specified, output the path as a matrix to
        # that file for visualization
        if fname:
            with open(fname+'.txt', 'w') as outFile:
                outFile.write(solutionPath.printAsMatrix())
        
        # pdef.simplifySolution() 
        solutionPath.interpolate(1000)
        # print(solutionPath)

        plot_path(solutionPath, start(), goal(), mapDimensions, obstacleList, fname)

        return solutionPath.length(), runTime, valid
    else:
        print("No solution found.")
        return None, None, None

    

def dubBenchmark(runTime, plannerSet, objectiveType, fname, radius, obstacleMap):

    # NEED TO DEBUG

    space = ob.DubinsStateSpace(radius)
    
    # Set the bounds of space to be in [0,1].
    # space.setBounds(0.0, 1.0)
    bounds = ob.RealVectorBounds(2)

    mapDimensions = [-2, -2, 18, 12]
    bounds.setLow(0, mapDimensions[0])
    bounds.setLow(1, mapDimensions[1])
    bounds.setHigh(0, mapDimensions[2])
    bounds.setHigh(1, mapDimensions[3])

    space.setBounds(bounds)
    

    # Construct a space information instance for this state space
    si = ob.SpaceInformation(space)

    obstacleList = getObstacleSet(obstacleMap) #Can be avoided

    # Set the object used to check which states in the space are valid
    # validityChecker = ValidityChecker(si)
    validityChecker = ValidityChecker(si, obstacleList)
    si.setStateValidityChecker(validityChecker)
    si.setup()

    # Set our robot's starting state (x,y,yaw).
    start = ob.State(space)
    start().setX(0)
    start().setY(4.5)
    start().setYaw(0)

    # Set our robot's goal state (x,y,yaw).
    goal = ob.State(space)
    goal().setX(16)
    goal().setY(4.5)
    goal().setYaw(0)

    ss = og.SimpleSetup(si)

    ss.setStartAndGoalStates(start(), goal())
    ss.setOptimizationObjective(allocateObjective(si, objectiveType))

    b = ot.Benchmark(ss, "experiment")
    for planner in plannerSet:
        b.addPlanner(allocatePlanner(si, planner))

    req = b.Request()
    req.maxTime = 5
    req.maxMem = 100.0
    req.runCount = 50
    req.displayProgress = True
    b.benchmark(req)
    b.saveResultsToFile()

    # print(space.distance(start(), goal()))
    # path = space.DubinsPath()
    # s = ob.State(space)
    # '''


  
if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Optimal motion planning demo program.')

    # Add a filename argument
    parser.add_argument('-t', '--runtime', type=float, default=60, help=\
        '(Optional) Specify the runtime in seconds. Defaults to 1 and must be greater than 0.')
    parser.add_argument('-p', '--planner', default='RRTstar', \
        choices=['BFMTstar', 'BITstar', 'FMTstar', 'InformedRRTstar', 'PRMstar', 'RRT', 'RRTstar', \
        'SORRTstar'], \
        help='(Optional) Specify the optimal planner to use, defaults to RRTstar if not given.')
    parser.add_argument('-o', '--objective', default='PathLength', \
        choices=['PathClearance', 'PathLength', 'ThresholdPathLength', \
        'WeightedLengthAndClearanceCombo'], \
        help='(Optional) Specify the optimization objective, defaults to PathLength if not given.')
    parser.add_argument('-f', '--file', default=None, \
        help='(Optional) Specify an output path for the found solution path.')
    parser.add_argument('-r', '--radius', type=float, default=1, \
        help='Specify the radius of curvature. Defaults to 1 and must be greater than 0.')
    parser.add_argument('-m', '--map', type=str, default='./test_folder/new.pkl', \
        help='Specify the path of the .pkl file for the Map')
    parser.add_argument('-i', '--info', type=int, default=0, choices=[0, 1, 2], \
        help='(Optional) Set the OMPL log level. 0 for WARN, 1 for INFO, 2 for DEBUG.' \
        ' Defaults to WARN.')
    

    # Parse the arguments
    args = parser.parse_args()

    # Check that time is positive
    if args.runtime <= 0:
        raise argparse.ArgumentTypeError(
            "argument -t/--runtime: invalid choice: %r (choose a positive number greater than 0)" \
            % (args.runtime,))

    # Set the log level
    if args.info == 0:
        ou.setLogLevel(ou.LOG_WARN)
    elif args.info == 1:
        ou.setLogLevel(ou.LOG_INFO)
    elif args.info == 2:
        ou.setLogLevel(ou.LOG_DEBUG)
    else:
        ou.OMPL_ERROR("Invalid log-level integer.")

    # Solve the planning problem
    # dubplan(args.runtime, args.planner, args.objective, args.file, args.radius, './test_folder/obs_10/obstacleList_Map_test_o10.pkl')

    # dubBenchmark(args.runtime, plannerSet, args.objective, args.file, args.radius, args.map)

    plannerSet = ['RRT', 'RRTstar', 'PRMstar', 'BFMTstar', 'BITstar', 'FMTstar', 'InformedRRTstar', 'PRMstar', 'SORRTstar']

    radiusSet = [1, 2, 3]
    # radiusSet = [2]

    folders = ['./instance_set1']

    for planner in plannerSet:

        for path in folders:

            with open(path+'/tolerances.yaml') as tolerances_yaml:
                tolerances = yaml.load(tolerances_yaml, Loader=yaml.FullLoader)

            # result_fields = ['name', 'path', 'Obstacles','turning_radius', 'continuity_tolerance', 'angle_tolerance', 'length_BFMTstar', 'BFMTstar_time',  'length_BITstar', 'BITstar_time', 'length_FMTstar', 'FMTstar_time', 'length_InformedRRTstar', 'InformedRRTstar_time', 'length_PRMstar', 'PRMstar_time', 'length_RRT', 'RRT_time', 'length_RRTstar', 'RRTstar_time', 'length_SORRTstar', 'SORRTstar_time' ] 
            result_fields = ['name', 'path', 'Obstacles','turning_radius', 'continuity_tolerance', 'angle_tolerance', 'length_'+planner, 'time_'+planner,  'feasible_path' ] 
            result_filename = path+'/ompl_results_'+planner+'_'+time.strftime("%Y%m%d-%H%M%S")+'.csv'
                
            # writing to csv file 
            with open(result_filename, 'w') as csvfile: 
                # creating a csv writer object 
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(result_fields) 
                    
                # writing the data rows 
                instance_results = []
                for (instance_path, dirs, files) in os.walk(path):
                    for f in files:
                        if f.endswith('.pkl') and f.split('_')[0] == 'obstacleList':
                            obstacle_count = f.split("_")[-1]
                            obstacle_count = obstacle_count[1:3]
                            
                            mapPath = os.path.join(instance_path, f)
                            omplPath = instance_path + '/ompl_plots'

                            if not os.path.exists(omplPath):
                                os.mkdir(omplPath)

                        
                            for radius in radiusSet:

                                omplOutFile = omplPath + '/' + planner + '_r' + str(radius)
                                length_planner, time_planner, pathFeasible = dubplan(args.runtime, planner, args.objective, omplOutFile, radius, mapPath)

                                instance_data = [f, mapPath, obstacle_count, radius, tolerances['continuity'], tolerances['angular'], length_planner, time_planner, pathFeasible]
                                csvwriter.writerow(instance_data)
    