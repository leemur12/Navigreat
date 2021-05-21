import numpy as np
import collections
import random

import utils
import base

WALL=-1
PATH=0
WIN=1


class Maze(base.MazeBase):
    """This class contains the relevant algorithms for creating and solving."""
    def __init__(self):
        """Constructor."""
        super(Maze, self).__init__()

        self._dir_one = [
            lambda x, y: (x + 1, y),
            lambda x, y: (x - 1, y),
            lambda x, y: (x, y - 1),
            lambda x, y: (x, y + 1)
        ]
        self._dir_two = [
            lambda x, y: (x + 2, y),
            lambda x, y: (x - 2, y),
            lambda x, y: (x, y - 2),
            lambda x, y: (x, y + 2)
        ]
        self._range = list(range(4))

    def create(self, row_count, col_count, algorithm):
        """Creates a maze for a given row and column count."""
        if (row_count or col_count) <= 0:
            raise utils.MazeError("Row or column count cannot be smaller than zero.")

        self.maze = np.ones((row_count, col_count), dtype=np.uint8)*-1

        if algorithm == Maze.Create.BACKTRACKING or algorithm==0:
            return self._recursive_backtracking()
        # if algorithm == Maze.Create.HUNT:
        #     return self._hunt_and_kill()
        if algorithm == Maze.Create.ELLER or algorithm==1:
            return self._eller()
        if algorithm == Maze.Create.SIDEWINDER or algorithm==2:
            return self._sidewinder()
        if algorithm == Maze.Create.PRIM or algorithm==3:
            return self._prim()
        if algorithm == Maze.Create.KRUSKAL or algorithm==4:
            return self._kruskal()
        if algorithm == Maze.Create.RANDOM or algorithm==5:
            return self._randomGenerate()

        raise utils.MazeError(
            "Wrong algorithm <{}>.\n"
            "Use \"Maze.Create.<algorithm>\" to choose an algorithm.".format(algorithm)
        )

    @property
    def _random(self):
        """Returns a random range to iterate over."""
        random.shuffle(self._range)
        return self._range

    def _out_of_bounds(self, x, y):
        """Checks if indices are out of bounds."""
        return x < 0 or y < 0 or x >= self.row_count_with_walls or y >= self.col_count_with_walls


    def _create_walk(self, x, y):
        """Randomly walks from one pointer within the maze to another one."""
        for idx in self._random:  # Check adjacent cells randomly
            tx, ty = self._dir_two[idx](x, y)
            if not self._out_of_bounds(tx, ty) and self.maze[tx, ty] == WALL:  # Check if unvisited
                self.maze[tx, ty] = self.maze[self._dir_one[idx](x, y)] = PATH  # Mark as visited
                return tx, ty  # Return new cell

        return None, None  # Return stop values

    def _create_backtrack(self, stack):
        """Backtracks the stack until walking is possible again."""
        while stack:
            x, y = stack.pop()
            for direction in self._dir_two:  # Check adjacent cells
                tx, ty = direction(x, y)
                if not self._out_of_bounds(tx, ty) and self.maze[tx, ty] == -1:  # Check if unvisited
                    return x, y  # Return cell with unvisited neighbour

        return None, None  # Return stop values if stack is empty
    def _randomGenerate(self):
        self.maze[1:-1, 1:-1]= PATH

        row_count_with_walls, col_count_with_walls = self.row_count_with_walls, self.col_count_with_walls
        penalties= int(row_count_with_walls*col_count_with_walls/6)

        while penalties != 0:
            i = random.randint(0,row_count_with_walls-1)
            j = random.randint(0,col_count_with_walls-1)

        #if the spot is empty and it isn't the first or last positions add the penalty
            if self.maze[i,j] == 0 and [i,j] != [1,1] and [i,j] != [row_count_with_walls-1,col_count_with_walls-1]:
            #setting reward value
                self.maze[i,j] = WALL
                penalties-=1

    def _recursive_backtracking(self):
        """Creates a maze using the recursive backtracking algorithm."""
        stack = collections.deque()  # List of visited cells [(x, y), ...]

        x = 2 * random.randint(0, self.row_count - 1)+1
        y = 2 * random.randint(0, self.col_count - 1)+1
        self.maze[x, y] = PATH  # Mark as visited

        while x and y:
            while x and y:
                stack.append((x, y))
                x, y = self._create_walk(x, y)
            x, y = self._create_backtrack(stack)


    def _hunt(self, hunt_list):
        """Scans the maze for new position."""
        while hunt_list:
            for x in hunt_list:
                finished = True
                for y in range(1, self.col_count_with_walls - 1, 2):
                    if self.maze[x, y] == WALL:  # Check if unvisited
                        finished = False
                        for direction in self._dir_two:  # Check adjacent cells
                            tx, ty = direction(x, y)
                            if not self._out_of_bounds(tx, ty) and self.maze[tx, ty] == PATH:  # Check if visited
                                return x, y  # Return visited neighbour of unvisited cell
                if finished:
                    hunt_list.remove(x)  # Remove finished row
                    break  # Restart loop

        return None, None  # Return stop values if all rows are finished

    def _hunt_and_kill(self):
        """Creates a maze using the hunt and kill algorithm."""
        hunt_list = list(range(1, self.row_count_with_walls - 1, 2))  # List of unfinished rows [x, ...]

        x = 2 * random.randint(0, self.row_count - 1) + 1
        y = 2 * random.randint(0, self.col_count - 1) + 1
        self.maze[x, y] = PATH  # Mark as visited

        while hunt_list:
            print(hunt_list)
            print(self.maze)
            while x and y:
                x, y = self._create_walk(x, y)
            x, y = self._hunt(hunt_list)

    def _eller(self):
        """Creates a maze using Eller's algorithm."""
        row_stack = [0] * self.col_count  # List of set indices [set index, ...]
        set_list = []  # List of set indices with positions [(set index, position), ...]
        set_index = 1

        for x in range(1, self.row_count_with_walls - 1, 2):
            connect_list = collections.deque()  # List of connections between cells [True, ...]

            # Create row stack
            if row_stack[0] == 0:  # Define first cell in row
                row_stack[0] = set_index
                set_index += 1

            for y in range(1, self.col_count):  # Define other cells in row
                if random.getrandbits(1):  # Connect cell with previous cell
                    if row_stack[y] != 0:  # Cell has a set
                        old_index = row_stack[y]
                        new_index = row_stack[y - 1]
                        if old_index != new_index:  # Combine both sets
                            row_stack = [new_index if y == old_index else y for y in row_stack]  # Replace old indices
                            connect_list.append(True)
                        else:
                            connect_list.append(False)
                    else:  # Cell has no set
                        row_stack[y] = row_stack[y - 1]
                        connect_list.append(True)
                else:  # Do not connect cell with previous cell
                    if row_stack[y] == 0:
                        row_stack[y] = set_index
                        set_index += 1
                    connect_list.append(False)

            # Create set list and fill cells
            for y in range(self.col_count):
                maze_col = 2 * y + 1
                set_list.append((row_stack[y], maze_col))

                self.maze[x, maze_col] = 0  # Mark as visited
                if y < self.col_count - 1:
                    if connect_list.popleft():
                        self.maze[x, maze_col + 1] = PATH  # Mark as visited

            if x == self.row_count_with_walls - 2:  # Connect all different sets in last row
                for y in range(1, self.col_count):
                    new_index = row_stack[y - 1]
                    old_index = row_stack[y]
                    if new_index != old_index:
                        row_stack = [new_index if y == old_index else y for y in row_stack]  # Replace old indices
                        self.maze[x, 2 * y] = PATH  # Mark as visited
                break  # End loop with last row

            # Reset row stack
            row_stack = [0] * self.col_count

            # Create vertical links
            set_list.sort(reverse=True)
            while set_list:
                sub_set_list = collections.deque()  # List of set indices with positions for one set index [(set index, position), ...]
                sub_set_index = set_list[-1][0]
                while set_list and set_list[-1][0] == sub_set_index:  # Create sub list for one set index
                    sub_set_list.append(set_list.pop())
                linked = False
                while not linked:  # Create at least one link for each set index
                    for sub_set_item in sub_set_list:
                        if random.getrandbits(1):  # Create link
                            linked = True
                            link_set, link_position = sub_set_item

                            row_stack[link_position // 2] = link_set  # Assign links to new row stack
                            self.maze[x + 1, link_position] = PATH  # Mark link as visited

    def _sidewinder(self):
        """Creates a maze using the sidewinder algorithm."""
        # Create first row
        for y in range(1, self.col_count_with_walls - 1):
            self.maze[1, y] = PATH

        # Create other rows
        for x in range(3, self.row_count_with_walls, 2):
            row_stack = []  # List of cells without vertical link [y, ...]
            for y in range(1, self.col_count_with_walls - 2, 2):
                self.maze[x, y] = PATH # Mark as visited
                row_stack.append(y)

                if random.getrandbits(1):  # Create vertical link
                    idx = random.randint(0, len(row_stack) - 1)
                    self.maze[x - 1, row_stack[idx]] = PATH  # Mark as visited
                    row_stack = []  # Reset row stack
                else:  # Create horizontal link
                    self.maze[x, y + 1] = PATH # Mark as visited

            # Create vertical link if last cell
            y = self.col_count_with_walls - 2
            self.maze[x, y] = PATH  # Mark as visited
            row_stack.append(y)
            idx = random.randint(0, len(row_stack) - 1)
            self.maze[x - 1, row_stack[idx]] = PATH  # Mark as visited

    def _prim(self):
        """Creates a maze using Prim's algorithm."""
        frontier = []  # List of unvisited cells [(x, y),...]

        # Start with random cell
        x = 2 * random.randint(0, self.row_count - 1) + 1
        y = 2 * random.randint(0, self.col_count - 1) + 1
        self.maze[x, y] = PATH  # Mark as visited

        # Add cells to frontier for random cell
        for direction in self._dir_two:
            tx, ty = direction(x, y)
            if not self._out_of_bounds(tx, ty):
                frontier.append((tx, ty))
                self.maze[tx, ty] = 1  # Mark as part of frontier

        # Add and connect cells until frontier is empty
        while frontier:
            x, y = frontier.pop(random.randint(0, len(frontier) - 1))

            # Connect cells
            for idx in self._random:
                tx, ty = self._dir_two[idx](x, y)
                if not self._out_of_bounds(tx, ty) and self.maze[tx, ty] == PATH:  # Check if visited
                    self.maze[x, y] = self.maze[self._dir_one[idx](x, y)] = PATH  # Connect cells
                    break

            # Add cells to frontier
            for direction in self._dir_two:
                tx, ty = direction(x, y)
                if not self._out_of_bounds(tx, ty) and self.maze[tx, ty] == WALL:  # Check if unvisited
                    frontier.append((tx, ty))
                    self.maze[tx, ty] = 1  # Mark as part of frontier

    def _kruskal(self):
        """Creates a maze using Kruskal's algorithm."""
        xy_to_set = np.zeros((self.row_count_with_walls, self.col_count_with_walls), dtype=np.uint32)
        set_to_xy = []  # List of sets in order, set 0 at index 0 [[(x, y),...], ...]
        edges = collections.deque()  # List of possible edges [(x, y, direction), ...]
        set_index = 0

        for x in range(1, self.row_count_with_walls - 1, 2):
            for y in range(1, self.col_count_with_walls - 1, 2):
                # Assign sets
                xy_to_set[x, y] = set_index
                set_to_xy.append([(x, y)])
                set_index += 1

                # Create edges
                if not self._out_of_bounds(x + 2, y):
                    edges.append((x + 1, y, "v"))  # Vertical edge
                if not self._out_of_bounds(x, y + 2):
                    edges.append((x, y + 1, "h"))  # Horizontal edge

        random.shuffle(edges)  # Shuffle to pop random edges
        while edges:
            x, y, direction = edges.pop()

            x1, x2 = (x - 1, x + 1) if direction == "v" else (x, x)
            y1, y2 = (y - 1, y + 1) if direction == "h" else (y, y)

            if xy_to_set[x1, y1] != xy_to_set[x2, y2]:  # Check if cells are in different sets
                self.maze[x, y] = self.maze[x1, y1] = self.maze[x2, y2] = PATH # Mark as visited

                new_set = xy_to_set[x1, y1]
                old_set = xy_to_set[x2, y2]

                # Extend new set with old set
                set_to_xy[new_set].extend(set_to_xy[old_set])

                # Correct sets in xy sets
                for pos in set_to_xy[old_set]:
                    xy_to_set[pos] = new_set
    def evenFix(self):
        self.maze[-1, :]=-1
        self.maze[:,-1]=-1

    def removeWalls(self, numwalls):
        n= numwalls
        while n>0:
            r=random.randint(1, self.maze.shape[0]-1)
            c= random.randint(1, self.maze.shape[0] - 1)

            if self.maze[r,c]==WALL:
                self.maze[r,c]=PATH
                n-=1

    def addExits(self, exits):

        numExits= exits

        if self.maze.shape[0]%2==0:
            self.evenFix()

        while numExits>0:

            pos= random.randint(0, self.maze.shape[0]-1)

            isColumns= random.getrandbits(1)   #If true exits are on the columns
            isEnd= random.getrandbits(1) #If true, it is on the last row/col

            if isColumns:
                if isEnd:
                    if self.maze[pos, -2]==PATH:
                        self.maze[pos, -1]= WIN
                        numExits-=1
                else:
                    if self.maze[pos,1]==PATH:
                        self.maze[pos, 0]=WIN
                        numExits-=1

            else:
                if isEnd:
                    if self.maze[-2, pos]==PATH:
                        self.maze[-1, pos]=WIN
                        numExits-=1
                else:
                    if self.maze[1, pos]==PATH:
                        self.maze[0,pos]= WIN
                        numExits-=1



