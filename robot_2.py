# ============================================================
# Project  : Intelligent Urban Delivery Robot Simulation
# Module   : 1
# Description:
#   This program simulates an intelligent delivery robot
#   navigating a 15x15 grid-based urban environment.
#   The grid contains roads, buildings (obstacles), and
#   traffic zones with varying traversal costs.
#   The robot starts from a base station and completes
#   5 deliveries using one of five search algorithms:
#   BFS, DFS, UCS, Greedy Best First, and A*.
#   A tkinter GUI visualizes the robot moving in real time.
#   Performance metrics (cost, nodes explored, time) are
#   displayed after each delivery for comparison.
# ============================================================

import random
import time
import heapq
import math
import tkinter as tk
from tkinter import messagebox

# Grid size is fixed at 15x15 as per project instructions
GRID_SIZE = 15

# Total number of deliveries the robot must complete
NUM_DELIVERIES = 5

# Integer constants representing each type of grid cell
ROAD = 0
BUILDING = 1
TRAFFIC = 2
DELIVERY = 3
BASE = 4
ROBOT = 5
PATH = 6

# Color mapping for each cell type used in the GUI canvas
COLORS = {
    ROAD: "#d4e6b5",
    BUILDING: "#5c4033",
    TRAFFIC: "#f0a500",
    DELIVERY: "#e74c3c",
    BASE: "#2980b9",
    ROBOT: "#8e44ad",
    PATH: "#1abc9c"
}

def create_grid():
    # Creates and returns a 15x15 grid populated with ROAD cells by default.
    # Then randomly places BUILDING cells (20% chance per cell) as obstacles.
    # Then randomly places TRAFFIC cells (10% chance) on remaining ROAD cells.
    grid = [[ROAD for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if random.random() < 0.20:
                grid[row][col] = BUILDING
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if grid[row][col] == ROAD and random.random() < 0.10:
                grid[row][col] = TRAFFIC
    return grid

def assign_costs(grid):
    # Assigns a traversal cost to every cell in the grid based on cell type.
    # ROAD, DELIVERY, BASE cells get a random cost between 1 and 5.
    # TRAFFIC cells get a higher random cost between 10 and 20.
    # BUILDING cells get infinity cost because they cannot be traversed.
    cost_grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if grid[row][col] in (ROAD, DELIVERY, BASE):
                cost_grid[row][col] = random.randint(1, 5)
            elif grid[row][col] == TRAFFIC:
                cost_grid[row][col] = random.randint(10, 20)
            elif grid[row][col] == BUILDING:
                cost_grid[row][col] = float('inf')
    return cost_grid

def get_neighbors(row, col, grid):
    # Returns a list of valid neighboring cells from position (row, col).
    # Only checks up, down, left, right directions (no diagonals).
    # A neighbor is valid if it is inside the grid and is not a BUILDING.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors = []
    for delta_row, delta_col in directions:
        next_row = row + delta_row
        next_col = col + delta_col
        if 0 <= next_row < GRID_SIZE and 0 <= next_col < GRID_SIZE:
            if grid[next_row][next_col] != BUILDING:
                neighbors.append((next_row, next_col))
    return neighbors

def bfs(grid, cost_grid, start, goal):
    # Breadth First Search explores all neighbors level by level.
    # It finds the path with the fewest number of steps (not lowest cost).
    # Uses a queue (FIFO) and a visited set to avoid revisiting cells.
    # Returns the path, total traversal cost, and number of nodes explored.
    from collections import deque
    queue = deque()
    queue.append((start, [start]))
    visited = set()
    visited.add(start)
    nodes_explored = 0
    while queue:
        (row, col), path = queue.popleft()
        nodes_explored += 1
        if (row, col) == goal:
            total_cost = sum(cost_grid[r][c] for r, c in path)
            return path, total_cost, nodes_explored
        for next_row, next_col in get_neighbors(row, col, grid):
            if (next_row, next_col) not in visited:
                visited.add((next_row, next_col))
                queue.append(((next_row, next_col), path + [(next_row, next_col)]))
    return None, float('inf'), nodes_explored

def dfs(grid, cost_grid, start, goal):
    # Depth First Search explores as deep as possible before backtracking.
    # Uses a stack (LIFO) so it dives deep into one path first.
    # Not guaranteed to find the shortest or cheapest path.
    # Returns the path, total traversal cost, and number of nodes explored.
    stack = [(start, [start])]
    visited = set()
    nodes_explored = 0
    while stack:
        (row, col), path = stack.pop()
        if (row, col) in visited:
            continue
        visited.add((row, col))
        nodes_explored += 1
        if (row, col) == goal:
            total_cost = sum(cost_grid[r][c] for r, c in path)
            return path, total_cost, nodes_explored
        for next_row, next_col in get_neighbors(row, col, grid):
            if (next_row, next_col) not in visited:
                stack.append(((next_row, next_col), path + [(next_row, next_col)]))
    return None, float('inf'), nodes_explored

def ucs(grid, cost_grid, start, goal):
    # Uniform Cost Search always expands the node with lowest cumulative cost.
    # Uses a min-heap (priority queue) ordered by total path cost so far.
    # Guaranteed to find the optimal (lowest cost) path.
    # Returns the path, total cost, and number of nodes explored.
    heap = [(0, start, [start])]
    visited = {}
    nodes_explored = 0
    while heap:
        current_cost, (row, col), path = heapq.heappop(heap)
        if (row, col) in visited:
            continue
        visited[(row, col)] = current_cost
        nodes_explored += 1
        if (row, col) == goal:
            return path, current_cost, nodes_explored
        for next_row, next_col in get_neighbors(row, col, grid):
            if (next_row, next_col) not in visited:
                new_cost = current_cost + cost_grid[next_row][next_col]
                heapq.heappush(heap, (new_cost, (next_row, next_col), path + [(next_row, next_col)]))
    return None, float('inf'), nodes_explored

def euclidean_distance(point_a, point_b):
    # Calculates straight-line (Euclidean) distance between two grid cells.
    # Used as the heuristic function for Greedy Best First Search.
    return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)

def manhattan_distance(point_a, point_b):
    # Calculates Manhattan distance between two grid cells.
    # Counts steps horizontally + vertically, suitable for grid movement.
    # Used as the heuristic function for A* Search.
    return abs(point_a[0] - point_b[0]) + abs(point_a[1] - point_b[1])

def greedy(grid, cost_grid, start, goal):
    # Greedy Best First Search always expands the node closest to the goal.
    # Uses Euclidean distance as heuristic to guide the search direction.
    # Fast but not guaranteed to find the optimal path.
    # Returns the path, total traversal cost, and number of nodes explored.
    heap = [(euclidean_distance(start, goal), start, [start])]
    visited = set()
    nodes_explored = 0
    while heap:
        _, (row, col), path = heapq.heappop(heap)
        if (row, col) in visited:
            continue
        visited.add((row, col))
        nodes_explored += 1
        if (row, col) == goal:
            total_cost = sum(cost_grid[r][c] for r, c in path)
            return path, total_cost, nodes_explored
        for next_row, next_col in get_neighbors(row, col, grid):
            if (next_row, next_col) not in visited:
                heuristic_value = euclidean_distance((next_row, next_col), goal)
                heapq.heappush(heap, (heuristic_value, (next_row, next_col), path + [(next_row, next_col)]))
    return None, float('inf'), nodes_explored

def astar(grid, cost_grid, start, goal):
    # A* Search combines actual cost (g) and heuristic estimate (h) as f = g + h.
    # Uses Manhattan distance as heuristic for grid-based movement.
    # Optimal and efficient — balances cost and direction toward goal.
    # Returns the path, total cost (g), and number of nodes explored.
    initial_heuristic = manhattan_distance(start, goal)
    heap = [(initial_heuristic, 0, start, [start])]
    visited = {}
    nodes_explored = 0
    while heap:
        f_score, g_score, (row, col), path = heapq.heappop(heap)
        if (row, col) in visited:
            continue
        visited[(row, col)] = g_score
        nodes_explored += 1
        if (row, col) == goal:
            return path, g_score, nodes_explored
        for next_row, next_col in get_neighbors(row, col, grid):
            if (next_row, next_col) not in visited:
                new_g = g_score + cost_grid[next_row][next_col]
                new_h = manhattan_distance((next_row, next_col), goal)
                heapq.heappush(heap, (new_g + new_h, new_g, (next_row, next_col), path + [(next_row, next_col)]))
    return None, float('inf'), nodes_explored

def run_algorithm(algorithm_name, grid, cost_grid, start, goal):
    # Runs the selected search algorithm and measures execution time in milliseconds.
    # Calls the appropriate algorithm function based on algorithm_name string.
    # Returns path found, total cost, nodes explored, and time taken.
    time_start = time.time()
    if algorithm_name == "BFS":
        path, cost, nodes = bfs(grid, cost_grid, start, goal)
    elif algorithm_name == "DFS":
        path, cost, nodes = dfs(grid, cost_grid, start, goal)
    elif algorithm_name == "UCS":
        path, cost, nodes = ucs(grid, cost_grid, start, goal)
    elif algorithm_name == "Greedy":
        path, cost, nodes = greedy(grid, cost_grid, start, goal)
    elif algorithm_name == "A*":
        path, cost, nodes = astar(grid, cost_grid, start, goal)
    time_end = time.time()
    elapsed_ms = round((time_end - time_start) * 1000, 4)
    return path, cost, nodes, elapsed_ms

def find_free_cell(grid):
    # Randomly picks and returns a cell that is not a BUILDING.
    # Keeps retrying until a valid free cell is found.
    while True:
        random_row = random.randint(0, GRID_SIZE - 1)
        random_col = random.randint(0, GRID_SIZE - 1)
        if grid[random_row][random_col] != BUILDING:
            return (random_row, random_col)

class RobotApp:
    # Main application class that builds the GUI and manages simulation state.
    # Handles grid generation, algorithm selection, robot animation, and result logging.

    def __init__(self, root):
        # Initializes the main window, creates the first random map, and builds the UI.
        self.root = root
        self.root.title("Intelligent Urban Delivery Robot Simulation")
        self.root.configure(bg="#1a1a2e")
        self.cell_size = 42
        self.original_grid = None
        self.original_costs = None
        self.original_base = None
        self.original_deliveries = None
        self.selected_algo = tk.StringVar(value="A*")
        self.results = []
        self.current_path = []
        self.path_index = 0
        self.animating = False
        self.build_ui()
        self.new_map()

    def new_map(self):
        # Generates a completely new random grid with buildings, traffic zones,
        # a random base station, and 5 random delivery destinations.
        # Saves the original map so it can be reused for retry with a different algorithm.
        self.original_grid = create_grid()
        self.original_costs = assign_costs(self.original_grid)
        self.original_base = find_free_cell(self.original_grid)
        self.original_grid[self.original_base[0]][self.original_base[1]] = BASE
        delivery_list = []
        used_cells = {self.original_base}
        while len(delivery_list) < NUM_DELIVERIES:
            candidate_cell = find_free_cell(self.original_grid)
            if candidate_cell not in used_cells:
                delivery_list.append(candidate_cell)
                used_cells.add(candidate_cell)
                self.original_grid[candidate_cell[0]][candidate_cell[1]] = DELIVERY
        self.original_deliveries = delivery_list
        self.reset_state()

    def reset_state(self):
        # Resets the robot and delivery progress back to the beginning
        # while keeping the same map layout and costs.
        # Clears the results log and re-enables the Start button.
        import copy
        self.grid = copy.deepcopy(self.original_grid)
        self.cost_grid = self.original_costs
        self.base = self.original_base
        self.deliveries = list(self.original_deliveries)
        self.robot_pos = self.base
        self.current_delivery = 0
        self.results = []
        self.current_path = []
        self.path_index = 0
        self.animating = False
        for delivery_cell in self.deliveries:
            self.grid[delivery_cell[0]][delivery_cell[1]] = DELIVERY
        self.grid[self.base[0]][self.base[1]] = BASE
        self.btn_start.config(state=tk.NORMAL, text="▶  START DELIVERY", bg="#00d4ff")
        self.btn_reset.config(state=tk.DISABLED)
        self.btn_newmap.config(state=tk.NORMAL)
        self.status.config(text="Pick algorithm\nthen press START")
        self.result_box.config(state=tk.NORMAL)
        self.result_box.delete("1.0", tk.END)
        self.result_box.config(state=tk.DISABLED)
        self.draw_grid()

    def build_ui(self):
        # Builds the entire GUI layout with title, grid canvas on the left,
        # and algorithm selector, buttons, results log, and legend on the right.
        tk.Label(self.root, text=" @:-) Urban Delivery Robot",
                 font=("Courier New", 16, "bold"), bg="#1a1a2e", fg="#00d4ff").pack(pady=(10, 4))
        main_frame = tk.Frame(self.root, bg="#1a1a2e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        # Left panel holds the grid canvas
        left_panel = tk.Frame(main_frame, bg="#05596c", padx=2, pady=2)
        left_panel.pack(side=tk.LEFT, padx=(0, 12))
        self.canvas = tk.Canvas(left_panel,
                                width=GRID_SIZE * self.cell_size,
                                height=GRID_SIZE * self.cell_size,
                                bg="#0f0f1a", highlightthickness=0)
        self.canvas.pack()
        # Right panel holds all controls and results
        right_panel = tk.Frame(main_frame, bg="#1a1a2e")
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(right_panel, text="Select Algorithm:", bg="#1a1a2e", fg="#aaaacc",
                 font=("Courier New", 11, "bold")).pack(anchor=tk.W, pady=(0, 4))
        for algo_name in ["BFS", "DFS", "UCS", "Greedy", "A*"]:
            tk.Radiobutton(right_panel, text=algo_name, variable=self.selected_algo, value=algo_name,
                           bg="#1a1a2e", fg="#00d4ff", selectcolor="#0f3460",
                           activebackground="#1a1a2e",
                           font=("Courier New", 11, "bold")).pack(anchor=tk.W, pady=2)
        self.btn_start = tk.Button(right_panel, text="▶  START DELIVERY",
                                   command=self.start_delivery,
                                   bg="#00d4ff", fg="#0f0f1a",
                                   font=("Courier New", 11, "bold"),
                                   relief=tk.FLAT, padx=10, pady=7, cursor="hand2")
        self.btn_start.pack(fill=tk.X, pady=(14, 4))
        self.btn_reset = tk.Button(right_panel, text="🔄  RETRY SAME MAP",
                                   command=self.reset_state,
                                   bg="#f0a500", fg="#0f0f1a",
                                   font=("Courier New", 11, "bold"),
                                   relief=tk.FLAT, padx=10, pady=7, cursor="hand2",
                                   state=tk.DISABLED)
        self.btn_reset.pack(fill=tk.X, pady=4)
        self.btn_newmap = tk.Button(right_panel, text="🗺  NEW RANDOM MAP",
                                    command=self.new_map,
                                    bg="#2ecc71", fg="#0f0f1a",
                                    font=("Courier New", 11, "bold"),
                                    relief=tk.FLAT, padx=10, pady=7, cursor="hand2")
        self.btn_newmap.pack(fill=tk.X, pady=4)
        self.status = tk.Label(right_panel, text="Pick algorithm\nthen press START",
                               font=("Courier New", 9), bg="#1a1a2e", fg="#f0a500",
                               justify=tk.LEFT, wraplength=200)
        self.status.pack(anchor=tk.W, pady=(6, 6))
        tk.Label(right_panel, text="Results Log:", bg="#1a1a2e", fg="#aaaacc",
                 font=("Courier New", 10, "bold")).pack(anchor=tk.W)
        self.result_box = tk.Text(right_panel, height=14, width=30,
                                  bg="#0f0f1a", fg="#00ff99",
                                  font=("Courier New", 8),
                                  relief=tk.FLAT, state=tk.DISABLED)
        self.result_box.pack(fill=tk.BOTH, expand=True, pady=(4, 6))
        tk.Label(right_panel, text="Legend:", bg="#1a1a2e", fg="#aaaacc",
                 font=("Courier New", 9, "bold")).pack(anchor=tk.W)
        legend_items = [
            ("Road (cost 1-5)", COLORS[ROAD]),
            ("Building", COLORS[BUILDING]),
            ("Traffic (cost 10-20)", COLORS[TRAFFIC]),
            ("Delivery Point", COLORS[DELIVERY]),
            ("Base Station", COLORS[BASE]),
            ("Robot", COLORS[ROBOT]),
            ("Path", COLORS[PATH])
        ]
        for legend_label, legend_color in legend_items:
            legend_row = tk.Frame(right_panel, bg="#1a1a2e")
            legend_row.pack(anchor=tk.W)
            tk.Frame(legend_row, bg=legend_color, width=12, height=12).pack(side=tk.LEFT, padx=(0, 5))
            tk.Label(legend_row, text=legend_label, bg="#1a1a2e", fg="#cccccc",
                     font=("Courier New", 8)).pack(side=tk.LEFT)

    def draw_grid(self, path=None, robot=None):
        # Clears and redraws the entire grid canvas.
        # Colors each cell based on its type (road, building, traffic, etc).
        # Highlights path cells in teal and draws the robot emoji at current position.
        # Also draws delivery numbers and cost values on each cell.
        self.canvas.delete("all")
        path_set = set(path) if path else set()
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x1 = col * self.cell_size
                y1 = row * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                cell_type = self.grid[row][col]
                if robot and (row, col) == robot:
                    fill_color = COLORS[ROBOT]
                elif (row, col) in path_set and cell_type not in (BASE, DELIVERY):
                    fill_color = COLORS[PATH]
                else:
                    fill_color = COLORS.get(cell_type, COLORS[ROAD])
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="#1a1a2e", width=1)
                if cell_type == DELIVERY and (row, col) in self.deliveries:
                    delivery_number = self.deliveries.index((row, col)) + 1
                    self.canvas.create_text(x1 + self.cell_size // 2, y1 + self.cell_size // 2,
                                            text=str(delivery_number), fill="white",
                                            font=("Courier New", 9, "bold"))
                if cell_type != BUILDING:
                    cell_cost = self.cost_grid[row][col]
                    if cell_cost != float('inf'):
                        self.canvas.create_text(x1 + self.cell_size // 2, y2 - 6,
                                                text=str(cell_cost), fill="#333355",
                                                font=("Courier New", 6))
        if robot:
            robot_x = robot[1] * self.cell_size + self.cell_size // 2
            robot_y = robot[0] * self.cell_size + self.cell_size // 2
            self.canvas.create_text(robot_x, robot_y, text="🤖", font=("Arial", 14))

    def log_result(self, text):
        # Appends a single line of text to the results log text box.
        self.result_box.config(state=tk.NORMAL)
        self.result_box.insert(tk.END, text + "\n")
        self.result_box.see(tk.END)
        self.result_box.config(state=tk.DISABLED)

    def start_delivery(self):
        # Called when the Start button is pressed.
        # Disables buttons during animation and begins the delivery process.
        if self.animating:
            return
        if self.current_delivery >= NUM_DELIVERIES:
            return
        self.btn_start.config(state=tk.DISABLED)
        self.btn_reset.config(state=tk.DISABLED)
        self.btn_newmap.config(state=tk.DISABLED)
        self.deliver_next()

    def deliver_next(self):
        # Handles one delivery at a time in sequence.
        # Runs the selected algorithm, logs results, and starts animation.
        # When all 5 deliveries are done enables reset and new map buttons.
        if self.current_delivery >= NUM_DELIVERIES:
            self.status.config(text=" All 5 deliveries\ncompleted!")
            self.btn_start.config(state=tk.DISABLED, text=" DONE", bg="#555555")
            self.btn_reset.config(state=tk.NORMAL)
            self.btn_newmap.config(state=tk.NORMAL)
            self.show_summary()
            return
        goal_cell = self.deliveries[self.current_delivery]
        selected_algorithm = self.selected_algo.get()
        self.status.config(text=f"Delivery {self.current_delivery + 1}/5\n→ {goal_cell}\nAlgo: {selected_algorithm}")
        self.root.update()
        path, total_cost, nodes_explored, time_ms = run_algorithm(
            selected_algorithm, self.grid, self.cost_grid, self.robot_pos, goal_cell
        )
        if path is None:
            self.log_result(f"[{self.current_delivery + 1}] No path found to {goal_cell}!")
            self.current_delivery += 1
            self.deliver_next()
            return
        self.results.append({
            "delivery": self.current_delivery + 1,
            "algo": selected_algorithm,
            "cost": total_cost,
            "nodes": nodes_explored,
            "time_ms": time_ms,
            "steps": len(path)
        })
        self.log_result(f"--- Delivery {self.current_delivery + 1} ---")
        self.log_result(f"Algo  : {selected_algorithm}")
        self.log_result(f"Goal  : {goal_cell}")
        self.log_result(f"Cost  : {total_cost}")
        self.log_result(f"Nodes : {nodes_explored}")
        self.log_result(f"Time  : {time_ms} ms")
        self.log_result(f"Steps : {len(path)}")
        self.log_result("")
        self.current_path = path
        self.path_index = 0
        self.animating = True
        self.animate_robot()

    def animate_robot(self):
        # Moves the robot one step at a time along the planned path.
        # Redraws the grid after each step to show robot movement.
        # When path is complete triggers the next delivery after a short pause.
        if self.path_index < len(self.current_path):
            current_pos = self.current_path[self.path_index]
            self.robot_pos = current_pos
            self.draw_grid(path=self.current_path, robot=current_pos)
            self.path_index += 1
            self.root.after(120, self.animate_robot)
        else:
            self.animating = False
            self.current_delivery += 1
            self.root.after(400, self.deliver_next)

    def show_summary(self):
        # Prints a formatted summary table of all 5 deliveries in the results log.
        # Shows delivery number, algorithm used, cost, nodes explored per delivery.
        self.log_result("====== SUMMARY ======")
        self.log_result(f"{'#':<3} {'Algo':<7} {'Cost':<6} {'Nodes':<6}")
        self.log_result("-" * 26)
        for result in self.results:
            self.log_result(
                f"{result['delivery']:<3} {result['algo']:<7} "
                f"{result['cost']:<6} {result['nodes']:<6}"
            )
        self.log_result("=" * 26)

# Entry point — creates the tkinter window and starts the application
if __name__ == "__main__":
    root = tk.Tk()
    app = RobotApp(root)
    root.mainloop()