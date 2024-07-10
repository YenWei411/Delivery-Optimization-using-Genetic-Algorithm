import csv
import matplotlib.pyplot as plt # type: ignore
import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog, messagebox
import random
import numpy as np

# Read depots from file
def read_depots():
    depots = []
    try:
        with open('depot_loc.txt', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                depots.append({
                    'id': int(row[0]),
                    'latitude': float(row[1]),
                    'longitude': float(row[2])
                })
    except FileNotFoundError:
        pass
    return depots

# Write depots to file
def write_depots(depots):
    with open('depot_loc.txt', 'w', newline='') as file:
        writer = csv.writer(file)
        for depot in depots:
            writer.writerow([depot['id'], depot['latitude'], depot['longitude']])

# Read customers from file
def read_customers():
    customers = []
    try:
        with open('customer_loc.txt', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                customers.append({
                    'id': int(row[0]),
                    'latitude': float(row[1]),
                    'longitude': float(row[2]),
                    'demand': int(row[3])
                })
    except FileNotFoundError:
        pass
    return customers

# Write customers to file
def write_customers(customers):
    with open('customer_loc.txt', 'w', newline='') as file:
        writer = csv.writer(file)
        for customer in customers:
            writer.writerow([customer['id'], customer['latitude'], customer['longitude'], customer['demand']])

# Read vehicles from file
def read_vehicles():
    vehicles = []
    try:
        with open('vehicle.txt', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                vehicles.append({
                    'type': row[0],
                    'capacity': int(row[1]),
                    'cost_per_km': float(row[2])
                })
    except FileNotFoundError:
        pass
    return vehicles

# Write vehicles to file
def write_vehicles(vehicles):
    with open('vehicle.txt', 'w', newline='') as file:
        writer = csv.writer(file)
        for vehicle in vehicles:
            writer.writerow([vehicle['type'], vehicle['capacity'], vehicle['cost_per_km']])

# Initialize data
depots = read_depots()
customers = read_customers()
vehicles = read_vehicles()

# Function to add a depot
def add_depot(latitude, longitude):
    depot_id = len(depots) + 1
    depots.append({
        'id': depot_id,
        'latitude': latitude,
        'longitude': longitude
    })
    write_depots(depots)

# Function to edit a depot
def edit_depot(depot_id, latitude=None, longitude=None):
    for depot in depots:
        if depot['id'] == depot_id:
            if latitude is not None:
                depot['latitude'] = latitude
            if longitude is not None:
                depot['longitude'] = longitude
            break
    write_depots(depots)

# Function to delete a depot
def delete_depot(depot_id):
    global depots
    depots = [d for d in depots if d['id'] != depot_id]
    write_depots(depots)

# Function to add a customer
def add_customer(latitude, longitude, demand):
    customer_id = len(customers) + 1
    customers.append({
        'id': customer_id,
        'latitude': latitude,
        'longitude': longitude,
        'demand': demand
    })
    write_customers(customers)

# Function to edit a customer
def edit_customer(customer_id, latitude=None, longitude=None, demand=None):
    for customer in customers:
        if customer['id'] == customer_id:
            if latitude:
                customer['latitude'] = latitude
            if longitude:
                customer['longitude'] = longitude
            if demand:
                customer['demand'] = demand
            break
    write_customers(customers)

# Function to delete a customer
def delete_customer(customer_id):
    global customers
    customers = [c for c in customers if c['id'] != customer_id]
    write_customers(customers)

# Function to add a vehicle
def add_vehicle(vehicle_type, capacity, cost_per_km):
    vehicles.append({
        'type': 'Type ' + vehicle_type,
        'capacity': capacity,
        'cost_per_km': cost_per_km
    })
    write_vehicles(vehicles)

# Function to edit a vehicle
def edit_vehicle(vehicle_type, capacity=None, cost_per_km=None):
    for vehicle in vehicles:
        if vehicle['type'] == 'Type ' + vehicle_type:
            if capacity:
                vehicle['capacity'] = capacity
            if cost_per_km:
                vehicle['cost_per_km'] = cost_per_km
            break
    write_vehicles(vehicles)

# Function to delete a vehicle
def delete_vehicle(vehicle_type):
    global vehicles
    vehicles = [v for v in vehicles if v['type'] != 'Type ' + vehicle_type]
    write_vehicles(vehicles)

# Function to add a customer
def add_customer_ui():
    lat = simpledialog.askfloat("Input", "Enter Latitude:")
    lon = simpledialog.askfloat("Input", "Enter Longitude:")
    demand = simpledialog.askinteger("Input", "Enter Demand:")
    if lat is not None and lon is not None and demand is not None:
        add_customer(lat, lon, demand)
        #messagebox.showinfo("Info", "Customer added successfully!")
    refresh_customer_list()

# Function to edit a customer
def edit_customer_ui():
    cid = simpledialog.askinteger("Input", "Enter Customer ID to Edit:")
    lat = simpledialog.askfloat("Input", "Enter New Latitude (or leave empty):")
    lon = simpledialog.askfloat("Input", "Enter New Longitude (or leave empty):")
    demand = simpledialog.askinteger("Input", "Enter New Demand (or leave empty):")
    edit_customer(cid, lat, lon, demand)
    refresh_customer_list()
    #messagebox.showinfo("Info", "Customer edited successfully!")

# Function to delete a customer
def delete_customer_ui():
    cid = simpledialog.askinteger("Input", "Enter Customer ID to Delete:")
    delete_customer(cid)
    refresh_customer_list()
    #messagebox.showinfo("Info", "Customer deleted successfully!")

# Function to refresh customer list
def refresh_customer_list():
    # Clear current content
    for i in customer_tree.get_children():
        customer_tree.delete(i)

    # Insert new content
    for customer in customers:
        customer_tree.insert("", "end", values=(customer['id'], customer['latitude'], customer['longitude'], customer['demand']))

def plot_locations():
    plt.figure(figsize=(10, 8))
    
    for depot in depots:
        plt.scatter(depot['longitude'], depot['latitude'], label=f'Depot {depot['id']}', marker='D', s=50)
    
    for customer in customers:
        plt.scatter(customer['longitude'], customer['latitude'], label=f"Customer {customer['id']}")

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.title('Customer Locations and Depot')
    plt.show()

def read_vehicle_types(filename):
    vehicle_types = set()
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) > 0:
                    vehicle_type = parts[0]
                    vehicle_types.add(vehicle_type)
    except FileNotFoundError:
        print(f"{filename} not found. Starting with an empty vehicle list.")
    return vehicle_types

# Function to add a vehicle
def add_vehicle_ui():
    vtype = simpledialog.askstring("Input", "Enter Vehicle Type:")
    vtype = vtype.upper()
    if vtype:
        vtypecheck = 'Type ' + vtype

        # Read existing vehicle types
        existing_vehicle_types = read_vehicle_types('vehicle.txt')

        # Check if the vehicle type is already registered
        if vtypecheck in existing_vehicle_types:
            messagebox.showinfo("Info", "This vehicle type is already registered.")
        else:
            capacity = simpledialog.askinteger("Input", "Enter Capacity:")
            cost = simpledialog.askfloat("Input", "Enter Cost per km:")
            add_vehicle(vtype, capacity, cost)
            messagebox.showinfo("Info", "Vehicle added successfully!")
    refresh_vehicle_list()

# Function to edit a vehicle
def edit_vehicle_ui():
    vtype = simpledialog.askstring("Input", "Enter Vehicle Type to Edit:")
    vtype = vtype.upper()
    capacity = simpledialog.askinteger("Input", "Enter New Capacity (or leave empty):")
    cost = simpledialog.askfloat("Input", "Enter New Cost per km (or leave empty):")
    edit_vehicle(vtype, capacity, cost)
    refresh_vehicle_list()
    #messagebox.showinfo("Info", "Vehicle edited successfully!")

# Function to delete a vehicle
def delete_vehicle_ui():
    vtype = simpledialog.askstring("Input", "Enter Vehicle Type to Delete:")
    vtype = vtype.upper()
    delete_vehicle(vtype)
    refresh_vehicle_list()
    #messagebox.showinfo("Info", "Vehicle deleted successfully!")


# Refresh vehicle list to populate the treeview
def refresh_vehicle_list():
    # Clear current content
    for i in vehicle_tree.get_children():
        vehicle_tree.delete(i)

    # Insert new content
    for vehicle in vehicles:
        vehicle_tree.insert("", "end", values=(vehicle['type'], vehicle['capacity'], vehicle['cost_per_km']))

# Function to add a depot
def add_depot_ui():
    lat = simpledialog.askfloat("Input", "Enter Depot Latitude:")
    lon = simpledialog.askfloat("Input", "Enter Depot Longitude:")
    if lat is not None and lon is not None:
        add_depot(lat, lon)
        messagebox.showinfo("Info", "Depot added successfully!")
        refresh_depot_list()

# Function to edit a depot
def edit_depot_ui():
    did = simpledialog.askinteger("Input", "Enter Depot ID to Edit:")
    lat = simpledialog.askfloat("Input", "Enter New Depot Latitude (or leave empty):")
    lon = simpledialog.askfloat("Input", "Enter New Depot Longitude (or leave empty):")
    edit_depot(did, lat, lon)
    refresh_depot_list()
    #messagebox.showinfo("Info", "Depot edited successfully!")

# Function to delete a depot
def delete_depot_ui():
    did = simpledialog.askinteger("Input", "Enter Depot ID to Delete:")
    delete_depot(did)
    refresh_depot_list()
    #messagebox.showinfo("Info", "Depot deleted successfully!")
    
# Refresh depot list to populate the treeview
def refresh_depot_list():
    # Clear current content
    for i in depot_tree.get_children():
        depot_tree.delete(i)

    # Insert new content
    for depot in depots:
        depot_tree.insert("", "end", values=(depot['id'], depot['latitude'], depot['longitude']))

# Euclidean Distance Calculation
def euclidean_distance(lat1, lon1, lat2, lon2):
    return 100 * np.sqrt((lon2 - lon1) ** 2 + (lat2 - lat1) ** 2)

# Population Initialization
def initialize_population(population_size, customers):
    population = []
    for _ in range(population_size):
        individual = customers[:]
        random.shuffle(individual)
        population.append(individual)
    return population

# Route Cost Calculation
def calculate_route_cost(route, depot, vehicle):
    total_distance = 0
    prev_location = depot
    total_demand = 0
    for customer in route:
        total_distance += euclidean_distance(
            prev_location['latitude'], prev_location['longitude'],
            customer['latitude'], customer['longitude']
        )
        total_demand += customer['demand']
        prev_location = customer
    
    total_distance += euclidean_distance(
        prev_location['latitude'], prev_location['longitude'],
        depot['latitude'], depot['longitude']
    )
    
    if total_demand > vehicle['capacity']:
        return float('inf')
    return total_distance * vehicle['cost_per_km']

# Fitness Calculation
def fitness(individual, depot, vehicles):
    total_cost = 0
    current_route = []
    current_demand = 0
    vehicle_index = 0

    for customer in individual:
        if current_demand + customer['demand'] <= vehicles[vehicle_index]['capacity']:
            current_route.append(customer)
            current_demand += customer['demand']
        else:
            total_cost += calculate_route_cost(current_route, depot, vehicles[vehicle_index])
            current_route = [customer]
            current_demand = customer['demand']
            vehicle_index = (vehicle_index + 1) % len(vehicles)
    
    if current_route:
        total_cost += calculate_route_cost(current_route, depot, vehicles[vehicle_index])
    
    return total_cost

# Tournament Selection
def tournament_selection(population, fitnesses, k=3):
    selected = random.sample(list(zip(population, fitnesses)), k)
    selected = sorted(selected, key=lambda x: x[1])
    return selected[0][0]

# Order Crossover
def order_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    
    child1 = [None] * size
    child1[start:end+1] = parent1[start:end+1]
    
    child2 = [None] * size
    child2[start:end+1] = parent2[start:end+1]
    
    def fill_child(child, parent):
        current_pos = (end + 1) % size
        for gene in parent:
            if gene not in child:
                child[current_pos] = gene
                current_pos = (current_pos + 1) % size
    
    fill_child(child1, parent2)
    fill_child(child2, parent1)
    
    return child1, child2

# Swap Mutation
def swap_mutation(individual, mutation_rate=0.1):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(individual) - 1)
            individual[i], individual[j] = individual[j], individual[i]
    return individual

# Genetic Algorithm
def genetic_algorithm(customers, depot, vehicles, population_size=100, generations=500, mutation_rate=0.1):
    population = initialize_population(population_size, customers)
    best_solution = None
    best_cost = float('inf')
    
    for generation in range(generations):
        fitnesses = [fitness(individual, depot, vehicles) for individual in population]
        new_population = []
        
        for _ in range(population_size // 2):
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            
            child1, child2 = order_crossover(parent1, parent2)
            child1 = swap_mutation(child1, mutation_rate)
            child2 = swap_mutation(child2, mutation_rate)
            
            new_population.extend([child1, child2])
        
        population = new_population
        
        for individual in population:
            cost = fitness(individual, depot, vehicles)
            if cost < best_cost:
                best_cost = cost
                best_solution = individual
        
        if generation % 10 == 0:
            print(f"Generation {generation}: Best cost = {best_cost}")
    
    return best_solution, best_cost

def run_genetic_algorithm():
    depot = depots[0]  # Assuming there is only one depot, you can modify this as needed
    best_solution, best_cost = genetic_algorithm(customers, depot, vehicles)

    # Clear the previous content in the output_text widget
    output_text.delete(1.0, tk.END)

    # Output the best solution and cost
    output_text.insert(tk.END, f"Best Cost: RM {best_cost:.2f}\n")
    output_solution(best_solution, depot, vehicles)

def output_solution(solution, depot, vehicles):
    vehicle_index = 0
    current_route = []
    current_demand = 0
    for customer in solution:
        if current_demand + customer['demand'] <= vehicles[vehicle_index]['capacity']:
            current_route.append(customer)
            current_demand += customer['demand']
        else:
            print_route(current_route, depot, vehicles[vehicle_index])
            current_route = [customer]
            current_demand = customer['demand']
            vehicle_index = (vehicle_index + 1) % len(vehicles)
    
    if current_route:
        print_route(current_route, depot, vehicles[vehicle_index])

def print_route(route, depot, vehicle):
    total_distance = 0
    prev_location = depot
    output_text.insert(tk.END, f"Vehicle {vehicle['type']}:\n")
    for customer in route:
        distance = euclidean_distance(
            prev_location['latitude'], prev_location['longitude'],
            customer['latitude'], customer['longitude']
        )
        total_distance += distance
        output_text.insert(tk.END, f"Depot -> Customer {customer['id']} ({distance:.3f} km) -> ")
        prev_location = customer
    
    distance = euclidean_distance(
        prev_location['latitude'], prev_location['longitude'],
        depot['latitude'], depot['longitude']
    )
    total_distance += distance
    output_text.insert(tk.END, f"Depot ({distance:.3f} km)\n")
    output_text.insert(tk.END, f"Round Trip Distance: {total_distance:.3f} km, Cost: RM {total_distance * vehicle['cost_per_km']:.2f}, Demand: {sum(c['demand'] for c in route)}\n\n")

# Initialize main Tkinter window
root = tk.Tk()
root.title("Delivery Optimization")
root.geometry("1920x1080")
# Define grid configuration
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)

# Customer Frame
customer_frame = tk.LabelFrame(root, text="Customer", padx=10, pady=10)
#customer_frame.pack(padx=10, pady=10, fill="x")
customer_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

tk.Button(customer_frame, text="Add", command=add_customer_ui).pack(side="left", padx=5, pady=5)
tk.Button(customer_frame, text="Edit", command=edit_customer_ui).pack(side="left", padx=5, pady=5)
tk.Button(customer_frame, text="Delete", command=delete_customer_ui).pack(side="left", padx=5, pady=5)
tk.Button(customer_frame, text="Show Locations", command=plot_locations).pack(side="left", padx=5, pady=5)
# Create Treeview
columns = ("ID", "Latitude", "Longitude", "Demand")
customer_tree = ttk.Treeview(customer_frame, columns=columns, show='headings' , height=15)
# Set column headings and width
customer_tree.heading("ID", text="ID")
customer_tree.column("ID", width=50, anchor="w")
customer_tree.heading("Latitude", text="Latitude")
customer_tree.column("Latitude", width=60, anchor="w")
customer_tree.heading("Longitude", text="Longitude")
customer_tree.column("Longitude", width=60, anchor="w")
customer_tree.heading("Demand", text="Demand")
customer_tree.column("Demand", width=60, anchor="w")

# Pack Treeview
customer_tree.pack(fill="x")
# Refresh the customer list to populate the treeview
refresh_customer_list()

# Vehicle Frame
vehicle_frame = tk.LabelFrame(root, text="Vehicle", padx=10, pady=10)
#vehicle_frame.pack(padx=10, pady=4, fill="x")
vehicle_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

tk.Button(vehicle_frame, text="Add", command=add_vehicle_ui).pack(side="left", padx=5, pady=5)
tk.Button(vehicle_frame, text="Edit", command=edit_vehicle_ui).pack(side="left", padx=5, pady=5)
tk.Button(vehicle_frame, text="Delete", command=delete_vehicle_ui).pack(side="left", padx=5, pady=5)
# Create Vehicle Treeview
vehicle_columns = ("type", "capacity", "cost_per_km")
vehicle_tree = ttk.Treeview(vehicle_frame, columns=vehicle_columns, show='headings')

# Set column headings, width, and alignment for vehicle
vehicle_tree.heading("type", text="Type")
vehicle_tree.column("type", width=20, anchor="w")
vehicle_tree.heading("capacity", text="Capacity")
vehicle_tree.column("capacity", width=20, anchor="w")
vehicle_tree.heading("cost_per_km", text="RM per Km")
vehicle_tree.column("cost_per_km", width=20, anchor="w")

# Pack Vehicle Treeview
vehicle_tree.pack(fill="x")

# Refresh the vehicle list to populate the treeview
refresh_vehicle_list()

# Depot Frame
depot_frame = tk.LabelFrame(root, text="Depot", padx=10, pady=10)
#depot_frame.pack(padx=10, pady=4, fill="x")
depot_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

tk.Button(depot_frame, text="Add", command=add_depot_ui).pack(side="left", padx=5, pady=5)
tk.Button(depot_frame, text="Edit", command=edit_depot_ui).pack(side="left", padx=5, pady=5)
tk.Button(depot_frame, text="Delete", command=delete_depot_ui).pack(side="left", padx=5, pady=5)
# Create Depot Treeview
depot_columns = ("ID", "Latitude", "Longitude")
depot_tree = ttk.Treeview(depot_frame, columns=depot_columns, show='headings')

# Set column headings, width, and alignment for depot
depot_tree.heading("ID", text="ID")
depot_tree.column("ID", width=20, anchor="w")
depot_tree.heading("Latitude", text="Latitude")
depot_tree.column("Latitude", width=40, anchor="w")
depot_tree.heading("Longitude", text="Longitude")
depot_tree.column("Longitude", width=40, anchor="w")

# Pack Depot Treeview
depot_tree.pack(fill="x")

# Refresh the depot list to populate the treeview
refresh_depot_list()

#Optimization frame
optimization_frame = tk.LabelFrame(root, text="Optimization", padx=10, pady=10)
#optimization_frame.pack(padx=10, pady=10, fill="x")
optimization_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

# Button to run genetic algorithm
tk.Button(optimization_frame, text="Run Genetic Algorithm", command=run_genetic_algorithm).pack()

# Text field to display output
output_text = tk.Text(optimization_frame, height=20, width=80)
output_text.pack()

# Run the Tkinter main loop
root.mainloop()
