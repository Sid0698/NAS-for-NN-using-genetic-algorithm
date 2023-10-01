import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset from CSV file
data = pd.read_csv('./diabetes.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.Tensor(X_train)
y_train_tensor = torch.Tensor(y_train)
X_test_tensor = torch.Tensor(X_test)
y_test_tensor = torch.Tensor(y_test)

# Move the data to the GPU
X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

# Create a custom neural network class
class Net(nn.Module):
    def __init__(self, num_layers, num_units, activation):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(8, num_units))  # Input layer
        for _ in range(num_layers):
            self.layers.append(nn.Linear(num_units, num_units))
            if activation == 'relu':
                self.layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                self.layers.append(nn.Sigmoid())
        self.layers.append(nn.Linear(num_units, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.sigmoid(x)
        return x

# Genetic Algorithm hyperparameters
population_size = 10
mutation_rate = 0.01
num_generations = 10

#search space 
search_space = {
    'num_layers': [1,2,3,4,5],
    'num_units': [8,16, 32],
    'activation': ['relu','sigmoid']
}

# Function to generate a random individual
def generate_individual():
    return {
        'num_layers': random.choice(search_space['num_layers']),
        'num_units': random.choice(search_space['num_units']),
        'activation': random.choice(search_space['activation'])
    }

# Function to generate an initial population
def generate_population():
    return [generate_individual() for _ in range(population_size)]

# Function to evaluate the fitness of an individual
def evaluate_fitness(individual):
    model = Net(individual['num_layers'], individual['num_units'], individual['activation']).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    # Evaluate the model on the test set
    model.eval().to(device)
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predicted_labels = (outputs >= 0.5).squeeze().long()
        accuracy = (predicted_labels == y_test_tensor.long()).sum().item() / len(y_test) * 100

    return accuracy

# Function to perform selection of individuals for mating
def selection(population):
    return random.choices(population, k=2, weights=[ind['fitness'] for ind in population])

# Function to perform crossover between two individuals
def crossover(parent1, parent2):
    child = {}
    for key in parent1:
        if random.random() < 0.5:
            child[key] = parent1[key]
        else:
            child[key] = parent2[key]
    return child

# Function to perform mutation on an individual
def mutation(individual):
    for key in individual:
        if random.random() < mutation_rate:
            if key == 'num_layers':
                individual[key] = random.choice(search_space[key])
            elif key == 'num_units':
                individual[key] = random.choice(search_space[key])
            elif key == 'activation':
                individual[key] = random.choice(search_space[key])
    return individual


best_individual = []
best_fitness = 0
# Main genetic algorithm loop
population = generate_population()
for generation in range(num_generations):
    print(f"Generation {generation + 1} / {num_generations}")

    # Evaluate fitness of each individual
    for individual in population:
        individual['fitness'] = evaluate_fitness(individual)

    # Sort population based on fitness
    population = sorted(population, key=lambda x: x['fitness'], reverse=True)
#     print(population[:2])
    # Print the best individual in the current generation
#     if best_fitness < population[0]['fitness']:
#         best_fitness = population[0]['fitness']
    best_individual.append(dict(population[0]))
    print(f"Best Individual: {best_individual[-1]}")

    # Perform selection, crossover, and mutation to create the next generation
    next_generation = [population[0]]
    while len(next_generation) < population_size:
        parent1, parent2 = selection(population)
        child = crossover(parent1, parent2)
        mutated_child = mutation(child)
        next_generation.append(mutated_child)

    # Replace the current generation with the next generation
    population = next_generation

# Print the final best individual
# best_individual = population[0]
print("Final Best Individual:")
# best_individual = sorted(best_individual, key=lambda x: x['fitness'], reverse=True)
for i in best_individual:
    print(i)