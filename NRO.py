import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from math import gamma, log
import tempfile
from multiprocessing import Pool, cpu_count
from scipy.stats import sem
import time

# Constants
POPULATION_SIZE = 500
MAX_GENERATIONS = 30
PATIENCE = 5  # Early stopping if no improvement for 5 generations
ALPHA = 0.05  # Scaling factor for Lévy flight
BETA = 1.8    # Parameter for Lévy distribution
NUM_RUNS = 30  # Number of independent runs for stability and statistical analysis

# Dataset Paths
data_files = [
    "/Users/shahadsaeed/Desktop/Colon.arff",
    "/Users/shahadsaeed/Desktop/Lek1.arff",
    "/Users/shahadsaeed/Desktop/Lek2.arff",
    "/Users/shahadsaeed/Desktop/LungM.arff",
    "/Users/shahadsaeed/Desktop/Lym.arff",
    "/Users/shahadsaeed/Desktop/SRBCT.arff"
]

dataset_names = {
    "/Users/shahadsaeed/Desktop/Colon.arff": "Colon",
    "/Users/shahadsaeed/Desktop/Lek1.arff": "Leukemia1",
    "/Users/shahadsaeed/Desktop/Lek2.arff": "Leukemia2",
    "/Users/shahadsaeed/Desktop/LungM.arff": "Lung",
    "/Users/shahadsaeed/Desktop/Lym.arff": "Lymphoma",
    "/Users/shahadsaeed/Desktop/SRBCT.arff": "SRBCT"
}

# Lévy Flight Function (Equation 7, 9)
def levy_flight(beta):
    """Generates a Lévy flight step to introduce long-distance jumps for exploration."""
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / \
             (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, 1)
    v = np.random.normal(0, 1, 1)
    step = u / (np.abs(v) ** (1 / beta))
    return step

# Load ARFF Dataset with preprocessing
def load_arff_data(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()

    # Ensure unique attribute names
    attr_counts = {}
    new_content = []
    for line in content:
        if line.lower().startswith('@attribute'):
            parts = line.split()
            attr_name = parts[1].strip('" ')
            if attr_name in attr_counts:
                attr_counts[attr_name] += 1
                new_attr_name = f'"{attr_name}_{attr_counts[attr_name]}"'
                line = line.replace(attr_name, new_attr_name, 1)
            else:
                attr_counts[attr_name] = 0
            if '{' in line:
                before, nominals_part = line.split('{', 1)
                nominals, after = nominals_part.split('}', 1)
                cleaned_nominals = ','.join(nom.strip() for nom in nominals.split(','))
                line = f"{before}{{{cleaned_nominals}}}{after}"
        new_content.append(line)

    # Temporary file for modified ARFF
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.arff') as temp_file:
        temp_file.writelines(new_content)
        temp_file.seek(0)
        data, meta = arff.loadarff(temp_file.name)

    df = pd.DataFrame(data)

    # Decode categorical values
    for col in df.select_dtypes([object, 'category']):
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Impute missing values with mean
    X = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').fillna(pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').mean()).values

    # Z-score normalization
    X = StandardScaler().fit_transform(X)

    # Label encoding for the target
    y = LabelEncoder().fit_transform(y)

    return X, y

# Fitness Function
def fitness_function(args):
    """Computes the classification accuracy of a feature subset with a penalty for large subsets."""
    individual, X, y, clf = args
    features_selected = X[:, individual.astype(bool)]
    if features_selected.shape[1] == 0:
        return 0, 0, 0, 0  # No features selected

    loo = LeaveOneOut()
    y_true = []
    y_pred = []

    for train, test in loo.split(features_selected):
        try:
            clf.fit(features_selected[train], y[train])
            prediction = clf.predict(features_selected[test])[0]
            y_true.append(y[test][0])
            y_pred.append(prediction)
        except Exception as e:
            y_true.append(-1)  # Mark errors explicitly
            y_pred.append(-1)

    # Compute Precision, Recall, and F1-score on all predictions
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1 = report["weighted avg"]["f1-score"]
    
    # Compute Accuracy directly
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))

    # Penalize large feature subsets (scaled penalty)
    penalty = 0.0001 * np.sum(individual)  # Reduced penalty factor
    fitness = accuracy - penalty

    return max(fitness, 0), precision, recall, f1

# **Nuclear Fission Process (Equation 1)**
def nuclear_fission(population, X, y, clf, generation):
    """Performs nuclear fission to create new solutions with more diversity."""
    new_population = []
    for individual in population:
        # **Step Size Adjustment (Equations 2 & 3)**
        sigma = (log(generation + 1) / (generation + 1)) * np.abs(individual - population.mean(axis=0))

        # **Mutation Factors (Equations 4 & 5)**
        P_ne_s = np.round(np.random.rand() + 1)
        P_ne_e = np.round(np.random.rand() + 2)

        # **New solution using Gaussian perturbation (Equation 1)**
        if np.random.rand() <= 0.5:
            new_sol = np.random.normal(population.max(axis=0), sigma) + np.random.rand() * (population.max(axis=0) - P_ne_s * individual)
        else:
            new_sol = np.random.normal(individual, sigma) + np.random.rand() * (population.max(axis=0) - P_ne_e * individual)

        # **Force Stronger Mutations (Introduce 20% random elements)**
        mutation_indices = np.random.choice(X.shape[1], size=int(0.2 * X.shape[1]), replace=False)
        new_sol[mutation_indices] = 1 - new_sol[mutation_indices]

        new_sol += ALPHA * levy_flight(BETA)  # **Introduce Lévy flight (Equation 7)**
        new_population.append(np.clip(new_sol, 0, 1))

    return np.array(new_population)

# **Nuclear Fusion Process**
def nuclear_fusion(population, X_best):
    """Performs nuclear fusion to refine solutions while preventing stagnation."""
    new_population = []
    for individual in population:
        # **Ionization Step (Equation 6)**
        partner1, partner2 = population[np.random.choice(len(population), 2, replace=False)]
        if np.random.rand() < 0.5:
            X_ion = partner1 + np.random.rand() * (partner2 - individual)
        else:
            X_ion = partner1 - np.random.rand() * (partner2 - individual)

        # **Lévy Flight to Prevent Stagnation (Equation 7)**
        if np.allclose(partner1, partner2):
            X_ion += ALPHA * levy_flight(BETA) * (individual - partner1)

        # **Fusion Step (Equation 8, 9)**
        if np.random.rand() <= 0.5:
            X_fu = X_ion + np.random.rand() * (partner1 - X_best) + np.random.rand() * (partner2 - X_best)
        else:
            X_fu = X_ion + ALPHA * levy_flight(BETA) * (X_ion - X_best)

        # **Force Stronger Mutations (Random Reset for 10%)**
        if np.random.rand() < 0.1:  
            X_fu = np.random.rand(X_best.shape[0])

        new_population.append(np.clip(X_fu, 0, 1))
    return np.array(new_population)

def nro(X, y, clf):
    """Main NRO algorithm iterating over generations to optimize feature selection."""
    # Initialize population with more diversity
    population = np.random.rand(POPULATION_SIZE, X.shape[1]) > 0.3  # Lower threshold for more diversity

    # Ensure at least one feature is selected for each individual
    for i in range(POPULATION_SIZE):
        if np.sum(population[i]) == 0:  
            random_index = np.random.randint(0, X.shape[1])
            population[i][random_index] = 1  

        # Introduce 20% random variation to force exploration
        mutation_indices = np.random.choice(X.shape[1], size=int(0.2 * X.shape[1]), replace=False)
        population[i][mutation_indices] = 1 - population[i][mutation_indices]

    best_solution = None
    best_fitness = -np.inf
    no_improvement_count = 0  # Counter for early stopping

    for generation in range(MAX_GENERATIONS):
        # Evaluate Fitness using multiprocessing
        with Pool(cpu_count()) as pool:
            args = [(individual, X, y, clf) for individual in population]
            results = pool.map(fitness_function, args)

        fitness_scores = [r[0] for r in results]
        precisions = [r[1] for r in results]
        recalls = [r[2] for r in results]
        f1_scores = [r[3] for r in results]


        best_idx = np.argmax(fitness_scores)
        worst_fitness = min(fitness_scores)
        avg_fitness = np.mean(fitness_scores)

        # Early Stopping Check
        if fitness_scores[best_idx] > best_fitness:
            best_fitness = fitness_scores[best_idx]
            best_solution = population[best_idx]
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= PATIENCE:
            print(f"Early stopping at generation {generation+1} due to no improvement.")
            break

        # Print Progress of Each Generation
        print(f"Generation {generation+1}/{MAX_GENERATIONS} | Best: {best_fitness:.4f} | Avg: {avg_fitness:.4f} | Worst: {worst_fitness:.4f}")

        # Apply Nuclear Fission (Equation 1)
        population = nuclear_fission(population, X, y, clf, generation)
        
        # Apply Nuclear Fusion (Equation 8, 9)
        population = nuclear_fusion(population, best_solution)

        # Ensure No Empty Feature Subset
        for i in range(POPULATION_SIZE):
            if np.sum(population[i]) == 0:  
                random_index = np.random.randint(0, X.shape[1])
                population[i][random_index] = 1  
                
        
        # Evaluate best_solution only
        features_selected = X[:, best_solution.astype(bool)]
        y_true, y_pred = [], []

        if features_selected.shape[1] > 0:
            loo = LeaveOneOut()
            for train, test in loo.split(features_selected):
                clf.fit(features_selected[train], y[train])
                y_pred.append(clf.predict(features_selected[test])[0])
                y_true.append(y[test][0])

            accuracy = np.mean(np.array(y_true) == np.array(y_pred))
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            precision = report["weighted avg"]["precision"]
            recall = report["weighted avg"]["recall"]
            f1 = report["weighted avg"]["f1-score"]
        else:
            accuracy = precision = recall = f1 = 0.0

    return best_solution, accuracy, accuracy, accuracy, precision, recall, f1


# **Main Execution**
if __name__ == "__main__":
    classifiers = {
        "SVM": SVC(kernel='linear', C=1, random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=5)
    }
    
    results = []

    print("\nFinal Results Summary:")
    print(f"\n{'Dataset':<12} {'Classifier':<8} {'Total Genes':<12} {'Selected Genes':<15} {'Best Acc':<10} {'Worst Acc':<10} {'Avg Acc':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'CI (95%)':<15} {'Time (s)':<10}")
    print("-" * 130)

    for file_path in data_files:
        dataset_name = dataset_names[file_path]
        X, y = load_arff_data(file_path)
        total_genes = X.shape[1]
        
        for clf_name, clf in classifiers.items():
            best_accuracies = []
            worst_accuracies = []
            avg_accuracies = []
            precisions = []
            recalls = []
            f1_scores = []
            selected_genes_list = []
            execution_times = []

            print(f"\nProcessing Dataset: {dataset_name} | Classifier: {clf_name}")
            for run in range(NUM_RUNS):
                start_time = time.time()
                print(f"Run {run+1}/{NUM_RUNS} started...")
                best_solution, worst_acc, best_acc, avg_acc, avg_precision, avg_recall, avg_f1 = nro(X, y, clf)
                end_time = time.time()
                execution_times.append(end_time - start_time)

                best_accuracies.append(best_acc)
                worst_accuracies.append(worst_acc)
                avg_accuracies.append(avg_acc)
                precisions.append(avg_precision)
                recalls.append(avg_recall)
                f1_scores.append(avg_f1)
                selected_genes_list.append(np.sum(best_solution))

                print(f"Run {run+1}/{NUM_RUNS} completed | Time: {execution_times[-1]:.2f} seconds")

            # Calculate Confidence Intervals (95%)
            ci_lower = np.mean(avg_accuracies) - 1.96 * sem(avg_accuracies)
            ci_upper = np.mean(avg_accuracies) + 1.96 * sem(avg_accuracies)
            ci = f"[{ci_lower:.4f}, {ci_upper:.4f}]"

            # Calculate average execution time
            avg_time = np.mean(execution_times)

            # Store Results
            results.append([
                dataset_name, clf_name, total_genes, np.mean(selected_genes_list),
                np.max(best_accuracies), np.min(worst_accuracies), np.mean(avg_accuracies),
                np.mean(precisions), np.mean(recalls), np.mean(f1_scores),
                ci, avg_time
            ])

            # Print Results
            print(f"{dataset_name:<12} {clf_name:<8} {total_genes:<12} {np.mean(selected_genes_list):<15.2f} {np.max(best_accuracies):<10.4f} {np.min(worst_accuracies):<10.4f} {np.mean(avg_accuracies):<10.4f} {np.mean(precisions):<10.4f} {np.mean(recalls):<10.4f} {np.mean(f1_scores):<10.4f} {ci:<15} {avg_time:<10.2f}")

    # Save Results to CSV
    results_df = pd.DataFrame(results, columns=[
        "Dataset", "Classifier", "Total Genes", "Selected Genes", "Best Acc", "Worst Acc", "Avg Acc",
        "Precision", "Recall", "F1-Score", "CI (95%)", "Time (s)"
    ])
    results_df.to_csv("nro_results.csv", index=False)
    print("\nResults saved to 'nro_results.csv'.")