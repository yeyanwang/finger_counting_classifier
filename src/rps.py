import os
import numpy as np
import joblib
import random
import matplotlib.pyplot as plt

# mapping: 0 = rock, 2 = scissors, 5 = paper
label_to_move = {
    0: "Rock",
    2: "Scissors",
    5: "Paper"
}

# randomly generate computer move
def get_computer_move():
    return random.choice([0, 2, 5])

# compare player vs computer move and determine winner
def determine_winner(player, computer):
    if player == computer:
        return "Tie"
    if (player == 0 and computer == 2) or \
       (player == 2 and computer == 5) or \
       (player == 5 and computer == 0):
        return "Win"
    return "Lose"

# filter only rps labels
def filter_rps(X, y):
    mask = np.isin(y, [0, 2, 5])
    return X[mask], y[mask]

# load trained model & test data
def load_models(path, model_name):
    X_test = np.load(os.path.join(path, 'X_test_pca.npy'))
    y_test = np.load(os.path.join(path, 'y_test.npy'))

    model_path = os.path.join(path, f'knn_model_{model_name}.joblib')
    model = joblib.load(model_path)

    return X_test, y_test, model

def run_sim(X_test, y_test, model, rounds=5, show_images=False):
    X_test, y_test = filter_rps(X_test, y_test)

    if len(X_test) < rounds:
        rounds = len(X_test)

    indices = np.random.choice(len(X_test), rounds, replace=False)
    
    # result counters
    wins, losses, ties = 0, 0, 0
    correct = 0

    for i, idx in enumerate(indices):
        x = X_test[idx].reshape(1, -1)
        true_label = y_test[idx]

        pred_label = model.predict(x)[0]
        computer = get_computer_move()

        result = determine_winner(pred_label, computer)

        # track accuracy 
        if pred_label == true_label:
            correct += 1

        # game results
        if result == "Win":
            wins += 1
        elif result == "Lose":
            losses += 1
        else:
            ties += 1

        print(f"\nRound {i+1}")
        print(f"True Label: {label_to_move[true_label]}")
        print(f"Player (Predicted): {label_to_move[pred_label]}")
        print(f"Computer: {label_to_move[computer]}")
        print(f"Result: {result}")

        # display image
        if show_images:
            try:
                img = X_test[idx].reshape(64, 64)
                plt.imshow(img, cmap='gray')
                plt.title(f"True: {label_to_move[true_label]}")
                plt.axis('off')
                plt.show()
            except:
                pass

    # summary
    accuracy = correct / rounds if rounds > 0 else 0

    print(f"Finger Gesture Classification Accuracy: {accuracy:.2f}")
    print(f"Wins: {wins}, Losses: {losses}, Ties: {ties}")

    return {
        "accuracy": accuracy,
        "wins": wins,
        "losses": losses,
        "ties": ties
    }

# display game outcomes for final report...
def plot_outcome_examples(X_test, y_test, model, save_dir='./results'):
    """
    Finds one Win, one Lose, one Tie example from the test set
    and displays them side by side with the game result.
    """
    X_test, y_test = filter_rps(X_test, y_test)

    outcomes_needed = {"Win": None, "Lose": None, "Tie": None}
    indices = np.random.permutation(len(X_test))

    for idx in indices:
        if all(v is not None for v in outcomes_needed.values()):
            break

        x = X_test[idx].reshape(1, -1)
        true_label = y_test[idx]
        pred_label = model.predict(x)[0]
        computer = get_computer_move()
        result = determine_winner(pred_label, computer)

        if outcomes_needed[result] is None:
            outcomes_needed[result] = {
                'img': X_test[idx],
                'true': true_label,
                'pred': pred_label,
                'computer': computer,
                'result': result
            }

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = {"Win": "green", "Lose": "red", "Tie": "gray"}

    for ax, outcome in zip(axes, ["Win", "Lose", "Tie"]):
        data = outcomes_needed[outcome]

        if data is None:
            ax.set_title(f"{outcome}: no example found")
            ax.axis('off')
            continue

        try:
            img = data['img'].reshape(64, 64)
        except ValueError:
            side = int(len(data['img']) ** 0.5)
            img  = data['img'].reshape(side, side)

        ax.imshow(img, cmap='gray')
        ax.set_title(
            f"Result: {data['result']}\n"
            f"True: {label_to_move[data['true']]}\n"
            f"Player (Pred): {label_to_move[data['pred']]}\n"
            f"Computer: {label_to_move[data['computer']]}",
            color=colors[outcome],
            fontsize=10
        )
        ax.axis('off')

    plt.tight_layout()

    # save the figure
    save_path = os.path.join(save_dir, 'rps_outcomes.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()