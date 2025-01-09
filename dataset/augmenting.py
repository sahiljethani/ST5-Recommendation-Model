
import random

def item_crop(sequence, eta=0.8):
    if len(sequence) == 1:
        return sequence
    max_Lc = max(1, min(len(sequence) - 1, int(eta * len(sequence))))
    start = random.randint(0, len(sequence) - 1)
    end = min(start + max_Lc, len(sequence))
    return sequence[start:end]

def item_reorder(sequence, beta=0.7):
    if len(sequence) <= 2:
        return sequence
    Lr = max(2, min(len(sequence), int(beta * len(sequence))))
    start = random.randint(0, len(sequence) - Lr)
    reordered_part = sequence[start:start+Lr]
    random.shuffle(reordered_part)
    return sequence[:start] + reordered_part + sequence[start+Lr:]

def item_drop(sequence, delta=0.5):
    if len(sequence) == 1:
        return sequence
    max_Ld = max(1, min(len(sequence) - 1, int(delta * len(sequence))))
    num_to_drop = random.randint(1, max_Ld)
    drop_indices = set(random.sample(range(len(sequence)), num_to_drop))
    return [item for i, item in enumerate(sequence) if i not in drop_indices]

def random_augment(sequence):
    augmentation_functions = [
        (item_crop, "Crop"),
        (item_reorder, "Reorder"),
        (item_drop, "Drop")
    ]
    chosen_function, method_name = random.choice(augmentation_functions)
    return chosen_function(sequence), method_name
