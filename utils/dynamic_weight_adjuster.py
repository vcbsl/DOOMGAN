class DynamicWeightAdjuster:
    """Dynamically adjusts loss weights to be inversely proportional to their loss values."""
    def __init__(self, initial_weights, adj_rate=0.05, min_w=0.01, max_w=100.0):
        self.weights = initial_weights.copy()
        self.adj_rate, self.min_w, self.max_w = adj_rate, min_w, max_w

    def _normalize(self):
        total = sum(self.weights.values())
        if total > 0:
            for k in self.weights: self.weights[k] /= total

    def update_weights(self, loss_values):
        epsilon = 1e-8
        inverse_losses = {k: 1.0 / (v + epsilon) for k, v in loss_values.items()}
        total_inverse = sum(inverse_losses.values())
        if total_inverse == 0: return self.weights

        target_weights = {k: v / total_inverse for k, v in inverse_losses.items()}
        for key in self.weights:
            if key in target_weights:
                adjustment = self.adj_rate * (target_weights[key] - self.weights[key])
                self.weights[key] = max(self.min_w, min(self.weights[key] + adjustment, self.max_w))
        
        self._normalize()
        return self.weights