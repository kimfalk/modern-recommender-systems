class MultiTokenPredictor:
    def __init__(self, model, cache_horizon_hours=24, beta=0.5):
        self.model = model
        self.beta = beta  # Time decay half-life
        self.horizon = cache_horizon_hours
        
    def create_training_labels(self, user_history, future_events):
        """
        Create multi-token labels with time decay weighting
        
        Args:
            user_history: List of past items
            future_events: List of (item_id, timestamp, reward) tuples
        """
        labels = []
        weights = []
        
        for item_id, timestamp, reward in future_events:
            # Calculate time-decayed weight
            hours_ahead = timestamp  # Relative to history end
            time_weight = np.exp(-self.beta * hours_ahead)
            
            # Combine time decay with reward signal
            final_weight = time_weight * reward
            
            labels.append(item_id)
            weights.append(final_weight)
            
        return labels, weights
    
    def compute_loss(self, logits, labels, weights):
        """
        Multi-label cross-entropy with time-decayed weights
        """
        # Get logits for all labeled items
        label_logits = logits[:, labels]
        
        # Apply weighted cross-entropy
        loss = -torch.sum(
            weights * F.log_softmax(label_logits, dim=-1)
        )
        return loss

# Usage during training
trainer = MultiTokenPredictor(model, cache_horizon_hours=24)

for batch in training_data:
    # Get next 24 hours of user activity
    future_items = get_future_events(batch['user_id'], hours=24)
    labels, weights = trainer.create_training_labels(
        batch['history'], 
        future_items
    )
    
    logits = model(batch['history'])
    loss = trainer.compute_loss(logits, labels, weights)
    loss.backward()
```
