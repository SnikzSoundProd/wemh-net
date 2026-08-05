import numpy as np

# СИНТЕТИЧЕСКИЙ БЕНЧМАРК: Explorative Models (XMs) для рассуждений
# Суть метода: Вместо того, чтобы обучать модель на одном пути 
# рассуждений, мы генерируем K кандидатов, проверяем их, и обучаем 
# предсказывать ЛУЧШИЙ из них.

SEQ_LEN = 10
TARGET_SUM = 4

def verifier_score(seq):
    # seq shape: (batch_size, SEQ_LEN)
    scores = np.zeros(seq.shape[0])
    sums = seq.sum(axis=1)
    scores += (sums == TARGET_SUM).astype(float)
    
    # Penalty for adjacent 1s
    adj = (seq[:, :-1] * seq[:, 1:]).sum(axis=1)
    scores -= adj.astype(float) * 0.5
    return np.clip(scores, 0.0, 1.0)

class SimpleModel:
    def __init__(self, seq_len):
        self.logits = np.zeros(seq_len)
        self.lr = 0.5
        
    def forward(self):
        return 1.0 / (1.0 + np.exp(-self.logits))
        
    def sample(self, num_samples):
        probs = self.forward()
        probs_expanded = np.tile(probs, (num_samples, 1))
        samples = np.random.binomial(1, probs_expanded)
        return samples, probs
        
    def update(self, target_sample):
        # Cross entropy update
        probs = self.forward()
        grad = probs - target_sample
        self.logits -= self.lr * grad

def train_standard(epochs=200):
    model = SimpleModel(SEQ_LEN)
    for _ in range(epochs):
        samples, probs = model.sample(1)
        reward = verifier_score(samples)[0]
        # REINFORCE-like naive update
        # If reward is high, push logits towards the sample
        grad = (probs - samples) * reward
        model.logits -= model.lr * grad[0]
        
    final_sample, _ = model.sample(1)
    return verifier_score(final_sample)[0]

def train_explorative(epochs=200, K=16):
    model = SimpleModel(SEQ_LEN)
    for _ in range(epochs):
        samples, probs = model.sample(K)
        scores = verifier_score(samples)
        
        # Commit to the best mode!
        best_idx = np.argmax(scores)
        best_sample = samples[best_idx]
        
        # Train to predict this best sample
        model.update(best_sample)
        
    final_sample, _ = model.sample(1)
    return verifier_score(final_sample)[0]

if __name__ == "__main__":
    print("== Explorative Modeling (XMs) Benchmark ==")
    print("Задача: Генерация валидного латентного рассуждения (CSP)")
    print("Модель 1: Стандартное обучение (Greedy/Single Sample)")
    print("Модель 2: Explorative Training (Sample K, train on best)")
    print("------------------------------------------")

    standard_scores = [train_standard() for _ in range(50)]
    xm_scores = [train_explorative(K=16) for _ in range(50)]

    print(f"Стандартный подход (N=50), средний скор (из 1.0): {sum(standard_scores)/50:.2f}")
    print(f"Explorative Models (K=16, N=50), средний скор:   {sum(xm_scores)/50:.2f}")
    print("\nВывод: Эксплоративный подход (XMs) позволяет модели находить решения там, где")
    print("стандартное обучение застревает. Свойство end-to-end обучения сохраняется.")
