import numpy as np

# СИНТЕТИЧЕСКИЙ БЕНЧМАРК: Multiple Hypothesis Prediction (MHP)
# Основано на arXiv:1612.00197
# Идея: Когда данные имеют неопределенность (multi-modal), обычная модель (SHP)
# пытается усреднить ответ, что дает плохой результат. Модель с множественными
# гипотезами (MHP) использует meta loss (штрафует только самую близкую гипотезу),
# что заставляет модель покрывать разные моды неопределенности.

def generate_bimodal_data(N):
    """
    Генерируем X от -5 до 5.
    Y мультимодален: с вероятностью 50% Y = X + 2, с вероятностью 50% Y = -X - 2
    """
    X = np.random.uniform(-5, 5, N)
    mode = np.random.binomial(1, 0.5, N)
    Y = np.where(mode == 1, X + 2, -X - 2)
    return X, Y

class SHP_Model:
    def __init__(self):
        # h(x) = w * x + b
        self.w = np.random.randn()
        self.b = np.random.randn()
        self.lr = 0.01

    def forward(self, x):
        return self.w * x + self.b

    def train(self, X, Y, epochs=1000):
        for _ in range(epochs):
            pred = self.forward(X)
            loss = (pred - Y)**2
            
            # Gradients (MSE)
            dw = 2 * np.mean((pred - Y) * X)
            db = 2 * np.mean(pred - Y)
            
            self.w -= self.lr * dw
            self.b -= self.lr * db

class MHP_Model:
    def __init__(self, M=2):
        # M гипотез: h_i(x) = w_i * x + b_i
        self.M = M
        self.w = np.random.randn(M)
        self.b = np.random.randn(M)
        self.lr = 0.01

    def forward(self, x):
        # Возвращает форму (M, len(x))
        return np.outer(self.w, x) + self.b[:, None]

    def train(self, X, Y, epochs=1000):
        for _ in range(epochs):
            preds = self.forward(X) # (M, N)
            
            # Вычисляем потери для каждой гипотезы
            losses = (preds - Y)**2 # (M, N)
            
            # Ищем лучшую гипотезу (Winner-Takes-All)
            best_idx = np.argmin(losses, axis=0) # (N,)
            
            # Обновляем градиенты ТОЛЬКО для победившей гипотезы на каждом сэмпле
            dw = np.zeros(self.M)
            db = np.zeros(self.M)
            
            for i in range(self.M):
                mask = (best_idx == i)
                if np.sum(mask) > 0:
                    dw[i] = 2 * np.mean((preds[i, mask] - Y[mask]) * X[mask])
                    db[i] = 2 * np.mean(preds[i, mask] - Y[mask])
            
            self.w -= self.lr * dw
            self.b -= self.lr * db

if __name__ == "__main__":
    print("== Multiple Hypothesis Prediction (MHP) Benchmark ==")
    print("Данные: Би-модальные (Y = X+2 или Y = -X-2)")
    print("SHP: Пытается предсказать одно значение (обычный MSE)")
    print("MHP: Предсказывает M=2 значений (Meta Loss, Winner-Takes-All)")
    print("------------------------------------------")
    
    np.random.seed(42)
    X_train, Y_train = generate_bimodal_data(1000)
    
    shp = SHP_Model()
    shp.train(X_train, Y_train, epochs=2000)
    
    mhp = MHP_Model(M=2)
    mhp.train(X_train, Y_train, epochs=2000)
    
    # Тест
    test_x = np.array([2.0])
    print(f"Тест для X = {test_x[0]}")
    print(f"Реальные моды: {test_x[0]+2} и {-test_x[0]-2}")
    print(f"SHP предсказание: {shp.forward(test_x)[0]:.2f} (усреднение, не попадает никуда)")
    
    mhp_preds = mhp.forward(test_x).flatten()
    print(f"MHP предсказания: {mhp_preds[0]:.2f} и {mhp_preds[1]:.2f} (каждая гипотеза поймала свою моду)")
    print("\nВывод: Meta-loss (min over M hypotheses) позволяет разделить неопределённость")
    print("и адекватно предсказывать мультимодальные распределения, избегая размытия.")
