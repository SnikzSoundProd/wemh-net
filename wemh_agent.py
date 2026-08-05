# WEMH (Weak Explorative Multiple-Hypothesis) Decoder
import numpy as np

class WEMH_DecisionEngine:
    def __init__(self, k_hypotheses=10):
        self.K = k_hypotheses

    def generate_hypotheses(self, context):
        """
        1. XMs (Explorative Models): Генерируем K разнообразных путей/решений.
        Здесь LLM (или алгоритм) выдает разные варианты, а не один жадный.
        """
        # Эмуляция: гипотеза - это набор ограничений (1 - строгое, 0 - гибкое)
        # Чем больше 1, тем "короче/строже" гипотеза (MDL).
        # Чем больше 0, тем "слабее/шире" гипотеза (Weakness).
        return np.random.randint(0, 2, size=(self.K, 5))

    def evaluate_consistency(self, hypotheses, environment_rules):
        """
        2. MHP (Multiple Hypothesis): Оставляем только те гипотезы, 
        которые не нарушают жесткие правила среды (консистентны).
        """
        consistent = []
        for h in hypotheses:
            # Эмуляция: гипотеза не должна противоречить базовым правилам
            if np.all(h * environment_rules <= environment_rules):
                consistent.append(h)
        return consistent

    def select_optimal(self, consistent_hypotheses):
        """
        3. Weakness Maximisation: Выбираем не самую строгую (MDL), 
        а самую СЛАБУЮ (Weakest) гипотезу из рабочих.
        Ту, которая оставляет максимум нулей (свободы/опциональности).
        """
        if not consistent_hypotheses:
            return None
            
        # Считаем "слабость" (количество нулей/свободных параметров)
        weakness_scores = [np.sum(h == 0) for h in consistent_hypotheses]
        best_idx = np.argmax(weakness_scores)
        
        return consistent_hypotheses[best_idx]

    def run(self, context, env_rules):
        # Шаг 1: Разведка (XMs)
        hyps = self.generate_hypotheses(context)
        
        # Шаг 2: Выживание (MHP)
        valid_hyps = self.evaluate_consistency(hyps, env_rules)
        
        # Шаг 3: Выбор самого гибкого (Weakness)
        best_plan = self.select_optimal(valid_hyps)
        
        return best_plan

# Симуляция работы на примере AI-витрины / Web3
if __name__ == "__main__":
    engine = WEMH_DecisionEngine(k_hypotheses=50)
    
    # Жесткие правила среды (например: не терять деньги, не нарушать лимиты API)
    # 1 - правило активно, 0 - правило не применяется
    env_rules = np.array([1, 0, 0, 1, 0]) 
    
    plan = engine.run(context="Создать стратегию для бота", env_rules=env_rules)
    print("Оптимальный план (маска ограничений):", plan)
    print("Выбрана стратегия, которая соблюдает правила, но оставляет максимум гибкости.")
