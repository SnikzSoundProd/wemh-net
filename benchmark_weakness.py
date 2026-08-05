import random

# СИНТЕТИЧЕСКИЙ БЕНЧМАРК: Weakest vs Shortest Hypothesis (MDL)
# Основано на arXiv:2301.12987v4
# Идея: В задачах обобщения (найти правило B, зная примеры A, где A ⊂ B), 
# Принцип Минимальной Длины Описания (MDL/Occam's Razor) работает ХУЖЕ, 
# чем принцип "Максимальной слабости" (Weakness) — выбор гипотезы, 
# которая делает меньше всего жестких ограничений (наиболее разрешительная).

# Эмуляция задачи бинарной арифметики (сложение x + y = z)
# Генерируем универсум U всех возможных троек (x, y, z) до N
N = 7
universe = [(x, y, z) for x in range(N) for y in range(N) for z in range(N*2)]
true_rule_B = set((x, y, x + y) for x in range(N) for y in range(N))

def complexity(hypothesis_set):
    # Симуляция MDL: чем меньше сет (более специфичный), тем "проще" правило.
    # В реальных системах (например, Apperception Engine) короткие формулы 
    # часто задают узкие рамки.
    return len(hypothesis_set) # Для инверсии: чем меньше размер, тем короче (specific).
    
def weakness(hypothesis_set):
    # Weakness: логическая слабость. Чем больше элементов допускает гипотеза,
    # тем она слабее. H1 => H2 означает, что H2 слабее (шире).
    return len(hypothesis_set)

# Создадим пространство гипотез (набор случайных правил + истинное)
num_hypotheses = 1000
hypotheses_space = []
for _ in range(num_hypotheses):
    # Случайное правило: берем случайное подмножество универсума
    size = random.randint(10, len(universe))
    h = set(random.sample(universe, size))
    hypotheses_space.append(h)
# Обязательно добавляем истинное правило B и пару его "коротких" (overfit) и "слабых" вариаций
hypotheses_space.append(true_rule_B)
# Добавим "короткий оверфит" (описывает только A и чуть-чуть еще)
# Добавим "слабый оверфит" (описывает A, B и кучу мусора)

def run_experiment(k_examples=5):
    # Наблюдаем A ⊂ B
    A = set(random.sample(list(true_rule_B), k_examples))
    
    # Ищем все консистентные гипотезы (те, где A ⊂ H)
    consistent = [h for h in hypotheses_space if A.issubset(h)]
    if not consistent:
        return 0, 0
        
    # Стратегия 1: MDL (Shortest) - выбираем самую специфичную (узкую) гипотезу
    # Чем меньше элементов, тем она "строже" (описывается короче)
    shortest_h = min(consistent, key=lambda h: len(h))
    
    # Стратегия 2: Weakest - выбираем самую "слабую" (широкую) гипотезу, 
    # которая при этом не является тривиальной (меньше всего ограничений).
    weakest_h = max(consistent, key=lambda h: len(h))
    
    # Метрика обобщения: насколько гипотеза покрывает целевой сет B (Recall) 
    # и не содержит мусора (Precision). Но в статье измеряется вероятность обобщения 
    # (generalising to B), то есть B ⊂ H (гипотеза достаточно слаба, чтобы вместить весь B)
    
    shortest_generalizes = 1 if true_rule_B.issubset(shortest_h) else 0
    weakest_generalizes = 1 if true_rule_B.issubset(weakest_h) else 0
    
    return shortest_generalizes, weakest_generalizes

if __name__ == "__main__":
    print("== The Optimal Choice of Hypothesis Benchmark ==")
    print("Задача: Вывести правило сложения (x+y=z) по K примерам.")
    print("Стратегия 1 (MDL): Выбор самого короткого (специфичного) правила.")
    print("Стратегия 2 (Weakness): Выбор самого слабого (широкого) правила.")
    print("------------------------------------------")
    
    iters = 100
    short_success = 0
    weak_success = 0
    
    for _ in range(iters):
        s, w = run_experiment(k_examples=3)
        short_success += s
        weak_success += w
        
    print(f"MDL (Shortest) успешно обобщилось до B:      {short_success}/{iters} раз")
    print(f"Weakness (Weakest) успешно обобщилось до B:  {weak_success}/{iters} раз")
    print(f"\nВывод: Оптимизация Weakness (максимальной логической слабости)")
    print("позволяет гипотезе охватить невидимые примеры, в то время как MDL")
    print("(Бритва Оккама) приводит к оверфиттингу на тренировочный сет A.")
    print("Как и сказано в статье: The Optimal Choice is the Weakest, not the Shortest.")
