import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
from wemh_net import TinyWEMHNet, wemh_loss

# ==============================================================================
# WEMH Interactive Researcher (PoC)
# Обучение WEMH-Net на локальной базе знаний (скачанных статьях) и 
# запуск интерактивного режима (выбор самой слабой/глубокой гипотезы).
# ==============================================================================

# 1. Загрузка знаний и простой Char-Level Токенизатор
kb_files = glob.glob("paper_*.txt")
text_data = ""
for f in kb_files:
    with open(f, 'r', encoding='utf-8') as file:
        text_data += file.read() + "\n"

chars = sorted(list(set(text_data)))
vocab_size = len(chars)
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}

def encode(string):
    return [char2idx.get(c, 0) for c in string]

def decode(indices):
    return "".join([idx2char.get(i, "?") for i in indices])

data = torch.tensor(encode(text_data), dtype=torch.long)

# 2. Гиперпараметры для микро-модели (CPU Friendly)
SEQ_LEN = 32
BATCH_SIZE = 8
EPOCHS = 100 # Для PoC сделаем быстро
M_HYPOTHESES = 4 # 4 параллельные мысли

print(f"Размер базы знаний: {len(text_data)} символов. Словарь: {vocab_size} символов.")
print("Инициализация WEMH Researcher Core...")

model = TinyWEMHNet(vocab_size=vocab_size, d_model=64, n_heads=4, n_layers=2, M=M_HYPOTHESES)
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 3. Цикл обучения (Denoising / Reconstruction)
def get_batch():
    ix = torch.randint(len(data) - SEQ_LEN, (BATCH_SIZE,))
    x = torch.stack([data[i:i+SEQ_LEN] for i in ix])
    
    # Эмуляция диффузии: зашумляем инпут (заменяем 30% токенов случайными)
    noise_mask = torch.rand(x.shape) < 0.3
    noisy_x = x.clone()
    noisy_x[noise_mask] = torch.randint(0, vocab_size, (noise_mask.sum().item(),))
    
    return noisy_x, x # Вход - шум, таргет - чистый текст

print("Начинаем загрузку знаний в модель (Training)...")
model.train()
for step in range(EPOCHS):
    noisy_x, targets = get_batch()
    optimizer.zero_grad()
    
    logits = model(noisy_x) # (B, M, seq_len, vocab_size)
    loss, wta, ent = wemh_loss(logits, targets, weakness_alpha=0.2)
    
    loss.backward()
    optimizer.step()
    
    if (step+1) % 20 == 0:
        print(f"Шаг {step+1:3d} | WTA Ошибка: {wta.item():.4f} | Энтропия (Свобода): {ent.item():.4f}")

print("\nОбучение завершено. Запускаем интерактивный модуль (Simulated).")

# 4. Интерактивный Ресерчер
def ask_researcher(prompt):
    model.eval()
    print(f"\n[USER]: {prompt}")
    
    # Паддинг или обрезка промпта до SEQ_LEN
    encoded = encode(prompt)
    if len(encoded) > SEQ_LEN:
        encoded = encoded[:SEQ_LEN]
    else:
        encoded = encoded + [0]*(SEQ_LEN - len(encoded))
        
    x = torch.tensor([encoded], dtype=torch.long)
    
    with torch.no_grad():
        logits = model(x) # (1, M, seq_len, vocab_size)
    
    # Модель сгенерировала M гипотез. Посмотрим на них.
    print(f"[WEMH-Net]: Сгенерировано {M_HYPOTHESES} параллельных ветки размышлений.")
    
    hypotheses_texts = []
    entropies = []
    
    for m in range(M_HYPOTHESES):
        hyp_logits = logits[0, m, :, :] # (seq_len, vocab_size)
        probs = torch.softmax(hyp_logits, dim=-1)
        
        # Декодируем (жадно для визуализации)
        pred_idx = torch.argmax(probs, dim=-1).tolist()
        hyp_text = decode(pred_idx)
        hypotheses_texts.append(hyp_text)
        
        # Считаем энтропию этой ветки (Слабость / Опциональность)
        ent = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean().item()
        entropies.append(ent)
        
        print(f"  -> Ветка {m+1} [Энтропия: {ent:.2f}]: {hyp_text[:40]}...")

    # Выбор ЛУЧШЕЙ гипотезы через принцип Weakness (Максимальная Энтропия)
    best_m = max(range(M_HYPOTHESES), key=lambda i: entropies[i])
    
    print(f"\n[Выбор Архитектуры]: Активирован принцип Weakness (самая широкая гипотеза).")
    print(f"Выбрана Ветка {best_m+1}.")
    print(f"\n[RESEARCHER ASSISTANT]: {hypotheses_texts[best_m]}")

# Тестовые промпты
ask_researcher("Explorative Models can")
ask_researcher("Generalisation is")
