import numpy as np
import random

def greedy_knapsack(weights, values, max_weight):
    """
    Sırt Çantası Problemi için Klasik Açgözlü (Greedy) Algoritması.
    Eşyaları (Değer / Ağırlık) oranına göre büyükten küçüğe sıralar ve
    çantaya sığdığı sürece ekler. Bu bizim YZ modelini kıyaslayacağımız temel (baseline) yöntemdir.
    """
    num_items = len(weights)
    
    # 1. Her eşya için (Değer / Ağırlık) oranını hesapla
    ratios = np.zeros(num_items)
    for i in range(num_items):
        # Ağırlık 0 ise çok yüksek bir oran vererek en başa alınmasını sağla
        if weights[i] > 0:
            ratios[i] = values[i] / weights[i]
        else:
            ratios[i] = 1e9
        
    # 2. Oranlara göre eşya indekslerini büyükten küçüğe sırala
    # np.argsort küçükten büyüğe sıralar, [::-1] ile listeyi tersine çevirip büyükten küçüğe yapıyoruz.
    sorted_indices = np.argsort(ratios)[::-1]
    
    total_weight = 0
    total_value = 0
    chosen_items = []
    
    # 3. Sıralanmış eşyaları kapasiteyi aşmayacak şekilde çantaya ekle
    for idx in sorted_indices:
        if total_weight + weights[idx] <= max_weight:
            total_weight += weights[idx]
            total_value += values[idx]
            chosen_items.append(idx)
            
    return total_value, total_weight, chosen_items

# Hızlı bir test yapmak istersen bu dosyanın en altına şu kodları ekleyip sadece bu dosyayı çalıştırabilirsin:
if __name__ == "__main__":
    test_weights = [10, 20, 30]
    test_values = [60, 100, 120]
    test_capacity = 50
    
    val, wt, items = greedy_knapsack(test_weights, test_values, test_capacity)
    print(f"Greedy Algoritması Sonucu -> Toplam Değer: {val}, Toplam Ağırlık: {wt}, Seçilen Eşyalar (İndeks): {items}")

def dp_knapsack(weights, values, max_weight):
    """
    Sırt Çantası Problemi için Dinamik Programlama (DP) Algoritması.
    Bu yöntem her zaman %100 kesin ve optimum (en iyi) sonucu bulur.
    Ancak zaman karmaşıklığı O(N * W) olduğu için N ve W büyüdüğünde sistemin çökmesine
    veya çok yavaşlamasına sebep olur. Performans kıyaslaması için eklendi.
    """
    num_items = len(weights)
    W = int(max_weight)
    
    # DP tablosu oluşturma
    # Satırlar: eşyalar (0'dan num_items'a), Sütunlar: kapasiteler (0'dan W'ye)
    dp = np.zeros((num_items + 1, W + 1))
    
    for i in range(1, num_items + 1):
        w = int(weights[i-1])
        v = values[i-1]
        for c in range(W + 1):
            if w <= c:
                dp[i][c] = max(dp[i-1][c], dp[i-1][c-w] + v)
            else:
                dp[i][c] = dp[i-1][c]
                
    return dp[num_items][W]

def genetic_knapsack(weights, values, max_weight, pop_size=100, generations=200, mutation_rate=0.05):
    """
    Sırt Çantası Problemi için Geliştirilmiş Genetik Algoritma (GA).
    """
    num_items = len(weights)
    if num_items == 0: return 0, 0, []
    
    # Başlangıç popülasyonu: Sadece rastgele değil, biraz da Greedy mantığıyla başlat
    population = np.random.randint(2, size=(pop_size, num_items))
    
    def calculate_fitness(individual):
        curr_w = np.sum(individual * weights)
        curr_v = np.sum(individual * values)
        if curr_w > max_weight:
            # Sert ceza yerine kapasite aşımına göre doğrusal ceza
            return max(0, curr_v - (curr_w - max_weight) * 100)
        return curr_v

    for gen in range(generations):
        fitness = np.array([calculate_fitness(ind) for ind in population])
        
        # Elitizm: En iyi 2 bireyi koru
        best_indices = np.argsort(fitness)[-2:]
        elites = population[best_indices].copy()
        
        # Seleksiyon
        if np.sum(fitness) == 0:
            probs = np.ones(pop_size) / pop_size
        else:
            # Negatif fitness değerlerini 0 yap (ceza alanlar)
            temp_fit = np.maximum(fitness, 0)
            if np.sum(temp_fit) == 0:
                probs = np.ones(pop_size) / pop_size
            else:
                probs = temp_fit / np.sum(temp_fit)
            
        indices = np.random.choice(np.arange(pop_size), size=pop_size, p=probs)
        population = population[indices]
        
        # Çaprazlama
        for i in range(0, pop_size - 1, 2):
            if np.random.rand() < 0.8: # %80 crossover ihtimali
                cp = np.random.randint(1, num_items)
                population[i, cp:], population[i+1, cp:] = population[i+1, cp:].copy(), population[i, cp:].copy()
                
        # Mutasyon
        mutation_mask = np.random.rand(pop_size, num_items) < mutation_rate
        population = np.where(mutation_mask, 1 - population, population)
        
        # Elitleri geri koy
        population[0:2] = elites

    # Sonuç
    fitness = np.array([calculate_fitness(ind) for ind in population])
    best_idx = np.argmax(fitness)
    best_individual = population[best_idx]
    
    # Kapasite kontrolü (Eğer hala aşıyorsa rastgele eşya çıkar - Repair)
    while np.sum(best_individual * weights) > max_weight:
        ones = np.where(best_individual == 1)[0]
        if len(ones) == 0: break
        best_individual[random.choice(ones)] = 0
        
    total_value = np.sum(best_individual * values)
    total_weight = np.sum(best_individual * weights)
    chosen_items = np.where(best_individual == 1)[0].tolist()
    
    return total_value, total_weight, chosen_items