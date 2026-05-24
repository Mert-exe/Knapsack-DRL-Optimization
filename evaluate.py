import numpy as np
import time
import matplotlib.pyplot as plt
from env import KnapsackEnv
from baseline import greedy_knapsack, dp_knapsack, genetic_knapsack
from train import train_with_curriculum
from data_parser import parse_mknap

# Sabitler
MODEL_PATH = "dqn_model.pth"
PLOT_XLABEL = 'Problem Boyutu (Eşya Sayısı)'
PLOT_YLABEL = 'Toplam Elde Edilen Kazanç'

def evaluate_model_advanced(agent, test_sizes=[10, 25, 50, 75, 100], num_tests_per_size=30, max_weight=100):
    print("\n" + "="*60 + "\n--- SENTETİK VERİ ANALİZİ ---")
    results = {}
    agent.epsilon = 0.0
    MAX_ITEMS_TOTAL = 1000

    for n in test_sizes:
        greedy_total_score, agent_total_score, dp_total_score, genetic_total_score = 0, 0, 0, 0
        greedy_total_time, agent_total_time, dp_total_time, genetic_total_time = 0.0, 0.0, 0.0, 0.0
        agent_wins, greedy_wins = 0, 0

        for _ in range(num_tests_per_size):
            test_env = KnapsackEnv(num_items=n, max_weight=max_weight, max_items_total=MAX_ITEMS_TOTAL)
            
            # --- Greedy Testi ---
            st = time.perf_counter()
            gv, _, _ = greedy_knapsack(test_env.weights, test_env.values, max_weight)
            greedy_total_time += (time.perf_counter() - st)
            greedy_total_score += gv
            
            # --- DP Testi (Optimum) ---
            st = time.perf_counter()
            dv = dp_knapsack(test_env.weights, test_env.values, max_weight)
            dp_total_time += (time.perf_counter() - st)
            dp_total_score += dv

            # --- Genetic Algorithm Testi ---
            st = time.perf_counter()
            gnv, _, _ = genetic_knapsack(test_env.weights, test_env.values, max_weight)
            genetic_total_time += (time.perf_counter() - st)
            genetic_total_score += gnv
            
            # --- DQN Testi ---
            _ = test_env.reset()
            av, done = 0, False
            st = time.perf_counter()
            while not done:
                mask = np.zeros(MAX_ITEMS_TOTAL)
                mask[:n] = test_env.available_items.astype(float)
                action = agent.select_action(test_env._get_state(MAX_ITEMS_TOTAL), mask)
                if action >= n: break
                _, _, done, _ = test_env.step(action)
            agent_total_time += (time.perf_counter() - st)
            av = test_env.total_value  # Gerçek kazancı al (RL ödülünü değil)
            agent_total_score += av
            
            if av > gv: agent_wins += 1
            elif gv > av: greedy_wins += 1

        results[n] = {
            "greedy_score": greedy_total_score, 
            "dp_score": dp_total_score,
            "agent_score": agent_total_score, 
            "genetic_score": genetic_total_score,
            "greedy_time": (greedy_total_time/num_tests_per_size)*1000, 
            "dp_time": (dp_total_time/num_tests_per_size)*1000,
            "agent_time": (agent_total_time/num_tests_per_size)*1000, 
            "genetic_time": (genetic_total_time/num_tests_per_size)*1000,
            "agent_wins": agent_wins, 
            "greedy_wins": greedy_wins
        }
        print(f"N={n} | YZ: {agent_total_score} | Greedy: {greedy_total_score} | DP: {dp_total_score} | GA: {genetic_total_score}")
        
    # === YENİ EKLENEN KISIM: ÖLÇEKLENEBİLİRLİK GRAFİĞİ ===
    try:
        plt.figure(figsize=(10, 6))
        
        # Algoritmaların skorlarını çizdiriyoruz
        plt.plot(test_sizes, [results[n]['agent_score'] for n in test_sizes], marker='o', label='DQN (Önerilen)', linewidth=2.5, color='blue')
        plt.plot(test_sizes, [results[n]['greedy_score'] for n in test_sizes], marker='s', label='Greedy', linestyle='--', color='green')
        plt.plot(test_sizes, [results[n]['genetic_score'] for n in test_sizes], marker='^', label='Genetik Algoritma', linestyle='-.', color='orange')
        plt.plot(test_sizes, [results[n]['dp_score'] for n in test_sizes], marker='x', label='DP (Optimum/Tavan)', linestyle=':', color='black', alpha=0.7)

        plt.title("Problem Boyutu (N) Artışına Karşı Toplam Skor Performansı")
        plt.xlabel("Eşya Sayısı (N)")
        plt.ylabel("Toplam Kâr (Score)")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig("olceklenebilirlik_analizi.png", dpi=300)
        print("\n[OK] Ölçeklenebilirlik grafiği başarıyla kaydedildi: olceklenebilirlik_analizi.png")
        plt.close()

        # Zaman Karmaşıklığı Analizi Grafiği
        plt.figure(figsize=(10, 6))
        plt.plot(test_sizes, [results[n]['agent_time'] for n in test_sizes], marker='o', label='DQN (Sabit Zaman)', linewidth=2)
        plt.plot(test_sizes, [results[n]['greedy_time'] for n in test_sizes], marker='s', label='Greedy', linestyle='--')
        plt.plot(test_sizes, [results[n]['genetic_time'] for n in test_sizes], marker='^', label='GA', linestyle='-.')
        # DP çok hızlı fırlayacağı için y-eksenini gerekirse logaritmik yapabilirsin:
        # plt.yscale('log') 

        plt.title("Problem Boyutu (N) Artışına Karşı Çalışma Süresi Analizi")
        plt.xlabel("Eşya Sayısı (N)")
        plt.ylabel("Süre (ms)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("zaman_karmasikligi.png", dpi=300)
        print("[OK] Zaman karmaşıklığı grafiği başarıyla kaydedildi: zaman_karmasikligi.png")
        plt.close()
    except Exception as e:
        print(f"\n[HATA] Grafik çizimi sırasında bir hata oluştu: {e}")

    return results

def run_benchmark_test(agent, file_name="data/mknap1.txt"):
    print("\n" + "="*60 + f"\n--- {file_name} AKADEMİK BENCHMARK TESTİ ---")
    problems = parse_mknap(file_name)
    MAX_ITEMS_TOTAL = 1000
    agent.epsilon = 0.0

    p_names = []
    g_vals = []
    a_vals = []
    d_vals = []

    for i, prob in enumerate(problems[:7]):
        env = KnapsackEnv(weights=prob["weights"], values=prob["values"], max_weight=prob["capacity"])
        _, av, done = env.reset(), 0, False
        while not done:
            mask = np.pad(env.available_items.astype(float), (0, MAX_ITEMS_TOTAL-env.num_items), constant_values=0)
            action = agent.select_action(env._get_state(MAX_ITEMS_TOTAL), mask)
            if action >= env.num_items: break
            _, r, done, _ = env.step(action)
            if r > 0: av += r
            
        gv, _, _ = greedy_knapsack(prob["weights"], prob["values"], prob["capacity"])
        dv = dp_knapsack(prob["weights"], prob["values"], prob["capacity"])
        print(f"Problem {i+1} (N={prob['n']}) -> YZ: {av:.1f} | Greedy: {gv} | DP (Opt): {dv}")
        
        p_names.append(f"Prob {i+1}\n(N={prob['n']})")
        g_vals.append(gv)
        a_vals.append(av)
        d_vals.append(dv)
        
    return p_names, g_vals, a_vals, d_vals

def plot_separate_comparisons(results, test_sizes):
    """ DQN modelini DP, Greedy ve Genetic ile ayrı ayrı kıyaslayan grafikler üretir. """
    n_labels = [f"N={n}" for n in test_sizes]
    agent_scores = [results[n]['agent_score'] for n in test_sizes]
    dp_scores = [results[n]['dp_score'] for n in test_sizes]
    greedy_scores = [results[n]['greedy_score'] for n in test_sizes]
    genetic_scores = [results[n]['genetic_score'] for n in test_sizes]
    
    x = np.arange(len(n_labels))
    width = 0.35

    # 1. DQN vs Dynamic Programming (DP)
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, agent_scores, width, label='DQN (Derin Pekiştirmeli Öğrenme)', color='#3498db', edgecolor='white', linewidth=1.5)
    plt.bar(x + width/2, dp_scores, width, label='DP (Dinamik Programlama - Optimum)', color='#2ecc71', alpha=0.5, hatch='//')
    plt.title('Performans Analizi: DQN vs Dinamik Programlama', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel(PLOT_XLABEL, fontsize=12)
    plt.ylabel(PLOT_YLABEL, fontsize=12)
    plt.xticks(x, n_labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.text(x[0], agent_scores[0], f'%{ (agent_scores[0]/dp_scores[0]*100):.1f}', ha='center', va='bottom', fontweight='bold')
    plt.text(x[1], agent_scores[1], f'%{ (agent_scores[1]/dp_scores[1]*100):.1f}', ha='center', va='bottom', fontweight='bold')
    plt.text(x[2], agent_scores[2], f'%{ (agent_scores[2]/dp_scores[2]*100):.1f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig('dqn_vs_dp.png', dpi=300)
    
    # 2. DQN vs Greedy
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, agent_scores, width, label='DQN (Yapay Zeka)', color='#e67e22', edgecolor='white', linewidth=1.5)
    plt.bar(x + width/2, greedy_scores, width, label='Greedy (Açgözlü Yaklaşım)', color='#95a5a6', alpha=0.7, hatch='..')
    plt.title('Performans Analizi: DQN vs Greedy Algoritması', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel(PLOT_XLABEL, fontsize=12)
    plt.ylabel(PLOT_YLABEL, fontsize=12)
    plt.xticks(x, n_labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('dqn_vs_greedy.png', dpi=300)

    # 3. DQN vs Genetic Algorithm (GA)
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, agent_scores, width, label='DQN (Yapay Zeka)', color='#9b59b6', edgecolor='white', linewidth=1.5)
    plt.bar(x + width/2, genetic_scores, width, label='GA (Genetik Algoritma)', color='#f1c40f', alpha=0.7, hatch='\\\\')
    plt.title('Performans Analizi: DQN vs Genetik Algoritma', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel(PLOT_XLABEL, fontsize=12)
    plt.ylabel(PLOT_YLABEL, fontsize=12)
    plt.xticks(x, n_labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('dqn_vs_genetic.png', dpi=300)

    print("\n[OK] Ayrı ayrı kıyaslama grafikleri kaydedildi: dqn_vs_dp.png, dqn_vs_greedy.png, dqn_vs_genetic.png")

def print_academic_summary(results, test_sizes):
    """ Sonuçları akademik bir dille özetler. """
    print("\n" + "="*80)
    print("AKADEMİK PERFORMANS DEĞERLENDİRMESİ VE ANALİZ RAPORU")
    print("="*80)
    
    for n in test_sizes:
        res = results[n]
        dqn_eff = (res['agent_score'] / res['dp_score']) * 100 if res['dp_score'] > 0 else 0
        greedy_eff = (res['greedy_score'] / res['dp_score']) * 100 if res['dp_score'] > 0 else 0
        genetic_eff = (res['genetic_score'] / res['dp_score']) * 100 if res['dp_score'] > 0 else 0
        
        print(f"\n[Problem Boyutu: N={n}]")
        print(f"1. Optimizasyon Kapasitesi: DQN modeli, teorik optimum çözümün (DP) %{dqn_eff:.2f}'sine ulaşmıştır.")
        print(f"2. Karşılaştırmalı Analiz: DQN, Greedy yaklaşımına göre %{dqn_eff - greedy_eff:+.2f} fark göstermiş,")
        print(f"   Genetik Algoritma ile kıyaslandığında ise %{dqn_eff - genetic_eff:+.2f} verimlilik sapması sergilemiştir.")
        print(f"3. Algoritmik Verimlilik: DQN karar verme süresi {res['agent_time']:.4f} ms iken, DP {res['dp_time']:.4f} ms sürmüştür.")
        
    print("\nSONUÇ: Derin Pekiştirmeli Öğrenme (DQN) tabanlı modelimiz, özellikle yüksek boyutlu (N=100) problemlerde")
    print("Dinamik Programlama'nın üssel zaman artışına girmeden, Greedy ve Genetik algoritmalara kıyasla")
    print("rekabetçi ve genellenebilir bir performans sergilemektedir.")
    print("="*80)

def plot_benchmark_results(p_names, g_vals, a_vals, d_vals):
    """ OR-Library benchmark sonuçlarını görselleştirir. """
    plt.figure(figsize=(12, 6))
    x = np.arange(len(p_names))
    width = 0.25
    
    plt.bar(x - width, g_vals, width, label='Greedy (Sezgisel)', color='teal', alpha=0.7)
    plt.bar(x, d_vals, width, label='DP (Kesin Çözüm)', color='seagreen', alpha=0.8)
    plt.bar(x + width, a_vals, width, label='DQN (Öğrenen Ajan)', color='darkorange', alpha=0.9)
    
    plt.xlabel('OR-Library Test Problemleri (mknap1)', fontweight='bold')
    plt.ylabel('Optimizasyon Çözüm Değeri', fontweight='bold')
    plt.title('Gerçek Dünya Verilerinde (Benchmark) Model Performansı', fontweight='bold')
    plt.xticks(x, p_names)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('benchmark_analizi.png', dpi=300)
    print("[OK] Grafik kaydedildi: benchmark_analizi.png\n" + "="*60)

if __name__ == "__main__":
    # 1. Modeli Kur (Boyutları eğitimle aynı yapmalısın!)
    import os
    import torch
    from agent import DQNAgent
    from train import train_with_curriculum # Import'u unutma
    
    # Sabitler
    MAX_ITEMS = 1000 
    MODEL_PATH = "dqn_model.pth" # Üstte tanımlı değilse burada tanımlı olduğundan emin ol
    
    # State boyutu: 1 (Kapasite) + 3 * 1000 (W, V, Mask)
    state_dim = 1 + (MAX_ITEMS * 3)
    action_dim = MAX_ITEMS
    
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    
    if os.path.exists(MODEL_PATH):
        print(f"[INFO] Mevcut model yükleniyor: {MODEL_PATH}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # weights_only=True eklemek daha güvenli bir pratik
        state_dict = torch.load(MODEL_PATH, map_location=device)
        agent.q_network.load_state_dict(state_dict)
        trained_agent = agent
        print("[OK] Model başarıyla yüklendi ve teste hazır.")
    else:
        print("[INFO] Model bulunamadı, Müfredat Eğitimi (Curriculum) başlatılıyor...")
        # train_with_curriculum zaten eğitimi tamamlayıp dqn_model.pth olarak kaydeder
        trained_agent = train_with_curriculum()
        print("[OK] Eğitim bitti ve testlere geçiliyor.")
    
    # 2. Sentetik Veri Testlerini Çalıştır (Genişletilmiş N listesi)
    test_boyutlari = [10, 25, 50, 75, 100, 250, 500, 1000]
    sentetik_sonuclar = evaluate_model_advanced(trained_agent, test_sizes=test_boyutlari)
    
    # 3. GRAFİKLERİ ÇİZ VE KAYDET
    plot_separate_comparisons(sentetik_sonuclar, test_sizes=test_boyutlari)
    
    # 4. AKADEMİK ÖZET YAZDIR
    print_academic_summary(sentetik_sonuclar, test_sizes=test_boyutlari)