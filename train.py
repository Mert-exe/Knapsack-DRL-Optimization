import numpy as np
import torch
import matplotlib.pyplot as plt
from env import KnapsackEnv
from agent import DQNAgent

def train_with_curriculum(total_episodes=3000, batch_size=64):
    """
    Müfredat Öğrenme (Curriculum Learning) ile DQN eğitimi.
    Problemi kademeli olarak zorlaştırarak ajanın genelleme yeteneğini artırır.
    """
    MAX_ITEMS_TOTAL = 1000  # Sabit sinir ağı giriş boyutu
    state_dim = 1 + (3 * MAX_ITEMS_TOTAL)
    action_dim = MAX_ITEMS_TOTAL
    
    # Ajanı bir kez oluşturuyoruz (Ağırlıklar kademeler arasında korunur)
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    agent.epsilon_decay = 0.998 # Daha yavaş azalma, her kademede keşif için
    
    # Müfredat Kademeleri (Stage-by-Stage)
    # episodes: Her kademede kaç bölüm eğitilecek
    # num_items: O kademedeki eşya sayısı
    curriculum = [
        {"num_items": 20,   "max_weight": 50,  "episodes": 600,  "desc": "Aşama 1: Temel Kavramlar"},
        {"num_items": 100,  "max_weight": 100, "episodes": 800,  "desc": "Aşama 2: Kapasite Yönetimi"},
        {"num_items": 500,  "max_weight": 200, "episodes": 800,  "desc": "Aşama 3: Büyük Ölçekli Seçim"},
        {"num_items": 1000, "max_weight": 500, "episodes": 800,  "desc": "Aşama 4: N=1000 Optimizasyonu"}
    ]
    
    rewards_history = []
    
    print(f"Müfredat Eğitimi Başlıyor... Hedef: N={MAX_ITEMS_TOTAL}")
    
    for stage in curriculum:
        n = stage["num_items"]
        ep_count = stage["episodes"]
        print(f"\n>>> {stage['desc']} (N={n}, {ep_count} Episode)")
        
        # Her kademe başında epsilon'u biraz artırarak "yeni dünyayı" keşfetmesini sağlıyoruz (Epsilon Bump)
        agent.epsilon = max(agent.epsilon, 0.3) 
        
        for episode in range(ep_count):
            # Mevcut kademenin zorluğuna göre ortamı kur
            env = KnapsackEnv(num_items=n, max_weight=stage["max_weight"], max_items_total=MAX_ITEMS_TOTAL)
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Geçerli eşyalar maskesi
                available_mask = np.zeros(MAX_ITEMS_TOTAL)
                available_mask[:n] = env.available_items
                
                # Aksiyon seçimi
                action = agent.select_action(state, available_mask)
                
                # Adım at
                next_state, reward, done, _ = env.step(action)
                
                # Hafızaya ekle ve eğit
                agent.memory.push(state, action, reward, next_state, done)
                agent.train_step(batch_size)
                
                state = next_state
                episode_reward += reward
            
            agent.update_epsilon()
            if episode % 10 == 0:
                agent.update_target_network()
            
            rewards_history.append(episode_reward)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                print(f"[{n}] Ep: {episode+1}/{ep_count} | Ort. Ödül: {avg_reward:.2f} | Eps: {agent.epsilon:.3f}")

    # Modeli kaydet
    torch.save(agent.q_network.state_dict(), "dqn_model.pth")
    print("\nEğitim Tamamlandı ve dqn_model.pth kaydedildi.")
    
    # Grafiği çiz
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.title("Müfredat Öğrenme Süreci (Curriculum Learning)")
    plt.xlabel("Toplam Bölüm")
    plt.ylabel("Ödül")
    plt.axvline(x=600, color='r', linestyle='--', label='N=20 Sonu')
    plt.axvline(x=1400, color='g', linestyle='--', label='N=100 Sonu')
    plt.axvline(x=2200, color='y', linestyle='--', label='N=500 Sonu')
    plt.legend()
    plt.savefig("ogrenme_egrisi_curriculum.png")
    
    return agent

if __name__ == "__main__":
    # Eğitim testi
    train_with_curriculum()
