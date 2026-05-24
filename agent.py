import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    """
    Dueling DQN Mimarisi: Durum değerini (Value) ve aksiyon avantajını (Advantage) 
    ayrı kollarla öğrenerek daha hassas bir tahminleme yapar.
    """
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        
        # Ortak Özellik Çıkarıcı Katman (Giriş)
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        
        # Değer Kolu (Value Stream): Durumun genel iyiliği
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Avantaj Kolu (Advantage Stream): Her aksiyonun (eşyanın) avantajı
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Dueling birleştirme formülü
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

class PrioritizedReplayBuffer:
    """
    Önceliklendirilmiş Deneyim Hafızası: TD-Hatası yüksek olan 
    deneyimlerin seçilme olasılığını artırır.
    """
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # Önceliklendirme derecesi
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
            
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance Sampling (IS) weights (Akademik stabilite için)
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        
        states, actions, rewards, next_states, dones = zip(*samples)
        return np.array(states), np.array(actions), np.array(rewards), \
               np.array(next_states), np.array(dones), indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio + 1e-5 # 0 olmaması için küçük değer

    def __len__(self):
        return len(self.buffer)

# 3. YZ Ajanı (Agent)
class DQNAgent:
    """
    Ortamla etkileşime giren, kararlar veren ve sinir ağını güncelleyen ana Ajan sınıfı.
    """
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma # Gelecekteki ödüllerin önem derecesi (Discount Factor)
        
        # Epsilon-Greedy parametreleri (Keşfetme vs Öğrendiğini Kullanma)
        self.epsilon = epsilon_start 
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Ekran kartı (GPU) varsa kullan, yoksa işlemciye (CPU) geç
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # İki farklı ağ kullanıyoruz: Biri aktif öğrenen, diğeri hedef (stabilite için)
        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = PrioritizedReplayBuffer(10000) # Son 10.000 işlemi hatırla

    def select_action(self, state, available_items):
        """
        Mevcut duruma göre bir eşya seçer.
        Bunu yaparken Epsilon-Greedy stratejisini ve Aksiyon Maskelemeyi (Action Masking) kullanır.
        """
        # Keşfetme (Rastgele seçim)
        if random.random() < self.epsilon:
            # Sadece henüz çantaya girmemiş eşyalar arasından rastgele seç
            valid_actions = np.where(available_items == 1)[0]
            if len(valid_actions) > 0:
                return random.choice(valid_actions)
            else:
                return random.randint(0, self.action_dim - 1)
        
        # Sömürü (Öğrendiği en iyi kararı seçme)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor).cpu().numpy()[0]
            
            # Zaten alınmış olan eşyaların seçilmemesi için Q değerlerini eksi sonsuz (-inf) yapıyoruz
            q_values[available_items == 0] = -float('inf')
            
            # Kalanlar arasından en yüksek Q değerine sahip olanı (argmax) seç
            return np.argmax(q_values)

    def train_step(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        # samples yerine indices ve weights değerlerini de alıyoruz
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            # Double DQN mantığı devam ediyor
            next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
            max_next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # TD-Hatasını hesapla (Öncelikleri güncellemek için)
        td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors.flatten())

        # Kayıp fonksiyonuna IS weightlerini uygula
        loss = (weights * nn.MSELoss(reduction='none')(q_values, target_q_values)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        # Keşfetme oranını yavaş yavaş azalt
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        # Belirli aralıklarla hedef ağı güncelle
        self.target_network.load_state_dict(self.q_network.state_dict())