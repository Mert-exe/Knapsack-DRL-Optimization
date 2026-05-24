import numpy as np

class KnapsackEnv:
    def __init__(self, num_items=10, max_weight=50, weights=None, values=None, seed=None, max_items_total=1000):
        self.max_items_total = max_items_total
        if seed is not None:
            np.random.seed(seed)
            
        # Dışarıdan veri gelmişse onu kullan, yoksa rastgele üret
        if weights is not None and values is not None:
            self.weights = np.array(weights)
            self.values = np.array(values)
            self.num_items = len(self.weights)
            self.max_weight = max_weight
        else:
            self.num_items = num_items
            self.max_weight = max_weight
            self.weights = np.random.randint(1, 20, size=num_items)
            self.values = np.random.randint(10, 100, size=num_items)

        self.reset()

    def reset(self):
        self.current_weight = 0
        self.total_value = 0
        self.available_items = np.ones(self.num_items, dtype=bool) 
        self.done = False
        return self._get_state(self.max_items_total)

    def _get_state(self, max_items_total=1000):
        """
        State verilerini 0 ile 1 arasına normalize ederek döndürür.
        """
        remaining_cap_norm = (self.max_weight - self.current_weight) / self.max_weight
        
        w_norm = self.weights / (np.max(self.weights) + 1e-5)
        v_norm = self.values / (np.max(self.values) + 1e-5)
        avail = self.available_items.astype(float)
        
        pad_size = max_items_total - self.num_items
        if pad_size > 0:
            w_norm = np.concatenate([w_norm, np.zeros(pad_size)])
            v_norm = np.concatenate([v_norm, np.zeros(pad_size)])
            avail = np.concatenate([avail, np.zeros(pad_size)])
            
        state = np.concatenate([[remaining_cap_norm], w_norm, v_norm, avail])
        return np.array(state, dtype=np.float32)

    def step(self, action):
        if self.done: return self._get_state(), 0, self.done, {}
        
        # Geçersiz aksiyon veya zaten seçilmiş eşya cezası
        if action >= self.num_items or not self.available_items[action]:
            return self._get_state(), -100.0, True, {"msg": "Geçersiz!"}

        item_weight, item_value = self.weights[action], self.values[action]
        
        # Eşya çantaya sığıyorsa
        if self.current_weight + item_weight <= self.max_weight:
            self.current_weight += item_weight
            self.total_value += item_value
            self.available_items[action] = False
            
            # --- REWARD SHAPING (ÖDÜL ŞEKİLLENDİRME) ---
            # 1. Temel Ödül: Eşyanın net kârı
            base_reward = item_value
            
            # 2. Verimlilik Bonusu: Eşyanın (Değer / Ağırlık) oranı
            # Bu, ajana Greedy algoritmasının temel mantığını aşılar.
            efficiency = item_value / item_weight
            efficiency_bonus = efficiency * 5.0 # Katsayıyı öğrenmeyi tetiklemek için artırdık
            
            reward = base_reward + efficiency_bonus
            
            # 3. Doluluk Bonusu: Çanta %95+ doluluğa ulaştıysa teşvik et
            if (self.current_weight / self.max_weight) > 0.98:
                reward += 100.0
                
        else:
            # Kapasite aşımı cezası: Kalan boş alana göre dinamik ceza
            # Çok fazla boş yer varken kapasiteyi aşarsa daha büyük ceza alır.
            remaining_gap = self.max_weight - self.current_weight
            reward = -10.0 - (remaining_gap * 0.5) 
            self.done = True

        # Tüm eşyalar bittiyse bitir
        if not np.any(self.available_items): self.done = True
        
        return self._get_state(self.max_items_total), float(reward), self.done, {}