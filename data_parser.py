import numpy as np

def parse_mknap(file_path):
    """
    OR-Library mknap formatındaki dosyaları okur.
    Format: n m best -> profits -> weights (m x n) -> capacities (m)
    """
    with open(file_path, 'r') as f:
        content = f.read().split()
    
    ptr = 0
    num_problems = int(content[ptr])
    ptr += 1
    problems = []

    for _ in range(num_problems):
        n = int(content[ptr]) # Eşya sayısı
        m = int(content[ptr+1]) # Kısıt sayısı
        # 3. değeri (optimal çözüm) atlıyoruz
        ptr += 3
        
        # 1. Kâr değerlerini oku (n adet)
        values = [float(x) for x in content[ptr:ptr+n]]
        ptr += n
        
        # 2. Ağırlıkları oku (m satır, n sütun = m*n adet)
        weights_flat = [float(x) for x in content[ptr:ptr+(m*n)]]
        ptr += (m*n)
        weights_matrix = np.array(weights_flat).reshape(m, n)
        
        # 3. Kapasiteleri oku (m adet)
        capacities = [float(x) for x in content[ptr:ptr+m]]
        ptr += m
            
        problems.append({
            "n": n,
            "values": np.array(values),
            "weights": weights_matrix[0], # İlk kısıtı ana ağırlık olarak alıyoruz
            "capacity": capacities[0]
        })
    
    return problems