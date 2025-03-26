def read_unique_chars(filename):
    """读取文件并返回去重后的字符列表及集合"""
    chars = []
    seen = set()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            char = line.strip()
            if char not in seen:
                chars.append(char)
                seen.add(char)
    return chars, seen

# 读取v1和v2的去重数据
v1_chars, v1_seen = read_unique_chars('./dict/ppocr_keys_v1.txt')
v2_chars, _ = read_unique_chars('./dict/ppocr_keys_v2.txt')

# 合并v2中未出现在v1的字符
combined_chars = v1_chars.copy()
seen = v1_seen.copy()

for char in v2_chars:
    if char not in seen:
        combined_chars.append(char)
        seen.add(char)

# 将结果写回v1.txt
with open('ppocr_keys_v1.txt', 'w', encoding='utf-8') as f:
    for char in combined_chars:
        f.write(f"{char}\n")

print("合并完成，ppocr_keys_v1.txt已更新。")