import json
files = [
    ('All_Beauty', 81),
    ('Amazon_Fashion', 14),
    ('Appliances', 3),
    ('Baby_Products', 94),
    ('CDs_and_Vinyl', 35),
    ('Gift_Cards', 31),
    ('Handmade_Products', 28),
    ('Health_and_Personal_Care', 17),
    ('Magazine_Subscriptions', 94),
    ('Software', 13)
]
with open('/home/yumingfeng/repo/SumForU/results/user_study/user_study.jsonl', 'w') as out:
    for cat, idx in files:
        with open(f'/home/yumingfeng/repo/SumForU/dataset/data/processed/sft/test/{cat}.jsonl', 'r') as f:
            lines = f.readlines()
            entry = json.loads(lines[idx].strip())  # idx is 1-based
            json.dump(entry, out)
            out.write('\n')