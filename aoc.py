def read_data(date_num, kind):
    path = f'../data/{kind}-day{date_num}.txt'
    with open(path) as f:
        data = f.read().splitlines()
    return data

class Elf:
    def __init__(self):   
        self.inventory = []
        
    def add_inventory(self, item):
        self.inventory.append(int(item))
        self.get_total_calories()
        return
    
    def get_total_calories(self):
        self.total_calories = sum(self.inventory)
    

def get_elf_calorie_inventory(test):
    elf_group = [Elf()]
    for x in test:
        if x == '':
            elf_group.append(Elf())
        else:
            elf = elf_group.pop()
            elf.add_inventory(x)
            elf_group.append(elf)
    return elf_group

def ask_who_has_most(elf_list, n=1):
    return sum(sorted([x.total_calories for x in elf_list], reverse=True)[0:n])
