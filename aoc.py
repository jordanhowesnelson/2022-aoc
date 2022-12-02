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

class RockPaperScissors:
    def __init__(self, instructions):
        self.instructions = instructions
        self.left_player = 0
        self.right_player = 0
        self.winning_hands = [[1, 3], [3, 2], [2, 1]] #(r > s, s > p, p > r) --> r:1, p:2, s:3 --> 1>3, 3>2, 2>1 
        self.rescored = False
        self.scored_rounds = None
        
        self.format_instructions()
        self.score_cards()
        self.score_play(self.scored_rounds)
        
    def format_instructions(self):
        self.rounds = [x.split(' ') for x in self.instructions]
    
    def score_cards(self):
        score_dict = {'A':1, 'B':2, 'C':3, 'X':1, 'Y':2, 'Z':3}
        self.scored_rounds = [[score_dict[p] for p in r] for r in self.rounds]
    
    def assign_points(self, roundy):
        self.left_player += roundy[0]
        self.right_player += roundy[1]

    def score_play(self, rounds):
        for x in rounds:
            self.assign_points(x)
            if x[0] == x[1]:
                self.assign_points([3, 3])
            elif x in self.winning_hands: 
                self.assign_points([6, 0])
            else:
                self.assign_points([0, 6])
    
    def get_new_card(self, roundy):
        if roundy[1] == 1: #you lose, opponent wins
            return [x for x in self.winning_hands if x[0] == roundy[0]][0]
        elif roundy[1] == 2: #draw
            return [roundy[0], roundy[0]]
        else: 
            #what is the win against the opponent (their card, in loose side)
            return [[roundy[0], x[0]] for x in self.winning_hands if roundy[0] == x[1]][0]
    
    def play_strategy_game(self):
        if self.rescored:
            print('game already rescored')
            return
        
        self.left_player = 0
        self.right_player = 0
        self.scored_strategy_rounds = []
        
        for rounds in self.scored_rounds:
            self.scored_strategy_rounds.append(self.get_new_card(rounds))
            
        self.score_play(self.scored_strategy_rounds)