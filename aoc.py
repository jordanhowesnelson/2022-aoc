#### General 
import string
import numpy as np

def read_data(date_num, kind):
    path = f'../data/{kind}-day{date_num}.txt'
    with open(path) as f:
        data = f.read().splitlines()
    return data

#### Day 1

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

#### Day 2 ####

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

#### Day 3 ####

letters = string.ascii_lowercase+string.ascii_uppercase
priority_map = {x:letters.index(x)+1 for x in letters}

class Sack:
    def __init__(self, contents):
        self.contents = contents
        self.compartment1 = []
        self.compartment2 = []
        self.common_contents = []
        self.prioritized_contents = get_priorities(contents)
    
        self.compart_contents(self.prioritized_contents)
        self.get_common_contents()
        
    def compart_contents(self, contents):
        n = int(len(contents)/2)
        self.compartment1 = contents[0:n]
        self.compartment2 = contents[n:]
    
    
    def get_common_contents(self):
        self.common_contents = [x for x in set(self.compartment1) if x in set(self.compartment2)]

def get_priorities(contents):
    return [priority_map[x] for x in contents]
    
def organize_sacks(content_list, get_total=True):
    sack_list = []
    priorities = []
    for x in content_list:
        sack = Sack(x)
        priorities += sack.common_contents
        sack_list.append(sack)
    print('priority total ', sum(priorities))
    return sack_list, sum(priorities)   

class ElfGroup:
    def __init__(self, sack_list):
        self.sack_list = sack_list
        self.get_badge_priority()
        
    def get_badge_priority(self):
        elf1, elf2, elf3 = [sorted(list(set(x.prioritized_contents))) for x in self.sack_list]
        self.badge = [y for y in [x for x in elf1 if x in elf2] if y in elf3][0]

def get_total_badges(sack_list, n=3):
    grouped_list = []
    badge_priority = []
    for val in range(0, len(sack_list), n):
        grp = ElfGroup(sack_list[val:val+n])
        grouped_list.append(grp)
        badge_priority.append(grp.badge)
    print('badge total ', sum(badge_priority))
    return grouped_list, sum(badge_priority)


#### Day 4 ####
class CleaningAssignment:
    def __init__(self, cleaning_pairs):
        self.raw_pairs = cleaning_pairs
        
        self.clean_raw()
        self.get_containment_status()
        self.get_overlap_status()
        
    def clean_raw(self):
        splits = [x for x in self.raw_pairs.split('-')]
        end, beg = splits[1].split(',')
        self.rangeA = list(range(int(splits[0]), int(end)+1))
        self.rangeB = list(range(int(beg), int(splits[2])+1))
        self.lenA = len(self.rangeA)
        self.lenB = len(self.rangeB)
        self.len_combined = self.lenA+self.lenB
        self.len_combined_set = len(set(self.rangeA+self.rangeB))
        
    def get_containment_status(self):
        '''returns true if the smaller range is fully contained within the larger range'''
        max_size = max(self.lenA, self.lenB)
        self.contained = (max_size == self.len_combined_set)
        
    def get_overlap_status(self):
        '''returns true if there is any overlap between the two pairs'''
        self.overlapped = self.len_combined_set < self.len_combined
        
def process_assignments(cleaning_list):
    contained_list = []
    overlap_list = []
    processed_assignments = []
    for x in cleaning_list:
        pair = CleaningAssignment(x)
        processed_assignments.append(pair)
        contained_list.append(pair.contained)
        overlap_list.append(pair.overlapped)
    print('total contained assignments ', sum(contained_list))
    print('total overlapped assignments ', sum(overlap_list))
    return processed_assignments, (sum(contained_list), sum(overlap_list))

#### Day 5 ####
class CrateCrane:
    def __init__(self, map_directions, crate_mover_9000=True):
        self.raw_map_directions = map_directions
        self.splitter = map_directions.index('')
        self.directions = map_directions[self.splitter+1:]
        self.raw_stack_height = len(map_directions[self.splitter-1])
        self.raw_stacks = map_directions[:self.splitter-1]
        self.movements = [x.split(' ') for x in self.directions]
        self.crate_mover_9000 = crate_mover_9000
        
        self.process_map()

    def process_map(self):
        self.stack_dict = {}
        stack_list = []
        stacks_parsed = [[y for y in x] for x in self.raw_stacks]

        for row in range(0, self.raw_stack_height):
            stack = [x[row] for x in stacks_parsed if x[row].isalnum()]
            if stack == []:
                continue
            else:
                stack_list.append(stack)
        
        for stack_row in range(0, len(stack_list)):
            self.stack_dict[stack_row+1] = stack_list[stack_row]
            self.stack_dict[stack_row+1].reverse()
    
    def move_crate(self, parsed_direction):
        move = int(parsed_direction[1])
        c_from = int(parsed_direction[3])
        c_to = int(parsed_direction[-1])
        moving_crates = self.stack_dict[c_from][-move:]
        if self.crate_mover_9000:
            moving_crates.reverse()
        new_len = len(self.stack_dict[c_from])-move
        self.stack_dict[c_from] = self.stack_dict[c_from][:new_len]
        self.stack_dict[c_to] = self.stack_dict[c_to]+(moving_crates)
        
    def complete_movement(self):
        for moves in self.movements:
            self.move_crate(moves)
        self.top_boxes = ''.join([x[-1] for x in self.stack_dict.values() if x != []])
        
#### Day 6 ####
class CommunicationDevice:
    def __init__(self, signal):
        self.signal = signal
        self.marker_detection = None
        
        self.marker_detection = self.process_signal(n=4)
        self.message_detection = self.process_signal(n=14)
    
    def process_signal(self, n):
        set_list = []
        for idx in range(0, len(self.signal)):
            sig = self.signal[idx]
            set_list.append(sig)
            if len(set(set_list)) < len(set_list):
                #remove everything from the first index 
                set_list = set_list[set_list.index(sig)+1:]
            if len(set_list) == n:
                return idx+1
            
#### Day 7 ####
class FileSystem:
    def __init__(self, raw_tree):
        self.raw_tree = raw_tree
        self.n = len(self.raw_tree)
        self.tree_dict = {}
        self.cd=None
        self.path = []
        self.total_disk_space = 70000000
        self.update_disk_space = 30000000
        
        self.parse_tree()
        self.get_dir_sizes()
        
    def get_path_str(self):
        return ''.join(self.path)
    
    def get_file_size(self):
        return int(self.branch.split(' ')[0])
    
    def clean_branch(self):
        return self.branch.replace('$ cd ', '').replace('dir ', '')
        
    def parse_tree(self):
        self.branch = self.raw_tree[0]
        self.tree_dict[self.clean_branch()] = {'files':[], 'dir':[], 'fileSize':0, 'dirSize':0}
        self.path.append(self.clean_branch())
        for x in range(1, self.n):
            self.branch = self.raw_tree[x]
            if self.branch.endswith('..'):
                self.path.pop()
            elif self.branch.startswith('$ cd'): 
                self.path.append(self.clean_branch()+'/')
                self.tree_dict[self.get_path_str()] = {'files':[], 'dir':[], 'fileSize':0, 'dirSize':0}
                continue
            elif self.branch.startswith('$ ls'):
                continue
            elif self.branch.startswith('dir'):
                self.tree_dict[self.get_path_str()]['dir'].append(self.clean_branch())
            elif self.branch[0].isnumeric():
                self.tree_dict[self.get_path_str()]['files'].append(self.branch)
                self.tree_dict[self.get_path_str()]['fileSize'] += self.get_file_size()
                
    def get_dir_sizes(self):
        self.sub_dir_list = sorted(self.tree_dict.keys(), key=lambda x: x.count('/'), reverse=True)
        for key in self.sub_dir_list:
        #if they have no sub dirs just get total
            if self.tree_dict[key]['dir'] != []:
                for sub in self.tree_dict[key]['dir']:
                    self.tree_dict[key]['dirSize'] += self.tree_dict[key+sub+'/']['totalSize']
            self.tree_dict[key]['totalSize'] = (self.tree_dict[key]['dirSize'] + 
                                                self.tree_dict[key]['fileSize'])
    
    def get_total_size_limit(self, lim=100000, func=np.less_equal):
        '''returns list of dir totals all less than lim'''
        return [v['totalSize'] for k, v in self.tree_dict.items() if func(v['totalSize'], lim)]
    
    def get_delete_size(self):
        used_space = self.tree_dict['/']['totalSize']
        available_space = (self.total_disk_space - used_space)
        space_needed = self.update_disk_space - available_space
        return sorted(self.get_total_size_limit(lim=space_needed, func=np.greater))[0]
    
#### Day 8 ####
class TreeMap:
    def __init__(self, raw_map):
        self.raw_map = raw_map
        self.map = np.array([[int(y) for y in x] for x in self.raw_map])
        self.row, self.col = self.map.shape
        self.map_dict = {}
        
        self.assess_visibility()
        self.assess_scenery()
        
    def get_max(self, cur, arr_slice):
        return np.product(np.greater(cur, arr_slice))
    
    def assess_visibility(self):
        for r in range(self.row):
            for c in range(self.col):

                cd = r, c
                ch = self.map[cd]
                
                self.map_dict[cd] = {'n':None, 's':None, 'e':None, 'w':None}
                self.map_dict[cd]['height'] = ch

                #need to take a sub array from each direction
                #if its the max, it's visible
                #print(cd)
                #print('cur point', self.map[cd])

                #east
                #print('e arr to check', self.map[r, c+1:])
                neighbors = self.map[r, c+1:]
                self.map_dict[cd]['eNeigh'] = neighbors
                try:
                    e = self.get_max(ch, neighbors)
                except:
                    e = 1
                self.map_dict[cd]['e'] = e

                #west
                #print('w arr to check', self.map[r, :c])
                neighbors = self.map[r, :c]
                self.map_dict[cd]['wNeigh'] = np.flip(neighbors)
                try: 
                    w = self.get_max(ch, neighbors)
                except:
                    w = 1
                self.map_dict[cd]['w'] = w

                #north
                #print('n arr to check', self.map[:r, c])
                neighbors = self.map[:r, c]
                self.map_dict[cd]['nNeigh'] = np.flip(neighbors)
                try:
                    n = self.get_max(ch, neighbors)
                except:
                    n = 1
                self.map_dict[cd]['n'] = n

                #south
                #print('s arr to check', self.map[r+1:, c])
                neighbors = self.map[r+1:, c]
                self.map_dict[cd]['sNeigh'] = neighbors
                try:
                    s = self.get_max(ch, neighbors)
                except:
                    s = 1
                self.map_dict[cd]['s'] = s

                #eval vis
                self.map_dict[cd]['visible'] = np.max([e, w, n, s])
        self.num_vis = sum([v['visible'] for v in self.map_dict.values()])
        return
    
    def assess_scenery(self):
        for k in self.map_dict.keys():
            #print(k)
            scene_list = []
            cd = self.map_dict[k]['height']
            for hood in ['eNeigh', 'wNeigh', 'nNeigh', 'sNeigh']:
                scene_count = 0
                nei_list = self.map_dict[k][hood]
                if len(nei_list) == 0:
                    scene_list.append(scene_count)
                    continue
                else:
                    for idx in range(len(nei_list)):
                        #print(hood)
                        cn = nei_list[idx]
                        #print('cd', cd, 'cn', cn)
                        
                        if (cn >= cd):
                            scene_count += 1
                            #print('cn >= cd, ending run')
                            break

                        elif (cd > cn):
                            scene_count += 1
                            #print('continuing to next')

                scene_list.append(scene_count)
            self.map_dict[k]['sceneList'] = scene_list
            self.map_dict[k]['sceneCount'] = np.product(scene_list)
        return

    def get_highest_scene(self):
        return np.max([v['sceneCount'] for v in self.map_dict.values()])