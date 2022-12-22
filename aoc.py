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
    
#### Day 9 ####

def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))

class Snake:
    def __init__(self, moves, knot_list=['head', 'tail']):
        self.moves = moves
        self.knot_list = knot_list
        self.track_dict = {}

        self.move_dict = {'R':lambda x: (x, 0), #would need to flip for range
                          'L':lambda x: (-x, 0),
                          'U':lambda y: (0, y),
                          'D':lambda y: (0, -y) #would need to flip for range
                         }
        
        for knot in self.knot_list:
            self.__setattr__(knot, [0,0])
        
        self.check_and_record((0,0), record_for=self.knot_list)
        self.process_instructions()
        
    def build_dict(self, move):
        self.track_dict[move] = {}
        for x in self.knot_list:
            self.track_dict[move][x] = 0
        
    def parse_move(self, move):
        self.current_move = move
        d_, n_ = move.split(' ')
        self.cur_direction = d_
        self.step_n = int(n_)
        self.cord_move = self.move_dict[self.cur_direction](self.step_n)

    
    def check_and_record(self, clean_move, record_for=['head']):
        check_for = tuple(clean_move)
        #print('dict check for', check_for)
        if check_for not in self.track_dict.keys():
            self.build_dict(check_for)
        for rec in record_for:
            self.track_dict[check_for][rec] += 1
            
    def get_passing_distance(self, head, tail):
        self.distance_between = distance(head, tail)
        return ((self.distance_between == 0) | 
                (self.distance_between == 1) |
                (self.distance_between == np.sqrt(2))
               )

    def process_move(self, move):
        #print('~~~~~~~move~~~~~~~', move)
        #org contents
        self.parse_move(move)
        
        #move H iteratively, checking tail each time
        for val in np.ones(self.step_n):
            for i in range(len(self.knot_list)-1):
                head = self.knot_list[i]
                tail = self.knot_list[i+1]

                if i == 0:
                    self.__setattr__(head, np.add(self.move_dict[self.cur_direction](int(val)), self.__getattribute__(head)))
                    self.check_and_record(self.__getattribute__(head), [head])
                self.distance_between = distance(self.__getattribute__(tail), self.__getattribute__(head))
#                 print(f'distance between {tail} and {head}', self.distance_between)
                if ((self.distance_between == 0) | 
                    (self.distance_between == 1) |
                    (self.distance_between == np.sqrt(2))
                   ):
#                     print('no move needed')
#                     print(f'{head} at', self.__getattribute__(head))
#                     print(f'{tail} at', self.__getattribute__(tail))  
#                     print()
                    continue
                    
                else:
                    new_loc = self.get_new_location(self.__getattribute__(head), self.__getattribute__(tail))
                    self.__setattr__(tail, np.add(new_loc, self.__getattribute__(tail)))
                    self.check_and_record(self.__getattribute__(tail), [tail])

#                 print(f'{head} at', self.__getattribute__(head))
#                 print(f'{tail} at', self.__getattribute__(tail))      
#                 print()

                
        return
            
    def process_instructions(self):
        self.prior_tail = np.array([0,0])
        for m in self.moves:
            self.process_move(m)

    
    def get_new_location(self, head, tail):
        '''return the new point'''
        x1, y1 = head
        x2, y2 = tail
        x_diff = x1 - x2
        y_diff = y1 - y2
        #if x's are zero, return np.sign of ys diff
        if x_diff == 0:
            move_n = np.sign(y_diff)
            move_1 = self.move_dict['U'](move_n)
            return move_1
        elif y_diff == 0:
            move_n = np.sign(x_diff)
            move_1 = self.move_dict['R'](move_n)
            return move_1
        else:
            x_move = np.sign(x_diff)  
            y_move = np.sign(y_diff)   

            return [x_move, y_move]
    
                
    def get_space_counts(self, report_for='tail'):
        return sum([x[report_for] >= 1  for x in  self.track_dict.values()])
    
#### Day 10 ####
class Signal:
    def __init__(self, signal):
        self.signal = signal
        self.last = signal.split(' ')[-1]
        self.symbol = ''

        if self.last == 'noop':
            self.action = self.last
            self.X = [0]
            self.cycle = 1
        else:
            self.action = 'addx'
            self.X = [0, int(self.last)]
            self.cycle = 2

class Tube:
    def __init__(self, signals, interestings=[]):
        self.signals = signals
        self.cycle_list = []
        self.cycle = 0
        self.register = []
        self.processed = []
        self.interestings = interestings
        self.symbol_dict = {}
        self.symbol_str = ''
        

        self._run()
        
    def process_signal(self, signal):
        s = Signal(signal)
        self.processed.append(s)
        self.register += s.X
        self.cycle += s.cycle        
        return 
    
    def _run(self):
        for idx in range(len(self.signals)):
            self.cycle_list.append(self.cycle)
            self.position = idx
            signal = self.signals[idx]
            self.process_signal(signal)
        
    def get_register_value(self, n=None):
        if n:
            return 1+sum(self.register[:n-1])
        else:    
            return 1+sum(self.register[:self.cycle])

    
    def get_signal_strength(self, n):
        rv = self.get_register_value(n)
        if n:
            return rv*n
        else:
            return rv*self.cycle

    
    def get_interesting_signals(self):
        interesting_stregnths = []
        for i in self.interestings:
            interesting_stregnths.append(self.get_signal_strength(n=i))
        return sum(interesting_stregnths)
    
    def assign_symbols(self, consule_range=range(0, 241, 40)):
        for i in range(0, self.cycle+1):
            if i > consule_range.step-1:
                row_break = np.array(consule_range)
                idx = np.where(row_break <= i)[-1][-1]
                mod = row_break[idx]
                lookup = i%mod
            else:
                lookup = i
            self.symbol_str += (get_symbol(lookup, sum(self.register[0:i])))
            self.symbol_dict[i] = {lookup:sum(self.register[0:i])}
        return 

    def print_consule(self, consule_range=range(0, 241, 40)):
            if self.symbol_str == '':
                self.assign_symbols(consule_range)
            for x in consule_range:
                print(self.symbol_str[x-consule_range.step:x])
    
def get_symbol(cycle, sprite_1):
    #get from processed, type of signal so can get cycle passage
    s = sprite_1
    e = s+3
    if cycle in range(s, e):
        return '#'
    else:
        return '.'

#### Day 11 ####
class StolenItem:
    def __init__(self, initial_item):
        self.item_init = initial_item
        self.cur_item = initial_item
        self.item_tracker = [initial_item]
        self.multiples = [self.cur_item]
        self.factor_list = []
        self.remainder_list = []

    def get_new_item(self, new_item, factor=None, operation=None, remainder=[]):
        self.cur_item = new_item   
        self.item_tracker.append(new_item)
        return 
    
    def get_divisibility(self):
        return set(self.factor_list)
            
#keep away class
class KeepAway:
    def __init__(self, monkey_rules, worried=True):
        self.rules = monkey_rules
        self.parsed_rules = []
        self.monkey_list = []
        self.worried = worried
        
        self.parse_rules()
        
        self.monkey_list = [Monkey(r, self.worried) for r in self.parsed_rules]            
        self.divisibility_rules = [x.test for x in self.monkey_list]
        
        
    def parse_rules(self):
        n = len(self.rules)
        for x in list(range(6, n+1, 7)):
            self.parsed_rules.append(self.rules[x-6:x])
            
    def play_around(self, n):
        for x in range(0, n):
            for monkey in self.monkey_list:
                throwing = monkey.take_turn(self.divisibility_rules)
                for k, v in throwing.items():
                    self.monkey_list[k].get_thrown_item(v)
        return
    
    def get_monkey_business(self, n=2):
        sorted_totals =  sorted([x.inspection_count for x in self.monkey_list], 
                                reverse=True)
        return np.multiply(*sorted_totals[0:n])


class Monkey:
    def __init__(self, monkey_init, worried=True):
        self.monkey_init = monkey_init
        self.inspection_count = 0
        self.worried = worried
        
        self.set_attributes()
        
        
    def get_lookup(self, n):
        self.lambda_dict = {'old':lambda x: x, 
                        '*':lambda x, y: x*y, #x[0]*x[1],
                        '+':lambda x, y: x+y, #x[0]+x[1],
                        '-':np.subtract,
                        '/':np.divide
                       }
        try:
            return int(self.operation_list[n])
        except:
            return self.lambda_dict[self.operation_list[n]]
    
    def set_attributes(self):
        self.monkey_init
        self.name = int(self.monkey_init[0].split(' ')[-1].replace(':', ''))
        self.items = [StolenItem(int(x)) for x in self.monkey_init[1].split(': ')[-1].split(', ')]
        self.operation_list = self.monkey_init[2].split('= ')[-1].split(' ')
        self.operation_str = self.operation_list[1]
        self.operation_int = self.get_lookup(2)
        self.operation_function = self.get_lookup(1)
        self.old = self.get_lookup(0)
        self.test = int(self.monkey_init[3].split('by ')[-1])
        self.true = int(self.monkey_init[4].split('monkey ')[-1])
        self.false = int(self.monkey_init[5].split('monkey ')[-1])
        
        self.test_dict = {True:self.true, False:self.false}
        
    def build_operation(self, s_item):
        if type(self.operation_int) == type(lambda x:x):
            return self.operation_function(self.old(s_item), self.operation_int(s_item))
        return self.operation_function(self.old(s_item), self.operation_int)
    
    def take_turn(self, divisibility_rules):
        throw_dict = {} #monkey, items
        if self.items != []:           
            for stole in self.items:

                self.inspection_count += 1
                new = self.build_operation(stole.cur_item)
                
                if self.worried:
                        new = int(np.floor(new / 3))
                        stole.get_new_item(new)
                else: #if self.worried == 0 & test = true // this FAILS - misses appends
                    prod = np.product(divisibility_rules)
                    stole.get_new_item(new%prod)
                
                test = (new % self.test == 0)
               
                
                next_monkey = self.test_dict[test]
                if next_monkey in throw_dict.keys():
                    throw_dict[next_monkey].append(stole)
                else:
                    throw_dict[next_monkey] = [stole]
            self.items = []    
        return throw_dict
    
    def get_thrown_item(self, item):
        self.items += item
        