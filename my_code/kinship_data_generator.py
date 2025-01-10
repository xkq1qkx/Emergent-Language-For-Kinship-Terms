import random
from graphviz import Digraph
from collections import deque, defaultdict
from copy import deepcopy
import json
import numpy as np

MARRY_PROB = 0.5
MAX_CHILDREN = 2
MAX_GENERATION = 3

relationship_class_dict = {}
relationship_class_id = 0
data_id = 0
tree_id = 0

def bfs_shortest_path_with_types(edges, x, y):
    adj_list = defaultdict(list)
    for u, v, edge_type in edges:
        adj_list[u].append((v, edge_type))

    queue = deque([(x, [])])
    visited = set([x])

    while queue:
        current, path = queue.popleft()

        if current == y:
            return path

        for neighbor, edge_type in adj_list[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [edge_type]))

    return None

class Person:
    def __init__(self, id, gender, generation):
        self.id = id                
        self.gender = gender        
        self.children = []
        self.generation = generation
        self.couple = None

def opposite_gender(g):
    if g == "M":
        return "F"
    else:
        return "M"

class FamilyTree:
    def __init__(self, max_generations=3, max_children=3, begin_id = 0):
        self.root = None
        self.max_generations = max_generations
        self.max_children = max_children
        self.person_id = begin_id
        self.nodes = []
        self.edges = []

    def generate_random_family(self):
        self.root = self._generate_person(generation=0)
        return len(self.nodes)

    def _generate_person(self, generation):
        global MARRY_PROB
        if generation >= self.max_generations:
            return None

        gender = random.choice(["M", "F"])
        person = Person(id=self.person_id, gender=gender, generation=generation)
        self.nodes.append(person)
        self.person_id += 1
        if random.random() <= MARRY_PROB or generation == 0:
            couple = Person(id=self.person_id, gender=opposite_gender(gender), generation=generation)
            self.nodes.append(couple)
            self.person_id += 1
            person.couple = couple
            couple.couple = person
            
            if generation == 0:
                num_children = random.randint(1, self.max_children)
            else:
                num_children = random.randint(0, self.max_children)

            for _ in range(num_children):
                child = self._generate_person(generation + 1)
                if child:
                    person.children.append(child)
                    self.edges.append((person, child))
                    couple.children.append(child)
                    self.edges.append((couple, child))

        return person
    
    def add_children(self, person, couple):
        num_children = random.randint(1, self.max_children)
        generation = max(person.generation, couple.generation)

        for _ in range(num_children):
            child = self._generate_person(generation + 1)
            if child:
                person.children.append(child)
                self.edges.append((person, child))
                couple.children.append(child)
                self.edges.append((couple, child))
                # print(f'{person.id} {couple.id} born {child.id}')

    def visualize_family_tree(self, filename="family_tree"):
        dot = Digraph(comment="Family Tree")
        
        for person in self.nodes:
            color = "lightblue" if person.gender == "M" else "pink"
            dot.node(str(person.id), f"Person {person.id} ({person.gender})", style="filled", fillcolor=color)
    

        for person in self.nodes:
            if person.couple:
                with dot.subgraph() as s:
                    s.attr(rank="same")
                    s.edge(str(person.id), str(person.couple.id), style="dotted")  # Invisible edge to align

        for edge in self.edges:
            dot.edge(str(edge[0].id), str(edge[1].id))

        dot.format = "png"
        dot.render(filename, cleanup=True)
        # print(f"Family tree saved as {filename}.png")

    def get_all_data(self, extra=''):
        # Encoder 输入是 图emb, caller_id, listener_id；Decoder 输入是 图emb, listener_id'', 输出是标签
        # 图，callerid, listenerid, test_listener, labels
        nodes = deepcopy(self.nodes)
        pairs = []
        for node1 in nodes:
            for node2 in nodes:
                if node1 == node2:
                    continue
                pairs.append((node1.id, node2.id))
        edges = self.build_graph()
        data = []
        global relationship_class_dict
        global relationship_class_id
        global data_id
        global tree_id
        for pair1 in pairs:
            caller_id, listener_id = pair1[0], pair1[1]
            generations, genders, ids, edges_feature = self.construct_graph_input_data()
            
            if caller_id == listener_id:
                continue
            
            caller_listener_path = bfs_shortest_path_with_types(edges, caller_id, listener_id) 
            # print(caller_listener_path)
            if extra != '':
                flag = False
                for x in caller_listener_path:
                    if x not in [0, 2, 5, 8]: # older -> younger, father/mother -> son/daughter edges
                        flag = True
                if flag is True:
                    continue
            # print(caller_listener_path) 
            k = ' '.join([str(pnode) for pnode in caller_listener_path])
            if k not in relationship_class_dict:
                relationship_class_dict[k] = relationship_class_id
                relationship_class_id += 1
            
            rel_class = relationship_class_dict[k]

            for test_listener in ids:
                labels = []
                for test_caller in ids:
                    if (caller_listener_path
                        == bfs_shortest_path_with_types(edges, test_caller, test_listener)):
                        labels.append(1)
                    else:
                        labels.append(0)

                if not np.array(labels).any(): # if labels are all zeros it is not wanted
                    continue

                data_item = {
                    "data_id": data_id,
                    "tree_id": tree_id,
                    "generations": generations,
                    "genders": genders,
                    "ids": ids,
                    "edges_feature": edges_feature,
                    "caller_id": caller_id,
                    "listener_id": listener_id,
                    "test_listener": test_listener,
                    "labels": labels,
                    "rel_class": rel_class
                }
                data_id += 1
                data.append(data_item)
        tree_id += 1
        return data

    def construct_graph_input_data(self):
        generations = [n.generation + 1 for n in self.nodes]
        genders = []
        for n in self.nodes:
            if n.gender == "M":
                genders.append(1)
            else:
                genders.append(2)
        ids = [n.id for n in self.nodes]

        kinship_graph=[]
        for person1, person2 in self.edges:
            if person1.gender=='M':
                if person2.gender=='M':
                    kinship_graph.append((person1.id,person2.id,0))
                    continue
                if person2.gender=='F':
                    kinship_graph.append((person1.id,person2.id,1))
                    continue
            if person1.gender=='F':
                if person2.gender=='M':
                    kinship_graph.append((person1.id,person2.id,2))
                    continue
                if person2.gender=='F':
                    kinship_graph.append((person1.id,person2.id,3))
                    continue
        for person1 in self.nodes:
            for person2 in self.nodes:
                if person1.id==person2.id:
                    continue
                if person1.couple==person2:
                    if person1.gender=='M':
                        kinship_graph.append((person1.id,person2.id,4))
                        continue
                    if person1.gender=='F':
                        kinship_graph.append((person1.id,person2.id,5))
                        continue
        return generations, genders, ids, kinship_graph

    def build_graph(self):
        kinship_graph=[]
        for person1 in self.nodes:
            for person2 in self.nodes:
                if person1.id==person2.id:
                    continue
                if person1.couple==person2:
                    if person1.gender=='M':
                        kinship_graph.append((person1.id,person2.id,3))
                        continue
                    if person1.gender=='F':
                        kinship_graph.append((person1.id,person2.id,6))
                        continue
        for person1, person2 in self.edges:
            if person1.gender=='M':
                if person2.gender=='M':
                    kinship_graph.append((person1.id,person2.id,0))
                    kinship_graph.append((person2.id,person1.id,1))
                    continue
                if person2.gender=='F':
                    kinship_graph.append((person1.id,person2.id,2))
                    kinship_graph.append((person2.id,person1.id,7))
                    continue
            if person1.gender=='F':
                if person2.gender=='M':
                    kinship_graph.append((person1.id,person2.id,5))
                    kinship_graph.append((person2.id,person1.id,4))
                    continue
                if person2.gender=='F':
                    kinship_graph.append((person1.id,person2.id,8))
                    kinship_graph.append((person2.id,person1.id,9))
                    continue
        return kinship_graph

def generate_data(pre="train", extra=''):
    global tree_id
    global MARRY_PROB
    max_node_num = 0
    if pre == "train":
        if extra == '':
            episode_num = 1000
            MAX_GENERATION = 3
        else:
            episode_num = 2000
            MARRY_PROB = 0.5
            MAX_GENERATION = 6
    if pre == "test":
        if extra == '':
            episode_num = 50
            MAX_GENERATION = 4
        else:
            episode_num = 100
            MAX_GENERATION = 8
    all_data = []
    for episode in range(episode_num):
        
        assert episode == tree_id
        family_tree1 = FamilyTree(max_generations=MAX_GENERATION, max_children=MAX_CHILDREN, begin_id=0)
        tree1_num = family_tree1.generate_random_family()
        family_tree1.visualize_family_tree("family_tree1")

        family_tree2 = FamilyTree(max_generations=MAX_GENERATION, max_children=MAX_CHILDREN, begin_id=tree1_num)
        tree2_num = family_tree2.generate_random_family()
        family_tree2.visualize_family_tree("family_tree2")

        if random.random() <= MARRY_PROB:
            flag = False
            for node_c1 in family_tree1.nodes:
                for node_c2 in family_tree2.nodes:
                    if node_c1.couple==None and node_c2.couple==None and node_c1.generation == node_c2.generation:
                        if node_c1.gender == node_c2.gender:
                            node_c2.gender = opposite_gender(node_c1.gender)
                        node_c1.couple=node_c2
                        node_c2.couple=node_c1
                        # print(node_c1.id, node_c2.id)
                        family_tree1.nodes.extend(family_tree2.nodes)
                        family_tree1.edges.extend(family_tree2.edges)
                        family_tree1.person_id = len(family_tree1.nodes)
                        flag = True
                        family_tree1.add_children(node_c1, node_c2)

                if flag:
                    break

        family_tree1.visualize_family_tree(f"family_tree_{pre}{extra}_{tree_id}")
        data = family_tree1.get_all_data(extra=extra)
        max_node_num = max(max_node_num, len(family_tree1.nodes))
        all_data.extend(data)
    print('max node', max_node_num)
    print('rel_class_num', relationship_class_dict)
    with open(f'data_{pre}{extra}.json', 'w') as f:
        json.dump(all_data, f)
#edge_type: gender1--->gender2  generation_delta
#0: M M 1
#1: M M -1
#2: M F 1
#3: M F 0
#4: M F -1
#5: F M 1
#6: F M 0
#7: F M -1
#8: F F 1
#9: F F -1
# graph_t1=family_tree1.build_graph()
# print(graph_t1)
generate_data(pre="train", extra="_only_mt")
# generate_data(pre="train", extra="")
# print(relationship_class_dict)
tree_id = 0
generate_data(pre="test", extra="_only_mt")
# generate_data(pre="test", extra="")
# training max node: 20
# testing max node: 21