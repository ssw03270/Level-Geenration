import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
from tqdm import tqdm

def vocab_mapping_function(original_word):
    dictionary = {'frame': 'beam', 'attic': 'roof', 'rafter': 'roof', 'torch': 'lighting', 'flowers': 'plant', 'sunroof': 'window', 'cabinet': 'furniture', 'garden': 'plant', 'lawn': 'plant', 'pipe': 'undetermined', 'woodcolumn': 'column', 'entrance walk way': 'corridor', 'entry path': 'corridor', 'torches': 'lighting', 'counter': 'furniture', 'music box': 'furniture', 'cornerstone': 'foundation', 'courtyard': 'ground', 'shutter': 'shutters', 'skywindow': 'window', 'spiral staircase': 'stairs', 'antenna': 'undetermined', 'couch': 'furniture', 'doorway': 'door', 'wraparound balcony': 'balcony', 'facade': 'balcony', 'column': 'column', 'beams': 'beam', 'side table': 'furniture', 'barred window': 'window', 'mill': 'undetermined', 'endable': 'undetermined', 'sidewalk': 'road', 'bins': 'furniture', 'glass floor': 'floor', 'porch light': 'lighting', 'tree leaves': 'plant', 'balcony rail': 'banister', 'torchlights': 'lighting', 'door': 'door', 'entry': 'door', 'balcony door': 'door', 'crafting table': 'furniture', 'lava': 'ground', 'decor': 'furniture', 'parapet': 'parapet', 'window boxes': 'furniture', 'garden soil': 'ground', 'wooden roof': 'roof', 'lightbulb': 'lighting', 'leaf': 'plant', '2nd floor lights': 'lighting', 'terrain': 'terrain', 'bottom': 'floor', 'driveway': 'road', 'yard': 'ground', 'first floor': 'floor', 'backyard': 'ground', 'ice pillar': 'column', 'porch overhang': 'awning', 'furniture': 'furniture', 'support beams': 'beam', 'concrete column': 'column', 'fan': 'furniture', 'torches (indoors, first floor) ()': 'lighting', 'porch framing': 'beam', 'end table': 'furniture', 'stonewall': 'wall', 'roof': 'roof', 'roof tile': 'roof', 'block island': 'undetermined', 'wall light': 'lighting', 'tile ceiling': 'ceiling', 'ceiling support': 'beam', 'mushroom': 'plant', 'display rack': 'furniture', 'dirt grass': 'ground', 'railing base': 'banister', 'sink': 'furniture', 'window trim': 'window', 'cornerstones': 'column', 'smoke': 'chimney', 'white roof': 'roof', 'fungation': 'undetermined', 'bedframe': 'furniture', 'ledge': 'wall', 'enchanting table': 'furniture', 'wooden post': 'column', 'ladder': 'stairs', 'window surround': 'window', 'pink cube': 'undetermined', 'cobwebs': 'chimney', 'interior vine': 'plant', 'second floor ceiling': 'floor', 'wood beam': 'beam', 'garden barrier': 'fence', 'structure': 'undetermined', 'stool': 'furniture', 'barricade': 'fence', 'bookcases': 'furniture', 'stair support': 'banister', 'entrance patio': 'ground', 'creditable': 'undetermined', 'cobwebs (indoors) ()': 'chimney', 'pumpkins': 'plant', 'glass block': 'window', 'pink railing': 'banister', 'shelfs': 'furniture', 'ice roof': 'roof', 'sofa': 'furniture', 'red wall': 'wall', 'stilt': 'column', 'balcony roof': 'roof', 'pool': 'pool', 'gravel': 'ground', 'bookcase table': 'furniture', 'doors': 'door', 'posts': 'column', 'white cubes': 'undetermined', 'f;ower garden': 'plant', 'roof flowers': 'plant', 'water': 'pool', 'wallpaper': 'wall', 'tower points': 'tower', 'workstation': 'furniture', 'lava basin': 'ground', 'platform frame': 'beam', 'hedge stand': 'plant', 'wooden walls': 'wall', 'balcony': 'balcony', 'bedside table': 'furniture', 'mushrooms': 'plant', 'closet': 'furniture', 'windchime': 'furniture', 'nightshade': 'plant', 'storage chest': 'furniture', 'roof edge': 'roof', 'hay patch 1': 'undetermined', 'workbenches': 'furniture', 'porch lights': 'lighting', 'wood post': 'column', 'window stills': 'window', 'window frame': 'window', 'ceiling detail': 'ceiling', 'floor tile': 'floor', 'roof center': 'roof', 'wood overhang': 'roof', 'kitchen': 'furniture', 'wallpapered wall': 'wall', 'rock': 'ground', 'fence path': 'fence', 'coffee table': 'furniture', 'torches (indoors) ()': 'lighting', 'brewing stations': 'furniture', 'grey boxes': 'undetermined', 'path': 'ground', 'pillar': 'column', 'brewing stand': 'furniture', 'ceiling lights': 'lighting', 'shutters': 'shutters', 'pavement': 'ground', 'pumpkin': 'plant', 'spiderweb': 'chimney', 'flowerbed': 'plant', 'ice column': 'column', 'decorativebox': 'furniture', 'steel support': 'beam', 'foundation support': 'ground', 'bottom of house': 'ground', 'light wires': 'lighting', 'floor lamp': 'lighting', 'jackolantern': 'lighting', 'woodwall': 'wall', 'been': 'undetermined', 'tree branch': 'plant', 'glass pillar': 'column', 'fixtures': 'furniture', 'torch path': 'lighting', 'front arch': 'arch', 'bamboo': 'plant', 'glow stone': 'lighting', 'grate': 'ground', 'bow': 'undetermined', 'angled wall': 'wall', 'inner wall': 'wall', 'border': 'fence', 'ladder entry': 'stairs', 'bathroom sink': 'furniture', 'none': 'undetermined', 'floor \\ awning': 'awning', 'wood support': 'beam', 'flour': 'undetermined', 'entrance': 'door', 'ceiling light': 'lighting', 'sand': 'undetermined', 'glass': 'window', 'pink floor': 'floor', 'awning': 'awning', 'purple block': 'undetermined', 'moat': 'pool', 'pin cube': 'undetermined', 'battlement': 'parapet', 'red floor': 'floor', 'stairwell': 'stairs', 'supports': 'beam', 'decorations': 'furniture', 'roof support column': 'column', 'chimney smoke': 'chimney', 'front lights': 'lighting', 'pathlight': 'lighting', 'pattern wall': 'wall', 'corner decoration': 'furniture', 'brick counter': 'furniture', 'external lighting': 'lighting', 'trashcan': 'furniture', 'glassfence': 'fence', 'wraparound porch': 'balcony', 'terrace railing': 'shutters', 'glass display': 'furniture', 'starboard': 'vehicle', 'support blocks': 'column', 'stones': 'undetermined', 'purplestatue': 'furniture', 'moat base': 'pool', 'top balcony': 'balcony', 'desk': 'furniture', 'fixture': 'furniture', 'rails': 'banister', 'chandelier': 'lighting', 'wooden wall trim': 'wall', 'wooden floor': 'floor', 'glassfloor': 'floor', 'ground': 'ground', 'threshold': 'door', 'jack olantern': 'lighting', 'baseboard': 'wall', 'enchantment table': 'furniture', 'workbench': 'furniture', 'greenery': 'plant', 'mound': 'ground', 'barrier': 'fence', 'portal': 'door', 'washing machine': 'furniture', 'shrub': 'plant', 'purple block island': 'undetermined', 'support foundation': 'ground', 'doormat': 'floor', 'tnt cache': 'undetermined', 'fountain': 'pool', 'glass ceiling': 'ceiling', 'purple pillar': 'column', 'balcony fence': 'fence', 'frontporch': 'balcony', 'stand': 'lighting', 'air vent': 'window', 'grass': 'ground', 'railings': 'banister', 'bush': 'plant', 'web': 'chimney', 'outside stairs': 'stairs', 'floor glass': 'floor', 'firelit': 'lighting', 'steps': 'stairs', 'lounge': 'furniture', 'bedroom door': 'door', 'bookshelves': 'furniture', 'ceiling fan': 'furniture', 'porch roof': 'roof', 'raised platform': 'floor', 'front step': 'stairs', 'bookcase decoration': 'furniture', 'woodchest': 'furniture', 'laddersupport': 'stairs', 'door platform': 'floor', 'middle': 'undetermined', 'table': 'furniture', 'walls': 'wall', 'pans': 'undetermined', 'flame': 'lighting', 'stair': 'stairs', 'dome': 'dome', 'big glow stone': 'lighting', 'entranceway': 'door', 'pedestal': 'column', 'glowstone decor': 'lighting', 'igloo': 'undetermined', 'library': 'undetermined', 'orch': 'undetermined', 'stairway': 'stairs', 'banister': 'banister', 'trees': 'plant', 'jukebox': 'furniture', 'television': 'furniture', 'floor': 'floor', 'bushes': 'plant', 'jigsaw blocks': 'undetermined', 'refrigerator': 'furniture', 'cobweb': 'chimney', 'throne': 'furniture', 'bathtub': 'furniture', 'outer wall': 'wall', 'lamps': 'lighting', 'Wainscott': 'undetermined', 'double beam': 'beam', 'top level blocks': 'undetermined', 'worktable': 'furniture', 'window shutter': 'shutters', 'sign': 'undetermined', 'hearth': 'floor', 'sandpit': 'ground', 'door torches': 'lighting', 'tower': 'tower', 'enclave': 'undetermined', 'landing': 'stairs', 'ladder support': 'stairs', 'lamp': 'lighting', 'headboard': 'furniture', 'corner of building': 'wall', 'torches (indoors, large room) ()': 'lighting', 'crenellation': 'parapet', 'gate': 'gate', 'solarpanel': 'roof', 'furnaces': 'furniture', 'railing': 'banister', 'cupboard': 'furniture', 'first floor ceiling': 'ceiling', 'house support': 'beam', 'interior lighting': 'lighting', 'hedges': 'fence', 'top of building': 'roof', 'soil': 'ground', 'wooden boxes': 'undetermined', 'skywalk': 'corridor', 'container': 'furniture', 'glass railing': 'banister', 'stream': 'pool', 'alchemy': 'furniture', 'front': 'gate', 'wooden stilts': 'column', 'table and chairs': 'furniture', 'window box': 'window', 'blocks': 'undetermined', 'deck': 'furniture', 'hay patch 2': 'undetermined', 'garage door': 'door', 'concrete floors': 'floor', 'support block': 'beam', 'basement': 'floor', 'trellis': 'fence', 'skylights': 'lighting', 'outdoor light': 'lighting', 'step': 'stairs', 'beam': 'beam', 'glass barrier': 'fence', 'support post': 'column', 'wall - doorway awning': 'awning', 'storage': 'undetermined', 'kitchen island': 'undetermined', 'ceiling': 'ceiling', 'painting': 'furniture', 'roof ornament': 'roof', 'wade': 'undetermined', 'greens': 'plant', 'doorframe': 'door', 'top': 'roof', 'wood floor': 'floor', 'display': 'furniture', 'helipad': 'roof', 'chair': 'furniture', 'second story floor': 'floor', 'piller': 'column', 'leaves': 'plant', 'chimney wall': 'wall', 'candle': 'lighting', 'windows': 'window', 'was': 'undetermined', 'umbrella': 'undetermined', 'flooring': 'floor', 'stone roof': 'roof', 'doorsteps': 'stairs', 'lighting': 'lighting', 'windowframe': 'window', 'bricks': 'wall', 'pink blocks': 'undetermined', 'houseplant': 'plant', 'pink cubes': 'undetermined', 'archway': 'arch', 'marble': 'undetermined', 'cabinetry': 'furniture', 'trim': 'wall', 'floor support': 'floor', 'interior walls': 'wall', 'fireplace': 'chimney', 'canopy': 'awning', 'oven': 'furniture', 'cornice': 'parapet', 'wall frame': 'wall', 'wall': 'wall', 'loveseat': 'furniture', 'wall vines': 'plant', 'staircase': 'stairs', 'roof level': 'roof', 'shingles': 'roof', 'bath': 'furniture', 'stereo': 'furniture', 'cauldron': 'furniture', 'shrubs': 'plant', 'dining table chairs': 'furniture', 'fence': 'fence', 'enchanter': 'furniture', 'porch floor': 'floor', 'seats': 'furniture', 'floor \\ ceiling': 'floor', 'iron': 'undetermined', 'statue': 'furniture', 'shrubbery': 'plant', 'bookshelf': 'furniture', 'structural support': 'beam', 'roof \\ floor': 'roof', 'glass roof': 'roof', 'torchlight': 'lighting', 'stern': 'undetermined', 'giant torch': 'lighting', 'panel': 'roof', 'porch wall': 'wall', 'patio railing': 'banister', 'post support': 'column', 'patio': 'ground', 'wood': 'plant', 'rail': 'banister', 'crafting bench': 'furniture', 'sconce': 'lighting', 'pond': 'pool', 'interior nook': 'furniture', 'rooftop': 'roof', 'front entrance': 'door', 'stair roof': 'stairs', 'tower lighting': 'lighting', 'cross support': 'beam', 'post': 'column', 'gates': 'gate', 'ches': 'undetermined', 'pathway': 'road', 'candles': 'lighting', 'front steps': 'stairs', 'internal lighting': 'lighting', 'dirt patch 2': 'undetermined', 'tree branches': 'plant', 'bookcase': 'furniture', 'floor tiles': 'floor', 'chest': 'furniture', 'light': 'lighting', 'arch': 'arch', 'stone foundation': 'parapet', 'dirt path': 'road', 'swimming pool': 'pool', 'stairs': 'stairs', 'skylight': 'window', 'walldecor': 'wall', 'gable': 'roof', 'ground support': 'ground', 'bridge': 'undetermined', 'shower': 'furniture', 'stone': 'undetermined', 'front door': 'door', 'river': 'pool', 'plant': 'plant', 'wooden lattice': 'fence', 'doordecor': 'door', 'light fixture': 'lighting', 'paperd wall': 'wall', 'chiming': 'undetermined', 'walkway': 'road', 'roof tiles': 'roof', 'picture window': 'window', 'dirt': 'ground', 'brickwall': 'wall', 'support column': 'column', 'plants': 'plant', 'boxes': 'furniture', 'pot': 'furniture', 'seating': 'furniture', 'brick roof': 'roof', 'stone archway': 'arch', 'bin': 'furniture', 'g;ow stone': 'lighting', 'porch fence': 'fence', 'torches (outdoors) ()': 'lighting', 'roof window': 'window', 'glass wall': 'wall', 'half-wall': 'wall', 'curb': 'road', 'garage': 'garage', 'aquarium': 'pool', 'nothing': 'undetermined', 'blue blocks': 'undetermined', 'building foundation': 'ground', 'brick column': 'column', 'retaining wall': 'wall', 'staircase support': 'stairs', 'stairssupport': 'stairs', 'ivy': 'plant', 'rocks': 'undetermined', 'utilities': 'undetermined', 'cover': 'roof', 'stone platform': 'ground', 'support beam': 'beam', 'forge': 'undetermined', 'pink boxes': 'undetermined', 'largebush': 'plant', 'shaft': 'column', 'moss': 'plant', 'rug': 'floor', 'nightstands': 'lighting', 'countertop': 'furniture', 'chimney': 'chimney', 'concrete': 'wall', 'block': 'undetermined', 'inner frame': 'beam', 'roof trim': 'roof', 'puddle': 'undetermined', 'kitchen floor': 'floor', 'gutter': 'roof', 'corner lamp': 'lighting', 'facade wall': 'wall', 'purple pilalr': 'column', 'tree trunk': 'plant', 'glass rail': 'banister', 'wall vine': 'plant', 'moat water': 'pool', 'box': 'furniture', 'stone base': 'ground', 'display case': 'furniture', 'column support': 'column', 'glasswall': 'wall', 'flower balcony': 'balcony', 'irrigation': 'ground', 'entry arch': 'arch', 'potions': 'furniture', 'hanging plant': 'plant', 'torches (medium room) ()': 'lighting', 'front yard': 'ground', 'second floor': 'floor', 'glass window': 'window', 'paper wall': 'wall', 'port': 'column', 'stone walkway': 'road', 'entryway': 'door', 'brown box': 'furniture', 'lattice': 'window', 'house stilts': 'column', 'waterfall': 'undetermined', 'fire': 'lighting', 'furnace': 'furniture', 'kitchen cabinets': 'furniture', 'roof beam': 'beam', 'bench': 'furnace', 'window ledge': 'window', 'hangingplant': 'plant', 'steps to entrance': 'stairs', 'hull': 'undetermined', 'overhang': 'roof', 'Galway': 'undetermined', 'eaves': 'roof', 'entry steps': 'stairs', 'torches (small room) ()': 'lighting', 'roofs': 'roof', 'base': 'ground', 'rack': 'furniture', 'concrete wall': 'wall', 'garden wall': 'wall', 'inside wall': 'wall', 'roof support': 'roof', 'planter box': 'plant', 'subfloor': 'floor', 'all': 'undetermined', 'wood archway': 'arch', 'exterior lighting': 'lighting', 'tile floor': 'floor', 'tables': 'furniture', 'dispenser': 'furniture', 'inside lights': 'lighting', 'stove': 'furniture', 'stone floor': 'floor', 'base support': 'ground', 'interior': 'furniture', 'stoop': 'stairs', 'entertainment center': 'furniture', 'hall floor': 'floor', 'shelf': 'furniture', 'railing post': 'banister', 'bar': 'furniture', 'windowsill': 'window', 'doorstep': 'stairs', 'axle': 'furniture', 'purple box': 'undetermined', 'trunk': 'plant', 'fern': 'plant', 'metalbox': 'undetermined', 'under window': 'window', 'stone pillar': 'column', 'indoor plants': 'plant', 'Wal': 'wall', 'verdant': 'plant', 'ladders': 'stairs', 'flower': 'plant', 'beds': 'furniture', 'round': 'undetermined', 'ground floor': 'floor', 'interior wall': 'wall', 'chairs': 'furniture', 'vines': 'plant', 'crate': 'furniture', 'torches (indoors, second floor) ()': 'lighting', 'roofgarden': 'balcony', 'brewer': 'undetermined', 'platform': 'ground', 'dirt patch 1': 'undetermined', 'chests': 'furniture', 'cement foundation': 'ground', 'support': 'beam', 'dresser': 'furniture', 'tree': 'plant', 'hedge': 'beam', 'flower garden': 'plant', 'headrest': 'undetermined', 'red box': 'undetermined', 'bedroom floor': 'floor', 'shelves': 'furniture', 'seat': 'furniture', 'torches (indoor) ()': 'lighting', 'outcroppings': 'ground', 'carpet': 'floor', 'tv stand': 'furniture', 'brewers': 'undetermined', 'platform to door': 'floor', 'cross': 'undetermined', 'blade': 'undetermined', 'random': 'undetermined', 'marquee': 'awning', 'lantern': 'lighting', 'metalchest': 'furniture', 'glass tower': 'tower', 'floor mat': 'floor', 'tile': 'wall', 'nightstand': 'lighting', 'basement floor': 'floor', 'pillars': 'column', 'wood column': 'column', 'bottom turret': 'tower', 'living room floor': 'floor', 'pink roof': 'roof', 'lights': 'lighting', 'layer': 'undetermined', 'cushions': 'furniture', 'porch railing': 'banister', 'bars': 'furniture', 'decoration': 'furniture', 'timetable': 'furniture', 'bunker': 'ground', 'columns': 'column', 'planter': 'plant', 'roof tower': 'tower', 'hatch': 'door', 'ice wall': 'wall', 'pew': 'furniture', 'wind chimes': 'furniture', 'upper floor': 'floor', 'porch': 'balcony', 'fencing': 'fence', 'cactus': 'plant', 'outside lights': 'lighting', 'deck railing': 'banister', 'torches (pathway) ()': 'lighting', 'blue': 'undetermined', 'stone columns': 'column', 'brick border': 'wall', 'wooden posts': 'column', 'cap': 'undetermined', 'support wall': 'wall', 'corner': 'wall', 'window': 'window', 'garden plot': 'plant', 'island': 'undetermined', 'cushion': 'furniture', 'side door': 'door', 'foundation': 'ground', 'shelving': 'furniture', 'farm': 'fence', 'battlements': 'parapet', 'decksupport': 'column', 'safe': 'furniture', 'ferns': 'plant', 'porch support': 'column', 'woodsupport': 'column', 'books': 'furniture', 'jack-o-lantern': 'lighting', 'bed': 'furniture'}
    return dictionary[original_word]

class CraftAssistDataset(Dataset):
    def __init__(self, data_type='train'):
        super(CraftAssistDataset, self).__init__()

        self.data_type = data_type

        if data_type == 'train':
            self.file_path = '../../../../datasets/training_data2.pkl'
        else:
            self.file_path = '../../../../datasets/training_data2.pkl'

        self.position_sequences = []
        self.block_id_sequences = []
        self.block_category_sequences = []
        self.next_id_sequences = []
        self.next_category_sequences = []
        self.pad_mask_sequences = []
        self.terrain_mask_sequences = []

        with open(self.file_path, 'rb') as f:
            data = pickle.load(f)
            input_sequences = data['input_sequences']

        block_category_values = set()
        for input_sequence in input_sequences:
            for input_data in input_sequence:
                category = vocab_mapping_function(input_data[2])
                block_category_values.add(category)

        # 집합을 리스트로 변환하고 정렬
        self.sorted_block_category_values = sorted(list(block_category_values))

        # 정렬된 리스트를 사용하여 인덱스 매핑 생성
        block_category_to_index = {value: idx + 3 for idx, value in enumerate(self.sorted_block_category_values)}
        for idx, value in enumerate(self.sorted_block_category_values):
            print(value, idx+3)

        for input_sequence in input_sequences:
            position_sequence = []
            block_id_sequence = []
            block_category_sequence = []
            terrain_mask_sequence = []

            data_length = len(input_sequence)
            if data_length > 2040:
                continue

            for input_data in input_sequence:
                position_sequence.append(input_data[0])
                block_id_sequence.append(input_data[1])
                block_category_sequence.append(vocab_mapping_function(input_data[2]))
                terrain_mask_sequence.append(input_data[2] != 'terrain')

            pad_length = 2048 - 2 - data_length
            position_sequence = [[0, 0, 0]] + position_sequence + [[0, 0, 0]] + [[0, 0, 0]] * pad_length

            next_id_sequence = block_id_sequence[1:]
            block_id_sequence = [0] + block_id_sequence + [0] + [0] * pad_length
            next_id_sequence = next_id_sequence + [0] + [0] * (pad_length + 2)

            block_category_sequence = [block_category_to_index[value] for value in block_category_sequence]
            next_category_sequence = block_category_sequence[1:]
            block_category_sequence = [0] + block_category_sequence + [1] + [2] * pad_length
            next_category_sequence = next_category_sequence + [1] + [2] * (pad_length + 2)

            pad_mask_sequence = [1] * (2048 - pad_length) + [0] * pad_length
            terrain_mask_sequence = [False] + terrain_mask_sequence + [False] + [False] * pad_length

            self.position_sequences.append(position_sequence)
            self.block_id_sequences.append(block_id_sequence)
            self.block_category_sequences.append(block_category_sequence)
            self.next_id_sequences.append(next_id_sequence)
            self.next_category_sequences.append(next_category_sequence)
            self.pad_mask_sequences.append(pad_mask_sequence)
            self.terrain_mask_sequences.append(terrain_mask_sequence)

        self.min_val = None
        self.max_val = None
        self.position_sequences = self.min_max_scaling(np.array(self.position_sequences))

        self.data_length = len(self.position_sequences)
        print(f'{data_type}: {self.data_length}')

    def __getitem__(self, idx):
        position_sequence = self.position_sequences[idx]
        block_id_sequence = self.block_id_sequences[idx]
        block_category_sequence = self.block_category_sequences[idx]
        next_id_sequence = self.next_id_sequences[idx]
        next_category_sequence = self.next_category_sequences[idx]
        pad_mask_sequence = self.pad_mask_sequences[idx]
        terrain_mask_sequence = self.terrain_mask_sequences[idx]

        position_sequence = torch.tensor(position_sequence, dtype=torch.float32)
        block_id_sequence = torch.tensor(block_id_sequence, dtype=torch.long)
        block_category_sequence = torch.tensor(block_category_sequence, dtype=torch.long)
        next_id_sequence = torch.tensor(next_id_sequence, dtype=torch.long)
        next_category_sequence = torch.tensor(next_category_sequence, dtype=torch.long)
        pad_mask_sequence = torch.tensor(pad_mask_sequence, dtype=torch.bool)
        terrain_mask_sequence = torch.tensor(terrain_mask_sequence, dtype=torch.bool)

        return position_sequence, block_id_sequence, block_category_sequence, \
            next_id_sequence, next_category_sequence, pad_mask_sequence, terrain_mask_sequence

    def __len__(self):
        return self.data_length

    def min_max_scaling(self, position_sequence):
        if self.min_val is None:
            self.min_val = np.min(position_sequence, axis=(0, 1))
            self.max_val = np.max(position_sequence, axis=(0, 1))

        scaled_position_sequence = (position_sequence - self.min_val) / (self.max_val - self.min_val)
        scaled_position_sequence = scaled_position_sequence * 2 - 1
        return scaled_position_sequence

    def restore_min_max_scaling(self, scaled_position_sequence):
        # [0, 1] 범위로 되돌리기
        original_scale_sequence = (scaled_position_sequence + 1) / 2
        # 원래의 범위로 복원
        restored_position_sequence = original_scale_sequence * (self.max_val - self.min_val) + self.min_val
        return restored_position_sequence