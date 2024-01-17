import plotly.graph_objects as go
import pickle
import numpy as np

def create_cube(center, size=1):
    # 정육면체의 중심에서 꼭지점으로의 방향 벡터
    dirs = np.array([[1, 1, -1],
                     [1, -1, -1],
                     [-1, -1, -1],
                     [-1, 1, -1],
                     [1, 1, 1],
                     [1, -1, 1],
                     [-1, -1, 1],
                     [-1, 1, 1]])
    # 꼭지점 계산
    return center + size * 0.5 * dirs

def create_mesh(vertices_list, category, color):
    faces = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [0, 1, 2, 3], [4, 5, 6, 7]])
    quad_to_tri = lambda quad: [(quad[0], quad[1], quad[2]), (quad[2], quad[3], quad[0])]
    vertices = []
    i_faces = []
    for i, cube_center in enumerate(vertices_list):
        cube_vertices = create_cube(cube_center)
        vertices.append(cube_vertices)
        for quad in faces:
            i_faces.extend(quad_to_tri(quad + 8 * i))

    vertices = np.concatenate(vertices, axis=0)
    i_faces = np.array(i_faces)
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    i, j, k = i_faces[:, 0], i_faces[:, 1], i_faces[:, 2]

    mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=1, name=category, showlegend=True)
    return mesh

with open('../../../datasets/training_data.pkl', 'rb') as f:
    data = pickle.load(f)

dictionary = {'frame': 'chimney', 'attic': 'lighting', 'rafter': 'stairs', 'torch': 'window',
              'flowers': 'fence', 'sunroof': 'stairs', 'cabinet': 'ground', 'garden': 'wall', 'lawn': 'fence',
              'pipe': 'dome', 'woodcolumn': 'shutters', 'entrance walk way': 'ground', 'entry path': 'vehicle',
              'torches': 'beam', 'counter': 'lighting', 'music box': 'wall', 'cornerstone': 'garage',
              'courtyard': 'wall', 'shutter': 'dome', 'skywindow': 'ground', 'spiral staircase': 'shutters',
              'antenna': 'parapet', 'couch': 'stairs', 'doorway': 'lighting', 'wraparound balcony': 'shutters',
              'facade': 'parapet', 'column': 'column', 'beams': 'beam', 'side table': 'stairs',
              'barred window': 'stairs', 'mill': 'roof', 'endable': 'tower', 'sidewalk': 'ground', 'bins': 'shutters',
              'glass floor': 'shutters', 'porch light': 'ground', 'tree leaves': 'ground', 'balcony rail': 'shutters',
              'torchlights': 'wall', 'door': 'door', 'entry': 'roof', 'balcony door': 'ground',
              'crafting table': 'awning', 'lava': 'corridor', 'decor': 'beam', 'parapet': 'parapet',
              'window boxes': 'stairs', 'garden soil': 'shutters', 'wooden roof': 'shutters', 'lightbulb': 'awning',
              'leaf': 'dome', '2nd floor lights': 'corridor', 'terrain': 'terrain', 'bottom': 'balcony',
              'driveway': 'door', 'yard': 'tower', 'first floor': 'wall', 'backyard': 'roof', 'ice pillar': 'tower',
              'porch overhang': 'shutters', 'furniture': 'furniture', 'support beams': 'stairs',
              'concrete column': 'stairs', 'fan': 'lighting', 'torches (indoors, first floor) ()': 'shutters',
              'porch framing': 'shutters', 'end table': 'wall', 'stonewall': 'beam', 'roof': 'roof',
              'roof tile': 'shutters', 'block island': 'stairs', 'wall light': 'corridor', 'tile ceiling': 'awning',
              'ceiling support': 'shutters', 'mushroom': 'lighting', 'display rack': 'shutters', 'dirt grass': 'awning',
              'railing base': 'stairs', 'sink': 'roof', 'window trim': 'shutters', 'cornerstones': 'ground',
              'smoke': 'lighting', 'white roof': 'furniture', 'fungation': 'fence', 'bedframe': 'awning',
              'ledge': 'chimney', 'enchanting table': 'banister', 'wooden post': 'stairs', 'ladder': 'lighting',
              'window surround': 'stairs', 'pink cube': 'plant', 'cobwebs': 'shutters', 'interior vine': 'wall',
              'second floor ceiling': 'stairs', 'wood beam': 'stairs', 'garden barrier': 'balcony', 'structure': 'pool',
              'stool': 'beam', 'barricade': 'wall', 'bookcases': 'shutters', 'stair support': 'stairs',
              'entrance patio': 'wall', 'creditable': 'furniture', 'cobwebs (indoors) ()': 'shutters',
              'pumpkins': 'shutters', 'glass block': 'wall', 'pink railing': 'balcony', 'shelfs': 'stairs',
              'ice roof': 'furniture', 'sofa': 'vehicle', 'red wall': 'wall', 'stilt': 'corridor',
              'balcony roof': 'shutters', 'pool': 'pool', 'gravel': 'ground', 'bookcase table': 'awning',
              'doors': 'roof', 'posts': 'column', 'white cubes': 'shutters', 'f;ower garden': 'shutters',
              'roof flowers': 'shutters', 'water': 'beam', 'wallpaper': 'shutters', 'tower points': 'ground',
              'workstation': 'stairs', 'lava basin': 'stairs', 'platform frame': 'ground', 'hedge stand': 'shutters',
              'wooden walls': 'shutters', 'balcony': 'balcony', 'bedside table': 'awning', 'mushrooms': 'fence',
              'closet': 'lighting', 'windchime': 'roof', 'nightshade': 'ground', 'storage chest': 'stairs',
              'roof edge': 'ground', 'hay patch 1': 'vehicle', 'workbenches': 'furniture', 'porch lights': 'stairs',
              'wood post': 'lighting', 'window stills': 'vehicle', 'window frame': 'ground', 'ceiling detail': 'stairs',
              'floor tile': 'shutters', 'roof center': 'wall', 'wood overhang': 'stairs', 'kitchen': 'wall',
              'wallpapered wall': 'awning', 'rock': 'beam', 'fence path': 'shutters', 'coffee table': 'shutters',
              'torches (indoors) ()': 'ground', 'brewing stations': 'furniture', 'grey boxes': 'banister',
              'path': 'tower', 'pillar': 'lighting', 'brewing stand': 'vehicle', 'ceiling lights': 'shutters',
              'shutters': 'shutters', 'pavement': 'corridor', 'pumpkin': 'dome', 'spiderweb': 'wall',
              'flowerbed': 'shutters', 'ice column': 'tower', 'decorativebox': 'roof', 'steel support': 'ground',
              'foundation support': 'wall', 'bottom of house': 'ground', 'light wires': 'stairs',
              'floor lamp': 'stairs', 'jackolantern': 'dormer', 'woodwall': 'roof', 'been': 'vehicle',
              'tree branch': 'wall', 'glass pillar': 'furniture', 'fixtures': 'wall', 'torch path': 'roof',
              'front arch': 'stairs', 'bamboo': 'dome', 'glow stone': 'pool', 'grate': 'roof', 'bow': 'door',
              'angled wall': 'wall', 'inner wall': 'wall', 'border': 'shutters', 'ladder entry': 'floor',
              'bathroom sink': 'shutters', 'none': 'lighting', 'floor \\ awning': 'stairs', 'wood support': 'dormer',
              'flour': 'stairs', 'entrance': 'tower', 'ceiling light': 'wall', 'sand': 'wall', 'glass': 'beam',
              'pink floor': 'ground', 'awning': 'awning', 'purple block': 'pool', 'moat': 'ceiling', 'pin cube': 'wall',
              'battlement': 'shutters', 'red floor': 'stairs', 'stairwell': 'tower', 'supports': 'wall',
              'decorations': 'corridor', 'roof support column': 'shutters', 'chimney smoke': 'stairs',
              'front lights': 'corridor', 'pathlight': 'ceiling', 'pattern wall': 'ceiling',
              'corner decoration': 'stairs', 'brick counter': 'stairs', 'external lighting': 'shutters',
              'trashcan': 'stairs', 'glassfence': 'plant', 'wraparound porch': 'awning', 'terrace railing': 'shutters',
              'glass display': 'shutters', 'starboard': 'parapet', 'support blocks': 'stairs', 'stones': 'furniture',
              'purplestatue': 'dormer', 'moat base': 'wall', 'top balcony': 'stairs', 'desk': 'furniture',
              'fixture': 'lighting', 'rails': 'beam', 'chandelier': 'awning', 'wooden wall trim': 'shutters',
              'wooden floor': 'shutters', 'glassfloor': 'stairs', 'ground': 'ground', 'threshold': 'wall',
              'jack olantern': 'gate', 'baseboard': 'stairs', 'enchantment table': 'wall', 'workbench': 'wall',
              'greenery': 'shutters', 'mound': 'lighting', 'barrier': 'lighting', 'portal': 'balcony',
              'washing machine': 'stairs', 'shrub': 'banister', 'purple block island': 'dormer',
              'support foundation': 'fence', 'doormat': 'stairs', 'tnt cache': 'shutters', 'fountain': 'shutters',
              'glass ceiling': 'shutters', 'purple pillar': 'fence', 'balcony fence': 'shutters', 'frontporch': 'pool',
              'stand': 'plant', 'air vent': 'shutters', 'grass': 'shutters', 'railings': 'shutters', 'bush': 'roof',
              'web': 'parapet', 'outside stairs': 'ground', 'floor glass': 'ground', 'firelit': 'ceiling',
              'steps': 'stairs', 'lounge': 'dome', 'bedroom door': 'stairs', 'bookshelves': 'awning',
              'ceiling fan': 'stairs', 'porch roof': 'shutters', 'raised platform': 'awning', 'front step': 'shutters',
              'bookcase decoration': 'shutters', 'woodchest': 'pool', 'laddersupport': 'stairs',
              'door platform': 'ground', 'middle': 'ceiling', 'table': 'pool', 'walls': 'corridor', 'pans': 'stairs',
              'flame': 'dome', 'stair': 'lighting', 'dome': 'dome', 'big glow stone': 'door', 'entranceway': 'stairs',
              'pedestal': 'chimney', 'glowstone decor': 'vehicle', 'igloo': 'road', 'library': 'dormer',
              'orch': 'column', 'stairway': 'lighting', 'banister': 'banister', 'trees': 'wall', 'jukebox': 'banister',
              'television': 'shutters', 'floor': 'floor', 'bushes': 'ground', 'jigsaw blocks': 'shutters',
              'refrigerator': 'furniture', 'cobweb': 'wall', 'throne': 'garage', 'bathtub': 'stairs',
              'outer wall': 'ground', 'lamps': 'window', 'Wainscott': 'banister', 'double beam': 'stairs',
              'top level blocks': 'pool', 'worktable': 'gate', 'window shutter': 'shutters', 'sign': 'vehicle',
              'hearth': 'chimney', 'sandpit': 'stairs', 'door torches': 'corridor', 'tower': 'tower',
              'enclave': 'lighting', 'landing': 'ceiling', 'ladder support': 'stairs', 'lamp': 'dome',
              'headboard': 'stairs', 'corner of building': 'shutters', 'torches (indoors, large room) ()': 'shutters',
              'crenellation': 'furniture', 'gate': 'gate', 'solarpanel': 'buttress', 'furnaces': 'shutters',
              'railing': 'chimney', 'cupboard': 'pool', 'first floor ceiling': 'shutters', 'house support': 'wall',
              'interior lighting': 'shutters', 'hedges': 'dome', 'top of building': 'stairs', 'soil': 'shutters',
              'wooden boxes': 'stairs', 'skywalk': 'awning', 'container': 'garage', 'glass railing': 'shutters',
              'stream': 'vehicle', 'alchemy': 'shutters', 'front': 'beam', 'wooden stilts': 'shutters',
              'table and chairs': 'awning', 'window box': 'wall', 'blocks': 'banister', 'deck': 'window',
              'hay patch 2': 'vehicle', 'garage door': 'stairs', 'concrete floors': 'shutters', 'support block': 'pool',
              'basement': 'tower', 'trellis': 'stairs', 'skylights': 'shutters', 'outdoor light': 'ground',
              'step': 'beam', 'beam': 'beam', 'glass barrier': 'ground', 'support post': 'pool',
              'wall - doorway awning': 'awning', 'storage': 'wall', 'kitchen island': 'furniture', 'ceiling': 'ceiling',
              'painting': 'corridor', 'roof ornament': 'shutters', 'wade': 'dome', 'greens': 'pool',
              'doorframe': 'shutters', 'top': 'lighting', 'wood floor': 'shutters', 'display': 'lighting',
              'helipad': 'shutters', 'chair': 'roof', 'second story floor': 'stairs', 'piller': 'banister',
              'leaves': 'buttress', 'chimney wall': 'shutters', 'candle': 'lighting', 'windows': 'window',
              'was': 'garage', 'umbrella': 'fence', 'flooring': 'stairs', 'stone roof': 'shutters',
              'doorsteps': 'stairs', 'lighting': 'lighting', 'windowframe': 'roof', 'bricks': 'roof',
              'pink blocks': 'buttress', 'houseplant': 'wall', 'pink cubes': 'ground', 'archway': 'lighting',
              'marble': 'stairs', 'cabinetry': 'shutters', 'trim': 'window', 'floor support': 'ground',
              'interior walls': 'wall', 'fireplace': 'chimney', 'canopy': 'chimney', 'oven': 'dome',
              'cornice': 'lighting', 'wall frame': 'ground', 'wall': 'wall', 'loveseat': 'corridor',
              'wall vines': 'shutters', 'staircase': 'lighting', 'roof level': 'ground', 'shingles': 'wall',
              'bath': 'roof', 'stereo': 'fence', 'cauldron': 'stairs', 'shrubs': 'pool',
              'dining table chairs': 'stairs', 'fence': 'fence', 'enchanter': 'pool', 'porch floor': 'shutters',
              'seats': 'beam', 'floor \\ ceiling': 'wall', 'iron': 'chimney', 'statue': 'tower',
              'shrubbery': 'shutters', 'bookshelf': 'stairs', 'structural support': 'buttress', 'roof \\ floor': 'wall',
              'glass roof': 'shutters', 'torchlight': 'wall', 'stern': 'chimney', 'giant torch': 'lighting',
              'panel': 'dome', 'porch wall': 'shutters', 'patio railing': 'awning', 'post support': 'pool',
              'patio': 'balcony', 'wood': 'lighting', 'rail': 'furniture', 'crafting bench': 'stairs',
              'sconce': 'dormer', 'pond': 'furniture', 'interior nook': 'shutters', 'rooftop': 'dome',
              'front entrance': 'ground', 'stair roof': 'shutters', 'tower lighting': 'ground', 'cross support': 'wall',
              'post': 'beam', 'gates': 'window', 'ches': 'wall', 'pathway': 'dome', 'candles': 'roof',
              'front steps': 'shutters', 'internal lighting': 'ground', 'dirt patch 2': 'vehicle',
              'tree branches': 'shutters', 'bookcase': 'shutters', 'floor tiles': 'shutters', 'chest': 'roof',
              'light': 'beam', 'arch': 'arch', 'stone foundation': 'banister', 'dirt path': 'wall',
              'swimming pool': 'awning', 'stairs': 'stairs', 'skylight': 'shutters', 'walldecor': 'pool',
              'gable': 'vehicle', 'ground support': 'ground', 'bridge': 'window', 'shower': 'stairs',
              'stone': 'lighting', 'front door': 'shutters', 'river': 'fence', 'plant': 'plant',
              'wooden lattice': 'wall', 'doordecor': 'roof', 'light fixture': 'wall', 'paperd wall': 'stairs',
              'chiming': 'banister', 'walkway': 'lighting', 'roof tiles': 'shutters', 'picture window': 'wall',
              'dirt': 'furniture', 'brickwall': 'window', 'support column': 'pool', 'plants': 'dormer',
              'boxes': 'column', 'pot': 'chimney', 'seating': 'chimney', 'brick roof': 'shutters',
              'stone archway': 'shutters', 'bin': 'beam', 'g;ow stone': 'ground', 'porch fence': 'shutters',
              'torches (outdoors) ()': 'ground', 'roof window': 'shutters', 'glass wall': 'shutters',
              'half-wall': 'shutters', 'curb': 'chimney', 'garage': 'garage', 'aquarium': 'ground', 'nothing': 'beam',
              'blue blocks': 'banister', 'building foundation': 'wall', 'brick column': 'dormer',
              'retaining wall': 'shutters', 'staircase support': 'ground', 'stairssupport': 'stairs', 'ivy': 'window',
              'rocks': 'vehicle', 'utilities': 'lighting', 'cover': 'tower', 'stone platform': 'shutters',
              'support beam': 'stairs', 'forge': 'chimney', 'pink boxes': 'stairs', 'largebush': 'buttress',
              'shaft': 'tower', 'moss': 'chimney', 'rug': 'tower', 'nightstands': 'awning', 'countertop': 'shutters',
              'chimney': 'chimney', 'concrete': 'wall', 'block': 'lighting', 'inner frame': 'furniture',
              'roof trim': 'shutters', 'puddle': 'chimney', 'kitchen floor': 'shutters', 'gutter': 'wall',
              'corner lamp': 'shutters', 'facade wall': 'shutters', 'purple pilalr': 'dormer', 'tree trunk': 'stairs',
              'glass rail': 'furniture', 'wall vine': 'ground', 'moat water': 'shutters', 'box': 'tower',
              'stone base': 'ground', 'display case': 'shutters', 'column support': 'buttress', 'glasswall': 'roof',
              'flower balcony': 'stairs', 'irrigation': 'ground', 'entry arch': 'wall', 'potions': 'shutters',
              'hanging plant': 'shutters', 'torches (medium room) ()': 'ground', 'front yard': 'awning',
              'second floor': 'wall', 'glass window': 'ground', 'paper wall': 'shutters', 'port': 'lighting',
              'stone walkway': 'wall', 'entryway': 'stairs', 'brown box': 'vehicle', 'lattice': 'chimney',
              'house stilts': 'shutters', 'waterfall': 'stairs', 'fire': 'lighting', 'furnace': 'dome',
              'kitchen cabinets': 'awning', 'roof beam': 'shutters', 'bench': 'lighting', 'window ledge': 'shutters',
              'hangingplant': 'corridor', 'steps to entrance': 'stairs', 'hull': 'vehicle', 'overhang': 'stairs',
              'Galway': 'ground', 'eaves': 'arch', 'entry steps': 'vehicle', 'torches (small room) ()': 'stairs',
              'roofs': 'roof', 'base': 'gate', 'rack': 'window', 'concrete wall': 'shutters', 'garden wall': 'shutters',
              'inside wall': 'wall', 'roof support': 'shutters', 'planter box': 'stairs', 'subfloor': 'furniture',
              'all': 'window', 'wood archway': 'furniture', 'exterior lighting': 'shutters', 'tile floor': 'awning',
              'tables': 'furniture', 'dispenser': 'shutters', 'inside lights': 'vehicle', 'stove': 'chimney',
              'stone floor': 'shutters', 'base support': 'ground', 'interior': 'wall', 'stoop': 'roof',
              'entertainment center': 'shutters', 'hall floor': 'stairs', 'shelf': 'beam', 'railing post': 'shutters',
              'bar': 'corridor', 'windowsill': 'furniture', 'doorstep': 'chimney', 'axle': 'chimney',
              'purple box': 'corridor', 'trunk': 'roof', 'fern': 'chimney', 'metalbox': 'roof',
              'under window': 'furniture', 'stone pillar': 'ground', 'indoor plants': 'shutters', 'Wal': 'pool',
              'verdant': 'stairs', 'ladders': 'shutters', 'flower': 'furniture', 'beds': 'vehicle', 'round': 'tower',
              'ground floor': 'wall', 'interior wall': 'wall', 'chairs': 'banister', 'vines': 'beam', 'crate': 'beam',
              'torches (indoors, second floor) ()': 'shutters', 'roofgarden': 'dormer', 'brewer': 'parapet',
              'platform': 'ground', 'dirt patch 1': 'roof', 'chests': 'window', 'cement foundation': 'stairs',
              'support': 'lighting', 'dresser': 'vehicle', 'tree': 'fence', 'hedge': 'dome',
              'flower garden': 'shutters', 'headrest': 'awning', 'red box': 'corridor', 'bedroom floor': 'shutters',
              'shelves': 'stairs', 'seat': 'window', 'torches (indoor) ()': 'ground', 'outcroppings': 'shutters',
              'carpet': 'balcony', 'tv stand': 'wall', 'brewers': 'dome', 'platform to door': 'shutters',
              'cross': 'window', 'blade': 'garage', 'random': 'pool', 'marquee': 'stairs', 'lantern': 'window',
              'metalchest': 'corridor', 'glass tower': 'wall', 'floor mat': 'shutters', 'tile': 'fence',
              'nightstand': 'furniture', 'basement floor': 'ground', 'pillars': 'beam', 'wood column': 'wall',
              'bottom turret': 'stairs', 'living room floor': 'shutters', 'pink roof': 'balcony', 'lights': 'beam',
              'layer': 'chimney', 'cushions': 'window', 'porch railing': 'awning', 'bars': 'pool',
              'decoration': 'ground', 'timetable': 'chimney', 'bunker': 'dome', 'columns': 'pool', 'planter': 'dormer',
              'roof tower': 'shutters', 'hatch': 'window', 'ice wall': 'pool', 'pew': 'corridor',
              'wind chimes': 'shutters', 'upper floor': 'wall', 'porch': 'roof', 'fencing': 'ground',
              'cactus': 'vehicle', 'outside lights': 'wall', 'deck railing': 'awning', 'torches (pathway) ()': 'ground',
              'blue': 'dome', 'stone columns': 'shutters', 'brick border': 'furniture', 'wooden posts': 'shutters',
              'cap': 'lighting', 'support wall': 'shutters', 'corner': 'corridor', 'window': 'window',
              'garden plot': 'stairs', 'island': 'pool', 'cushion': 'lighting', 'side door': 'shutters',
              'foundation': 'fence', 'shelving': 'beam', 'farm': 'fence', 'battlements': 'shutters',
              'decksupport': 'fence', 'safe': 'balcony', 'ferns': 'furniture', 'porch support': 'wall',
              'woodsupport': 'stairs', 'books': 'banister', 'jack-o-lantern': 'stairs', 'bed': 'vehicle'}

# Define a dictionary for category colors
category_colors = {
    category: f'rgb({(hash(category) & 0xFF)}, {(hash(category) >> 8) & 0xFF}, {(hash(category) >> 16) & 0xFF})'
    for category in set(dictionary.values())
}
# Initialize a dictionary to store mesh data for each category

for input_sequence in data['input_sequences'][10:20]:
    category_mesh_data = {category: {'coords': []} for category in category_colors}

    for input_data in input_sequence:
        coord = input_data[0]
        category = dictionary[input_data[2]]
        category_mesh_data[category]['coords'].append([coord[0], coord[2], coord[1]])

    fig = go.Figure()
    for category, coords in category_mesh_data.items():
        coord = coords['coords']
        if len(coord) == 0:
            continue

        color = category_colors[category]
        mesh = create_mesh(coord, category, color)

        fig.add_trace(mesh)

    # Update the layout
    fig.update_layout(
        title='3D Data Visualization with Categories',
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        ),
        showlegend=True
    )

    fig.show()