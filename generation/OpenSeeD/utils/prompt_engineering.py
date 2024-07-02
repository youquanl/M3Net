import numpy as np


def get_prompt_templates():
    prompt_templates = [
        '{}.',
        'a photo of a {}.',
        'a bad photo of a {}.',
        'a photo of many {}.',
        'a sculpture of a {}.',
        'a photo of the hard to see {}.',
        'a low resolution photo of the {}.',
        'a rendering of a {}.',
        'graffiti of a {}.',
        'a bad photo of the {}.',
        'a cropped photo of the {}.',
        'a tattoo of a {}.',
        'the embroidered {}.',
        'a photo of a hard to see {}.',
        'a bright photo of a {}.',
        'a photo of a clean {}.',
        'a photo of a dirty {}.',
        'a dark photo of the {}.',
        'a drawing of a {}.',
        'a photo of my {}.',
        'the plastic {}.',
        'a photo of the cool {}.',
        'a close-up photo of a {}.',
        'a black and white photo of the {}.',
        'a painting of the {}.',
        'a painting of a {}.',
        'a pixelated photo of the {}.',
        'a sculpture of the {}.',
        'a bright photo of the {}.',
        'a cropped photo of a {}.',
        'a plastic {}.',
        'a photo of the dirty {}.',
        'a jpeg corrupted photo of a {}.',
        'a blurry photo of the {}.',
        'a photo of the {}.',
        'a good photo of the {}.',
        'a rendering of the {}.',
        'a {} in a video game.',
        'a photo of one {}.',
        'a doodle of a {}.',
        'a close-up photo of the {}.',
        'the origami {}.',
        'the {} in a video game.',
        'a sketch of a {}.',
        'a doodle of the {}.',
        'a origami {}.',
        'a low resolution photo of a {}.',
        'the toy {}.',
        'a rendition of the {}.',
        'a photo of the clean {}.',
        'a photo of a large {}.',
        'a rendition of a {}.',
        'a photo of a nice {}.',
        'a photo of a weird {}.',
        'a blurry photo of a {}.',
        'a cartoon {}.',
        'art of a {}.',
        'a sketch of the {}.',
        'a embroidered {}.',
        'a pixelated photo of a {}.',
        'itap of the {}.',
        'a jpeg corrupted photo of the {}.',
        'a good photo of a {}.',
        'a plushie {}.',
        'a photo of the nice {}.',
        'a photo of the small {}.',
        'a photo of the weird {}.',
        'the cartoon {}.',
        'art of the {}.',
        'a drawing of the {}.',
        'a photo of the large {}.',
        'a black and white photo of a {}.',
        'the plushie {}.',
        'a dark photo of a {}.',
        'itap of a {}.',
        'graffiti of the {}.',
        'a toy {}.',
        'itap of my {}.',
        'a photo of a cool {}.',
        'a photo of a small {}.',
        'a tattoo of the {}.',
    ]
    return prompt_templates

nusc_predefine = {"barrier": ["barrier", "barricade"], 
                  "bicycle": ["bicycle"], 
                  "bus" :["bus"], 
                  "car": ["car"],
                 "construction vehicle":["bulldozer", "excavator", "concrete mixer", "crane", "dump truck"], 
                 "motorcycle": ["motorcycle"], 
                 "person":["person"], 
                 "traffic cone":["traffic cone"], 
                 "trailer":["trailer", "semi trailer", "cargo container", "shipping container", "freight container"], 
                 "truck":["truck"], 
                 "road":["road"], 
                 "other flat":["curb","traffic island","trafficmedian"], 
                 "sidewalk":["sidewalk"], 
                 "terrain":["grass", "grassland", "lawn", "meadow", "turf", "sod"], 
                 "manmade":["building", "wall", "pole", "awning","traffic sign", "traffic light"], 
                 "tree":["tree", "trunk", "tree trunk", "bush", "shrub", "plant", "flower", "woods"],
                 "sky":["sky"]}

kitti_predefine = {
        "car":["car"], 
        "bicycle": ["bicycle"],
        "motorcycle": ["motorcycle"], 
        "truck":["truck"],
        "other vehicle":["other vehicle","bulldozer", "excavator", "concrete mixer", "crane", "dump truck","bus","trailer", "semi trailer", "cargo container", "shipping container", "freight container"],
        "person":["person"], 
        "bicyclist":["bicyclist"], 
        "motorcyclist":["motorcyclist"],
        "road":["road"],
        "parking":["parking"], 
        "sidewalk":["sidewalk"], 
        "other ground":["other ground","curb","traffic island","trafficmedian"], 
        "building":["building"], 
        "fence":["fence"], 
        "tree":["tree"], 
        "trunk":["tree trunk","trunk"], 
        "terrain":["grass", "grassland", "lawn", "meadow", "turf", "sod"], 
        "pole":["pole"], 
        "traffic sign":["traffic sign"],
        "sky":["sky"],
}

waymo_predefine ={
     "car":["car"],
     "truck":["truck"],
     "bus":["bus"],
     "other vehicle": ["other vehicle","pedicab", "construction vehicle", "recreational vehicle", "limo", "tram","trailer", "semi trailer", "cargo container", "shipping container", "freight container","bulldozer", "excavator", "concrete mixer", "crane", "dump truck"],
     "motorcyclist":["motorcyclist"],
     "bicyclist":["bicyclist"], 
     "person":["person"],
     "traffic sign":["traffic sign",'parking-sign', 'direction-sign', 'traffic-sign without pole', 'traffic light box'],
     "traffic light":["traffic light"],
     "pole":["lamp post", "traffic sign pole"],
     "construction cone":["construction cone"],
     "bicycle": ["bicycle"],
     "motorcycle": ["motorcycle"], 
     "building":["building"],
     "vegetation":["bushes","tree branches", "tall grasses", "flowers","grass", "grassland", "lawn", "meadow", "turf", "sod"],
     "tree trunk":["tree trunk","trunk"],
     "curb":["curb"],
     "road":["road"],
     "lane marker":["lane marker"],
     "other ground":["other ground","bumps","cateyes","railtracks"],
     "walkable":["walkable","grassy hill","pedestrian walkway stairs"],
     "sidewalk":["sidewalk"],
     "sky":["sky"],
}

# waymo_predefine ={
#      00"car":["car"],
#      11"truck":["truck"],
#      22"bus":["bus"],
#      33"other vehicle": ["other vehicle","pedicab", "construction vehicle", "recreational vehicle", "limo", "tram","trailer", "semi trailer", "cargo container", "shipping container", "freight container","bulldozer", "excavator", "concrete mixer", "crane", "dump truck"],
#      44"motorcyclist":["motorcyclist"],
#      55"bicyclist":["bicyclist"], 
#      66"person":["person"],
#      9-7"traffic sign":["traffic sign",'parking-sign', 'direction-sign', 'traffic-sign without pole', 'traffic light box'],
#      10-8"traffic light":["traffic light"],
#      11-9"pole":["lamp post", "traffic sign pole"],
#      12-10"construction cone":["construction cone"],
#      7-11"bicycle": ["bicycle"],
#      8-12"motorcycle": ["motorcycle"], 
#      13 13"building":["building"],
#      14 14"vegetation":["bushes","tree branches", "tall grasses", "flowers","grass", "grassland", "lawn", "meadow", "turf", "sod"],
#      15 15"tree trunk":["tree trunk","trunk"],
#      16 16"curb":["curb"],
#      17 17"road":["road"],
#      18 18"lane marker":["lane marker"],
#      19 19"other ground":["other ground","bumps","cateyes","railtracks"],
#      20 20"walkable":["walkable","grassy hill","pedestrian walkway stairs"],
#      21 21"sidewalk":["sidewalk"],
#      22 22"sky":["sky"],
# }

ddd17_predifine = {
# 'background':['ground', 'parking', 'rail track', 'guard rail', 'bridge', 'tunnel', 'sky'],
'road':['road', 'driveable', 'street',  'lane marking', 'bicycle lane', 'roundabout lane', 'parking lane'], #0
'sky':['sky'], #1
'building':['building', 'house', 'bus stop building', 'garage', 'carport', 'scaffolding'], #1
'pole':['pole', 'electric pole', 'traffic sign pole', 'traffic light pole'], #2
'traffic sign':['traffic-sign', 'parking-sign', 'direction-sign', 'traffic-sign without pole', 'traffic light box'],#2
'vegetation':['vegetation', 'vertical vegetation', 'tree', 'tree trunk', 'hedge', 'woods'],#3

'person':['person', 'pedestrian', 'walking people', 'standing people', 'sitting people', 'toddler'], # 4


# 'sidewalk':['sidewalk', 'delimiting curb', 'traffic island', 'walkable', 'pedestrian zone'],
 
'car':['car', 'jeep', 'SUV', 'van', 'caravan', 'truck', 'box truck', 'pickup truck', 'trailer', 'bus', 'public bus', 'train', 'vehicle-on-rail', 'tram', 'motorbike', 'moped', 'scooter', 'bicycle'], 


}


dsec_predifine = {
# 'background':['ground', 'parking', 'rail track', 'guard rail', 'bridge', 'tunnel', 'sky'],
'sky':['sky'],
'building':['building', 'house', 'bus stop building', 'garage', 'carport', 'scaffolding'],
'fence':['fence', 'fence with hole'],
'person':['person', 'pedestrian', 'walking people', 'standing people', 'sitting people', 'toddler'],
'pole':['pole', 'electric pole', 'traffic sign pole', 'traffic light pole'],
'road':['road', 'driveable', 'street',  'lane marking', 'bicycle lane', 'roundabout lane', 'parking lane'],
'sidewalk':['sidewalk', 'delimiting curb', 'traffic island', 'walkable', 'pedestrian zone'],
'vegetation':['vegetation', 'vertical vegetation', 'tree', 'tree trunk', 'hedge', 'woods', 'terrain', 'grass', 'soil', 'sand', 'lawn', 'meadow', 'turf'], 
'car':['car', 'jeep', 'SUV', 'van', 'caravan', 'truck', 'box truck', 'pickup truck', 'trailer', 'bus', 'public bus', 'train', 'vehicle-on-rail', 'tram', 'motorbike', 'moped', 'scooter', 'bicycle'], 
'wall':['wall', 'standing wall'], 
'traffic sign':['traffic-sign', 'parking-sign', 'direction-sign', 'traffic-sign without pole', 'traffic light box']
}


dsec_predifine_rare = {
# 'background':['ground', 'parking', 'rail track', 'guard rail', 'bridge', 'tunnel', 'sky'],
'sky':['sky'],
'building':['building', 'house', 'bus stop building', 'garage', 'carport', 'scaffolding'],
'fence':['fence', 'fence with hole'],
'person':['person', 'pedestrian', 'walking people', 'standing people', 'sitting people', 'toddler'],
'pole':['pole', 'electric pole', 'traffic sign pole', 'traffic light pole'],
'road':['road', 'driveable', 'street',  'lane marking', 'bicycle lane', 'roundabout lane', 'parking lane'],
'sidewalk':['sidewalk', 'delimiting curb', 'traffic island', 'walkable', 'pedestrian zone'],
'tree':['vegetation', 'vertical vegetation', 'tree', 'tree trunk', 'hedge', 'woods', 'terrain', 'grass', 'soil', 'sand', 'lawn', 'meadow', 'turf'], 
'car':['car', 'jeep', 'SUV', 'van', 'caravan', 'truck', 'box truck', 'pickup truck',  'bus', 'public bus', 'train', 'vehicle-on-rail', 'tram', 'motorbike', 'moped', 'scooter', 'bicycle'], 
'wall':['wall', 'standing wall'], 
'traffic sign':['traffic-sign', 'parking-sign', 'direction-sign', 'traffic-sign without pole', 'traffic light box'],
'rubbish bin':['rubbish bin'],
'crane':['crane'],
'construction equipment':['construction equipment'],
'trailer':['trailer'],
'people in trailer':['people in trailer'],
'trolley':['trolley'],
'excavator':['excavator'],
'container':['container'],
'traffic-cone':['traffic-cone'],
'railway':['railway'],
'people-with-trolley':['people-with-trolley'],




}

dsec_19_predifine = {
'road':['road', 'driveable', 'street', 'lane marking', 'bicycle lane', 'roundabout lane', 'parking lane'],
'sidewalk':['sidewalk', 'delimiting curb', 'traffic island', 'walkable', 'pedestrian zone'],
'building':['building', 'house',  'skyscraper', 'bus stop building', 'garage', 'carport', 'scaffolding'],
'wall':['wall', 'standing wall'], 
'fence':['fence', 'fence with hole'],
'pole':['pole', 'electric pole', 'traffic sign pole', 'traffic light pole'],
 'traffic light':['traffic light box'],
'traffic sign':['traffic-sign', 'parking-sign', 'direction-sign', 'traffic-sign without pole'],
'vegetation':['vegetation', 'vertical vegetation', 'tree', 'tree trunk', 'hedge', 'woods'],
'terrain':['terrain', 'grass', 'soil', 'sand', 'lawn', 'meadow', 'turf'],
'sky':['sky'],
'person':['person', 'pedestrian', 'walking people', 'standing people', 'sitting people', 'toddler'],
'rider':['rider', 'bicyclist', 'motorcyclist'],
'car':['car', 'jeep', 'SUV', 'van', 'caravan'],
'truck':['truck', 'box truck', 'pickup truck', 'trailer'],
'bus':['bus', 'public bus'],
'train':['train', 'vehicle-on-rail', 'tram'],
'motorcycle':['motorbike', 'moped', 'scooter'],
'bicycle':['bike', 'bicycle']

}


def prompt_engineering(classnames, pre=None, topk=1, suffix='.'):
    prompt_templates = get_prompt_templates()
    temp_idx = np.random.randint(min(len(prompt_templates), topk))

    if isinstance(classnames, list):
        classname = random.choice(classnames)
    else:
        classname = classnames
    dataset = "waymo"

    if pre == "truck":
        dataset = "semantickitti"
    elif pre == "car":
        dataset = "nuscenes"
    elif pre == "other vehicle":
        dataset = "waymo"
    else:
        import pdb; pdb.set_trace()

    
    class_list = ""
    if dataset == "semantickitti":
        # import pdb; pdb.set_trace()
        classname_list = kitti_predefine[classname]
        for classname in classname_list:
            class_list += prompt_templates[temp_idx].replace('.', suffix).format(classname.replace(',', '').replace('+', ' '))
        return class_list
    
    elif dataset == "nuscenes":
        classname_list = nusc_predefine[classname]
        for classname in classname_list:
            class_list += prompt_templates[temp_idx].replace('.', suffix).format(classname.replace(',', '').replace('+', ' '))
        return class_list
    
    elif dataset == "waymo":
        classname_list = waymo_predefine[classname]
        for classname in classname_list:
            class_list += prompt_templates[temp_idx].replace('.', suffix).format(classname.replace(',', '').replace('+', ' '))
        return class_list
    
    elif dataset == "dsec":
        classname_list = dsec_predifine[classname]
        for classname in classname_list:
            class_list += prompt_templates[temp_idx].replace('.', suffix).format(classname.replace(',', '').replace('+', ' '))
        return class_list
    
    elif dataset == "dsec_19":
        classname_list = dsec_19_predifine[classname]
        for classname in classname_list:
            class_list += prompt_templates[temp_idx].replace('.', suffix).format(classname.replace(',', '').replace('+', ' '))
        return class_list
    
    elif dataset == "ddd17":
        classname_list = ddd17_predifine[classname]
        for classname in classname_list:
            class_list += prompt_templates[temp_idx].replace('.', suffix).format(classname.replace(',', '').replace('+', ' '))
        return class_list

    
    # elif dataset == "dsec_rare":
    #     classname_list = dsec_predifine_rare[classname]
    #     for classname in classname_list:
    #         class_list += prompt_templates[temp_idx].replace('.', suffix).format(classname.replace(',', '').replace('+', ' '))
    #     return class_list
    else:
        return prompt_templates[temp_idx].replace('.', suffix).format(classname.replace(',', '').replace('+', ' '))