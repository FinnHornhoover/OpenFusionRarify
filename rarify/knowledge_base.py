import json
from collections import defaultdict

from .config import Config, sanitize_key
from .drops import Drops, get_patched


class KnowledgeBase:
    def __init__(self, config: Config) -> None:
        self.crate_name_map = {
            'etc': 'ETC',
            'gumball': 'ETC',
            'standard': 'Standard',
            'gray': 'Standard',
            'normal': 'Standard',
            'special': 'Special',
            'orange': 'Special',
            'bronze': 'Special',
            'sooper': 'Sooper',
            'white': 'Sooper',
            'silver': 'Sooper',
            'sooper dooper': 'Sooper Dooper',
            'sooperdooper': 'Sooper Dooper',
            'yellow': 'Sooper Dooper',
            'gold': 'Sooper Dooper',
        }
        self.crate_name_order_map = {
            'ETC': 0,
            'Standard': 1,
            'Special': 2,
            'Sooper': 3,
            'Sooper Dooper': 4,
        }
        self.crate_order_name_map = {
            0: 'ETC',
            1: 'Standard',
            2: 'Special',
            3: 'Sooper',
            4: 'Sooper Dooper',
        }
        self.type_name_id_map = {
            'weapon': 0,
            'thrown': 0,
            'bomb': 0,
            'rifle': 0,
            'pistol': 0,
            'shattergun': 0,
            'gun': 0,
            'melee': 0,
            'sword': 0,
            'rocket': 0,
            'shirts': 1,
            'shirt': 1,
            'body': 1,
            'torso': 1,
            'pants': 2,
            'legs': 2,
            'leggings': 2,
            'shoes': 3,
            'boots': 3,
            'feet': 3,
            'hat': 4,
            'head': 4,
            'glass': 5,
            'glasses': 5,
            'face': 5,
            'mask': 5,
            'back': 6,
            'backpack': 6,
            'wings': 6,
            'general': 7,
            'other': 7,
            'misc': 7,
            'chest': 9,
            'crate': 9,
            'egg': 9,
            'capsule': 9,
            'vehicle': 10,
            'hoverboard': 10,
            'car': 10,
        }
        self.type_id_name_map = {
            0: 'Weapon',
            1: 'Shirts',
            2: 'Pants',
            3: 'Shoes',
            4: 'Hat',
            5: 'Glass',
            6: 'Back',
            7: 'General',
            9: 'Chest',
            10: 'Vehicle',
        }
        self.event_name_id_map = {
            'knishmas': 1,
            'christmas': 1,
            'halloween': 2,
            'easter': 3,
        }
        self.zone_to_egger_map = {
            zone_name: {
                crate_type: start_id + crate_order
                for crate_type, crate_order in self.crate_name_order_map.items()
                if crate_type != 'ETC'
            }
            for zone_name, start_id in zip(
                ['future', 'suburbs', 'downtown', 'wilds', 'darklands'],
                range(1140, 1180, 8)
            )
        }
        self.zone_to_mystery_egger_map = {
            zone_name: {
                crate_type: crate_id + 4
                for crate_type, crate_id in egger_map.items()
            }
            for zone_name, egger_map in self.zone_to_egger_map.items()
        }

        self.base_drops = Drops(config.base, config.patches)
        self.drops = Drops(config.base, config.patches)
        self.mobs = get_patched(
            base=(config.base.parent / 'mobs.json'),
            patches=[
                patch_path.parent / 'mobs.json'
                for patch_path in config.patches
                if (patch_path.parent / 'mobs.json').is_file()
            ],
        )
        self.eggs = get_patched(
            base=(config.base.parent / 'eggs.json'),
            patches=[
                patch_path.parent / 'eggs.json'
                for patch_path in config.patches
                if (patch_path.parent / 'eggs.json').is_file()
            ],
        )
        with open(config.xdt) as r:
            self.xdt = json.load(r)
        with open('info.json') as r:
            self.info = json.load(r)

        self.name_to_areas = defaultdict(list)
        for obj in self.info['MapRegions']:
            area_key = sanitize_key(obj['AreaName']).replace('the ', '')
            if 'Future' in obj['ZoneName']:
                area_key += ' f'
            self.name_to_areas[area_key].append(obj)

        self.egg_type_to_crate_id_map = {
            egg['Id']: egg['DropCrateId']
            for egg in self.eggs['EggTypes'].values()
        }

        self.area_eggs = defaultdict(set)
        for egg in self.eggs['Eggs'].values():
            crate_id = self.egg_type_to_crate_id_map.get(egg['iType'])
            if not crate_id:
                continue

            found = False
            for area_name, areas in self.name_to_areas.items():
                for area in areas:
                    if (
                        (area['X'] <= egg['iX'] < area['X'] + area['Width']) and
                        (area['Y'] <= egg['iY'] < area['Y'] + area['Height'])
                    ):
                        self.area_eggs[area_name].add(crate_id)
                        found = True
                        break
                if found:
                    break
        self.area_eggs = {k: list(v) for k, v in self.area_eggs.items()}

        self.loaded_mobs = {mob['iNPCType'] for mob in self.mobs['mobs'].values()}
        for mob_group in self.mobs['groups'].values():
            self.loaded_mobs.add(mob_group['iNPCType'])
            for mob in mob_group['aFollowers']:
                self.loaded_mobs.add(mob['iNPCType'])

        self.iz_name_id_map = {
            sanitize_key(obj['EPName']).replace('the ', ''): epid
            for epid, obj in self.base_drops['Racing'].iitems()
        }
        self.iz_name_id_map.update({
            'sector v f': 1,
            'pokey oaks north f': 2,
            'pokey oaks f': 2,
            'genius grove f': 3,
            'peach creek estates f': 4,
            'goats junk yard f': 5,
            'pokey oaks north': 7,
            'pokey oaks': 7,
            'genius grove': 8,
            'candy cove': 9,
            'peach creek estates': 10,
            'goats junk yard': 11,
            'eternal meadows': 12,
            'nuclear plant': 13,
            'habitat homes': 14,
            'charles darwin': 14,
            'city point': 15,
            'marquee row': 16,
            'sunny bridges': 16,
            'mount blackhead': 17,
            'leakey lake': 18,
            'townsville park': 19,
            'orchid bay': 20,
            'skate park': 20,
            'bravo beach': 21,
            'pimpleback mountains': 22,
            'morbucks towers': 23,
            'ruins': 24,
            'hanibaba temple': 24,
            'galaxy gardens': 25,
            'offworld plaza': 26,
            'space port': 26,
            'area 515': 27,
            'area 51': 27,
            'really twisted forest': 28,
            'monkey mountain': 29,
            'dinosaur pass': 30,
            'dino pass': 30,
            'dino graveyard': 30,
            'firepits': 31,
            'dark glade': 32,
            'dark tree': 32,
            'green maw': 33,
        })

        self.item_map = {
            (i, d['m_iItemNumber']): {
                **d,
                **self.xdt[f'm_p{table}ItemTable']['m_pItemStringData'][d['m_iItemName']]
            }
            for i, table in self.type_id_name_map.items()
            for d in self.xdt[f'm_p{table}ItemTable']['m_pItemData']
        }

        self.item_name_map = defaultdict(set)
        for item_tuple, d in self.item_map.items():
            name_sanitized = sanitize_key(d['m_strName'])
            self.item_name_map[name_sanitized].add(item_tuple)
        self.item_name_map = {k: list(v) for k, v in self.item_name_map.items()}

        self.mission_level_crate_id_map = defaultdict(set)
        for item_name, item_tuples in self.item_name_map.items():
            if 'mission crate' in item_name:
                level = int(item_name.split('lv')[0])
                for item_type, crate_id in item_tuples:
                    if item_type == 9 and crate_id in self.drops['Crates']:
                        self.mission_level_crate_id_map[level].add(crate_id)
        self.mission_level_crate_id_map = {
            k: list(v)
            for k, v in self.mission_level_crate_id_map.items()
            if v
        }

        self.npc_map = {
            d['m_iNpcNumber']: {
                **d,
                'm_strName': self.xdt['m_pNpcTable']['m_pNpcStringData'][d['m_iNpcName']]['m_strName']
            }
            for d in self.xdt['m_pNpcTable']['m_pNpcData']
        }

        self.npc_name_map = defaultdict(set)
        for npc_id, d in self.npc_map.items():
            name_sanitized = sanitize_key(d['m_strName'])
            self.npc_name_map[name_sanitized].add(npc_id)
        self.npc_name_map = {k: list(v) for k, v in self.npc_name_map.items()}

        self.tuple_irid_map = {
            (d['Type'], d['ItemID']): d['ItemReferenceID']
            for d in self.base_drops['ItemReferences'].values()
        }
        self.irid_tuple_map = {
            d['ItemReferenceID']: (d['Type'], d['ItemID'])
            for d in self.base_drops['ItemReferences'].values()
        }

        self.irid_is_map = defaultdict(list)
        for d in self.base_drops['ItemSets'].values():
            for irid in d['ItemReferenceIDs']:
                self.irid_is_map[irid].append(d)

        self.isid_crate_map = defaultdict(list)
        for d in self.base_drops['Crates'].values():
            self.isid_crate_map[d['ItemSetID']].append(d)

        self.cid_cdt_map = defaultdict(list)
        for d in self.base_drops['CrateDropTypes'].values():
            for cid in d['CrateIDs']:
                self.cid_cdt_map[cid].append(d)

        self.cdtid_mobdrop_map = defaultdict(list)
        for d in self.base_drops['MobDrops'].values():
            self.cdtid_mobdrop_map[d['CrateDropTypeID']].append(d)

        self.cdcid_cdc_map = {
            d['CrateDropChanceID']: d
            for d in self.base_drops['CrateDropChances'].values()
        }

        self.mdid_mob_map = defaultdict(list)
        for d in self.base_drops['Mobs'].values():
            self.mdid_mob_map[d['MobDropID']].append(d)

        self.mdid_event_map = defaultdict(list)
        for d in self.base_drops['Events'].values():
            self.mdid_event_map[d['MobDropID']].append(d)
