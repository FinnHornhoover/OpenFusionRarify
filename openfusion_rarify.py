import re
import sys
import json
from pathlib import Path
from copy import deepcopy
from operator import itemgetter
from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Set, List, Dict, Union, Tuple

import yaml
from sympy import Expr, Symbol, Eq
from sympy.solvers import solve


def patch(base_obj: Dict[str, Any], patch_obj: Dict[str, Any]) -> None:
    for key, value in patch_obj.items():
        if key[0] == '!':
            base_obj[key[1:]] = value
            continue

        if key in base_obj:
            if value is None:
                del base_obj[key]
            elif not isinstance(value, (dict, list)):
                base_obj[key] = value
            elif isinstance(value, list):
                base_obj[key].extend(value)
            else:
                patch(base_obj[key], value)
        else:
            base_obj[key] = value


def get_patched(base: Path, patches: List[Path]) -> Dict[str, Any]:
    with open(base) as r:
        base_obj = json.load(r)

    for patch_path in patches:
        with open(patch_path) as r:
            patch_obj = json.load(r)
        patch(base_obj, patch_obj)

    return base_obj


def generate_patch(base_obj: Dict[str, Any], updated_obj: Dict[str, Any]) -> Dict[str, Any]:
    target_obj = {}

    all_keys = dict.fromkeys(base_obj)
    all_keys.update(dict.fromkeys(updated_obj))

    for key in all_keys:
        if key in base_obj and key in updated_obj:
            base_val = base_obj[key]
            updated_val = updated_obj[key]

            if isinstance(updated_val, dict):
                target_val = generate_patch(base_val, updated_val)
                if target_val:
                    target_obj[key] = target_val
            elif base_val != updated_val:
                target_obj['!' + key] = updated_val

        elif key in base_obj:
            target_obj[key] = None
        else:
            target_obj[key] = updated_obj[key]

    return target_obj


def mapify_drops(drops):
    keymap = {
        'Racing': 'EPID',
        'NanoCapsules': 'Nano',
        'CodeItems': 'Code',
    }

    drop_maps = {}
    for key in drops:
        real_key = keymap.get(key, key[:-1] + 'ID')
        drop_maps[key] = {obj[real_key]: obj for obj in drops[key].values()}

    return drop_maps


@dataclass
class Config:
    base: Path
    patches: List[Path]
    xdt: Path
    output: Path
    generate_patch: bool


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

        self.base_drops = get_patched(config.base, config.patches)
        self.drops = deepcopy(self.base_drops)
        self.drop_maps = mapify_drops(self.base_drops)
        with open(config.xdt) as r:
            self.xdt = json.load(r)

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
            name_sanitized = re.sub(r'[^\w\s\d]', '', d['m_strName'].lower()).strip()
            self.item_name_map[name_sanitized].add(item_tuple)
        self.item_name_map = {k: list(v) for k, v in self.item_name_map.items()}

        self.npc_map = {
            d['m_iNpcNumber']: {
                **d,
                'm_strName': self.xdt['m_pNpcTable']['m_pNpcStringData'][d['m_iNpcName']]['m_strName']
            }
            for d in self.xdt['m_pNpcTable']['m_pNpcData']
        }

        self.npc_name_map = defaultdict(set)
        for npc_id, d in self.npc_map.items():
            name_sanitized = re.sub(r'[^\w\s\d]', '', d['m_strName'].lower()).strip()
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


@dataclass
class ItemConfig:
    type: int
    type_name: str
    id: int
    name: str
    criterion: str
    number_of_kills_till_item: Union[None, int]
    number_of_kills_till_item_per_mob: Union[None, Dict[int, int]]
    number_of_crates_till_item: Union[None, int]
    number_of_crates_till_item_per_crate_type: Union[None, Dict[str, int]]

    @staticmethod
    def from_dict(knowledge_base: KnowledgeBase, data: Dict[str, Any]):
        _type = data.get('type')
        _id = data.get('id')
        item_name = data.get('name')

        if _type and isinstance(_type, str):
            type_name_sanitized = re.sub(r'[^\w\s\d]', '', _type.lower()).strip()
            _type = knowledge_base.type_name_id_map.get(type_name_sanitized)

        item_tuple = _type, _id
        found_tuple = None, None

        if item_name:
            item_name_sanitized = re.sub(r'[^\w\s\d]', '', item_name.lower()).strip()
            found_tuples = knowledge_base.item_name_map.get(item_name_sanitized)
            if found_tuples:
                found_tuple = found_tuples[0]
                if len(found_tuples) > 1:
                    print('Warning: Item name', item_name, 'refers to items', found_tuples, ', using', found_tuple, '...')

        item_tuple_valid = not any(val is None for val in item_tuple)
        found_tuple_valid = not any(val is None for val in found_tuple)

        if item_tuple_valid and found_tuple_valid:
            if item_tuple != found_tuple:
                print('Warning:', item_name, 'refers to', found_tuple, 'but', item_tuple, 'was specified, using', item_tuple, '...')
        elif found_tuple_valid:
            item_tuple = found_tuple
        elif not item_tuple_valid and not found_tuple_valid:
            raise ValueError('Item cannot be identified: ' + json.dumps(data))

        type_id, item_id = item_tuple
        type_name = knowledge_base.type_id_name_map.get(type_id)

        if type_name is None:
            raise ValueError('Item name could not be idenfied: ' + json.dumps(data))

        item = knowledge_base.item_map.get(item_tuple)

        if not item:
            raise ValueError('Item cannot be identified: ' + json.dumps(data))

        item_name = item['m_strName']

        criteria = {
            'number_of_kills_till_item': None,
            'number_of_kills_till_item_per_mob': None,
            'number_of_crates_till_item': None,
            'number_of_crates_till_item_per_crate_type': None,
        }
        criterion = None

        for criterion_name in criteria:
            criterion_value = data.get(criterion_name.replace('_', '-'))

            if criterion_value:
                criterion = criterion_name
                criteria[criterion_name] = criterion_value
                break

        if criterion == 'number_of_kills_till_item_per_mob':
            new_map = {}

            for key, value in criteria[criterion].items():
                try:
                    mob_id = int(key)
                except ValueError:
                    mob_name_sanitized = re.sub(r'[^\w\s\d]', '', key.lower()).strip()
                    mob_ids = knowledge_base.npc_name_map.get(mob_name_sanitized)

                    if not mob_ids:
                        print('Warning: Mob', key, 'could not be found, skipping...')
                        continue

                    mob_id = mob_ids[0]
                    if len(mob_ids) > 1:
                        print('Warning: Mob name', key, 'refers to mobs', mob_ids, ', using', mob_id, '...')

                    mob = knowledge_base.npc_map.get(mob_id)

                if not mob:
                    print('Warning: Mob', key, 'could not be found, skipping...')
                    continue

                if value < 1:
                    continue

                new_map[mob_id] = value

            criteria[criterion] = new_map

        if criterion == 'number_of_crates_till_item_per_crate_type':
            new_map = {}

            for key, value in criteria[criterion].items():
                crate_name_sanitized = re.sub(r'[^\w\s\d]', '', key.lower()).replace('crate', '').strip()
                crate_name = knowledge_base.crate_name_map.get(crate_name_sanitized)

                if crate_name is None:
                    print('Warning: CRATE type', key, 'could not be found, skipping...')
                    continue

                if value < 1:
                    continue

                new_map[crate_name] = value

            criteria[criterion] = new_map

        if not any([(isinstance(value, int) and value > 0) or (isinstance(value, dict) and len(value) > 0) for value in criteria.values()]):
            raise ValueError('Edit configuration(s) not filled: ' + json.dumps(data))

        return ItemConfig(
            type=type_id,
            type_name=type_name,
            id=item_id,
            name=item_name,
            criterion=criterion,
            **criteria,
        )


def alter_chances(knowledge_base: KnowledgeBase, item_configs: List[ItemConfig]):
    affected_items = {
        knowledge_base.tuple_irid_map[(item_config.type, item_config.id)]: item_config
        for item_config in item_configs
        if item_config.criterion and getattr(item_config, item_config.criterion)
    }

    affected_itemsets = {
        itemset['ItemSetID']: itemset
        for ir_id in affected_items
        for itemset in knowledge_base.irid_is_map[ir_id]
    }

    itemset_ir_ids = {
        is_id: {ir_id for ir_id in itemset['ItemReferenceIDs'] if ir_id in affected_items}
        for is_id, itemset in affected_itemsets.items()
    }

    crate_ir_ids = {
        crate['CrateID']: ir_id_set
        for is_id, ir_id_set in itemset_ir_ids.items()
        for crate in knowledge_base.isid_crate_map[is_id]
    }

    cdt_ir_ids = {
        cdt['CrateDropTypeID']: {ir_id for cc in cdt['CrateIDs'] for ir_id in crate_ir_ids.get(cc, set())}
        for c_id in crate_ir_ids
        for cdt in knowledge_base.cid_cdt_map[c_id]
    }

    cdt_cdc_ir_ids = {
        (cdt_id, knowledge_base.drop_maps['CrateDropChances'][mobdrop['CrateDropChanceID']]['CrateDropChanceID']): ir_id_set
        for cdt_id, ir_id_set in cdt_ir_ids.items()
        for mobdrop in knowledge_base.cdtid_mobdrop_map[cdt_id]
    }

    def crate_symbols(gender_id: int, c_id: int, target_ir_ids: Set[int]) -> Dict[int, Expr]:
        crate = knowledge_base.drop_maps['Crates'][c_id]
        is_id = crate['ItemSetID']
        rw_id = crate['RarityWeightID']

        itemset = knowledge_base.drop_maps['ItemSets'][is_id]
        rarity_weights = knowledge_base.drop_maps['RarityWeights'][rw_id]['Weights']
        rw_sum = sum(rarity_weights)

        filtered_target_ir_ids = {ir_id for ir_id in itemset['ItemReferenceIDs']
                                  if ir_id in target_ir_ids}

        rarity_agg = defaultdict(float)

        for rarity_id, rw in zip(range(1, 5), rarity_weights):
            weights_filtered = {
                ir_id: itemset['AlterItemWeightMap'].get(str(ir_id), itemset['DefaultItemWeight'])
                for ir_id in itemset['ItemReferenceIDs']
                if (
                    (
                        itemset['IgnoreGender'] or
                        itemset['AlterGenderMap'].get(
                            str(ir_id),
                            knowledge_base.item_map[knowledge_base.irid_tuple_map[ir_id]].get('m_iReqSex', 0)
                        ) in [0, gender_id]
                    ) and
                    (
                        itemset['IgnoreRarity'] or
                        itemset['AlterRarityMap'].get(
                            str(ir_id),
                            knowledge_base.item_map[knowledge_base.irid_tuple_map[ir_id]].get('m_iRarity', 0)
                        ) in [0, rarity_id]
                    )
                )
            }
            total_weight = sum(weights_filtered.values())
            weights_scaled = {
                ir_id: weight / total_weight
                for ir_id, weight in weights_filtered.items()
            }

            weights_sym = {
                ir_id: Symbol(f'w_{is_id}_{ir_id}', real=True)
                for ir_id in weights_scaled
                if ir_id in filtered_target_ir_ids
            }
            target_weight_sum = sum(weights_sym.values())
            non_target_weight_sum = sum([w for ir_id, w in weights_scaled.items()
                                         if ir_id not in filtered_target_ir_ids])

            weights_sym.update({
                ir_id: (1. - target_weight_sum) * weight / non_target_weight_sum
                for ir_id, weight in weights_scaled.items()
                if ir_id not in filtered_target_ir_ids
            })

            for ir_id, sym_w in weights_sym.items():
                rarity_agg[ir_id] += sym_w * rw / rw_sum

        return rarity_agg

    def drop_symbols(cdt_id: int, cdc_id: int, target_ir_ids: Set[int]) -> List[Dict[int, Expr]]:
        cdc = knowledge_base.drop_maps['CrateDropChances'][cdc_id]
        cdt = knowledge_base.drop_maps['CrateDropTypes'][cdt_id]
        cdwt = sum(cdc['CrateTypeDropWeights'])

        mob_aggs = []

        for gender_id in range(1, 3):
            mob_agg = defaultdict(float)

            for crate_id, crate_w in zip(cdt['CrateIDs'], cdc['CrateTypeDropWeights']):
                for ir_id, sym_w in crate_symbols(gender_id, crate_id, target_ir_ids).items():
                    mob_agg[ir_id] += sym_w * crate_w * cdc['DropChance'] / (cdwt * cdc['DropChanceTotal'])

            mob_aggs.append(mob_agg)

        return mob_aggs

    def get_crate_weight(cdc: Dict[str, Any], crate_type: str) -> int:
        crate_order = knowledge_base.crate_name_order_map[crate_type]
        max_index = len(cdc['CrateTypeDropWeights']) - 1
        return cdc['CrateTypeDropWeights'][min(max_index, crate_order)]

    for (cdt_id, cdc_id), target_ir_ids in cdt_cdc_ir_ids.items():
        cdc = knowledge_base.drop_maps['CrateDropChances'][cdc_id]
        base_crate_prob = cdc['DropChance'] / cdc['DropChanceTotal']
        cdwt = sum(cdc['CrateTypeDropWeights'])

        boy_syms, girl_syms = drop_symbols(cdt_id, cdc_id, target_ir_ids)
        boy_solns, girl_solns = {}, {}
        boy_count, girl_count = 0, 0

        for ir_id in target_ir_ids:
            item_config = affected_items[ir_id]
            item_info = knowledge_base.item_map[knowledge_base.irid_tuple_map[ir_id]]

            if item_config.criterion == 'number_of_kills_till_item':
                prob = 1. / item_config.number_of_kills_till_item
            elif item_config.criterion == 'number_of_kills_till_item_per_mob':
                prob = min([
                    1. / item_config.number_of_kills_till_item_per_mob.get(mob['MobID'], 1)
                    for mobdrop in knowledge_base.cdtid_mobdrop_map[cdt_id]
                    for mob in knowledge_base.mdid_mob_map[mobdrop['MobDropID']]
                ])
            elif item_config.criterion == 'number_of_crates_till_item':
                prob = base_crate_prob / item_config.number_of_crates_till_item
            elif item_config.criterion == 'number_of_crates_till_item_per_crate_type':
                prob = min([
                    base_crate_prob * get_crate_weight(cdc, crate_type) / (cdwt * value)
                    for crate_type, value in item_config.number_of_crates_till_item_per_crate_type.items()
                ])
            else:
                print('Warning: No criteria applied for item', item_info['m_strName'])
                continue

            boy_solns.update(solve(Eq(boy_syms[ir_id], prob), dict=True)[0])
            girl_solns.update(solve(Eq(girl_syms[ir_id], prob), dict=True)[0])

            girl_prob_when_boy_soln = girl_syms[ir_id].subs(boy_solns)
            boy_prob_when_girl_soln = boy_syms[ir_id].subs(girl_solns)

            if girl_prob_when_boy_soln < boy_prob_when_girl_soln:
                girl_count += 1
            else:
                boy_count += 1

        solutions = boy_solns if boy_count > girl_count else girl_solns

        float_values = [{
            ir_id: (expr.subs(solutions) if isinstance(expr, Expr) else expr)
            for ir_id, expr in syms.items()
        } for syms in [boy_syms, girl_syms]]
        item_affected = {
            ir_id: isinstance(expr, Expr)
            for syms in [boy_syms, girl_syms]
            for ir_id, expr in syms.items()
        }

        def search(lo: int, hi: int) -> int:
            if hi < lo:
                return -1

            mi = (lo + hi) // 2

            current_values = [{
                ir_id: int(mi * value)
                for ir_id, value in values.items()
            } for values in float_values]

            cond = all(
                abs(sum(values.values()) / values[ir_id] - base_crate_prob / orig_values[ir_id]) < 0.5
                for values, orig_values in zip(current_values, float_values)
                for ir_id in values
            )

            if cond:
                left = search(lo, mi - 1)
                return mi if left == -1 else left
            else:
                return search(mi + 1, hi)

        scale = search(1, (1 << 31) - 1)
        scaled_values = {
            ir_id: int(scale * value)
            for values in float_values
            for ir_id, value in values.items()
            if item_affected[ir_id]
        }

        def calculate_weight_settings(
            itemset: Dict[str, Any],
            new_weights: Dict[int, int],
        ) -> Tuple[int, Dict[int, int]]:

            weights = {
                ir_id: itemset['AlterItemWeightMap'].get(str(ir_id), itemset['DefaultItemWeight'])
                for ir_id in itemset['ItemReferenceIDs']
            }
            weights.update(new_weights)

            counts = defaultdict(int)
            for weight in weights.values():
                counts[weight] += 1

            default_weight = max(counts.items(), key=itemgetter(1))[0]
            diff_weights = {
                ir_id: weight
                for ir_id, weight in weights.items()
                if weight != default_weight
            }

            return default_weight, diff_weights

        for is_id, affected_ir_ids in itemset_ir_ids.items():
            if not any(
                ir_id in affected_ir_ids
                for ir_id in scaled_values
            ):
                continue

            for itemset in knowledge_base.drops['ItemSets'].values():
                if is_id != itemset['ItemSetID']:
                    continue

                filtered_scaled_values = {
                    ir_id: scaled_values[ir_id]
                    for ir_id in itemset['ItemReferenceIDs']
                    if ir_id in scaled_values
                }

                default_weight, diff_weights = calculate_weight_settings(
                    itemset, filtered_scaled_values
                )

                itemset['DefaultItemWeight'] = default_weight
                itemset['AlterItemWeightMap'] = diff_weights
                break


def save_new_drops(knowledge_base: KnowledgeBase, config: Config):
    object_to_save = (
        generate_patch(knowledge_base.base_drops, knowledge_base.drops)
        if config.generate_patch else
        knowledge_base.drops
    )
    path_to_save = (
        config.output / 'new_patch'
        if config.generate_patch else
        config.output
    )

    path_to_save.mkdir(parents=True, exist_ok=True)
    with open(path_to_save / 'drops.json', 'w') as w:
        json.dump(object_to_save, w, indent=4)


def main():
    config_path = Path('rarifyconfig.yml' if len(sys.argv) < 2 else sys.argv[1])

    with open(config_path) as r:
        config_data = yaml.safe_load(r)

    config = Config(
        base=Path(config_data['base']),
        patches=[Path(patch_path) for patch_path in config_data['patches']],
        xdt=Path(config_data['xdt']),
        output=Path(config_data['output']),
        generate_patch=config_data['generate-patch'],
    )

    knowledge_base = KnowledgeBase(config)
    item_configs = [ItemConfig.from_dict(knowledge_base, data) for data in config_data['items']]

    alter_chances(knowledge_base, item_configs)
    save_new_drops(knowledge_base, config)


if __name__ == '__main__':
    main()
