import re
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass

OTHER_KEYWORDS = {
    'other',
    'others',
    'otherwise',
    'else',
}
OTHER_STANDARD_KEYWORD = 'Other'
OTHER_STANDARD_ID = -1


def sanitize_key(key: str) -> str:
    lower_spaced = key.lower().replace('-', '_').replace('_', ' ')
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s\d]', '', lower_spaced).strip())


@dataclass
class Config:
    base: Path
    patches: List[Path]
    xdt: Path
    output: Path
    generate_patch: bool


@dataclass
class ItemConfig:
    from .knowledge_base import KnowledgeBase

    type: int
    type_name: str
    id: int
    name: str
    kills_till_item: Optional[float]
    kills_till_item_per_mob: Optional[Dict[int, float]]
    crates_till_item: Optional[float]
    crates_till_item_per_mob: Optional[Dict[int, float]]
    crates_till_item_per_crate_type: Optional[Dict[str, float]]
    crates_till_item_per_mob_and_crate_type: Optional[Dict[int, Dict[str, float]]]
    crates_till_item_per_crate_type_and_mob: Optional[Dict[str, Dict[int, float]]]
    event_kills_till_item: Optional[Dict[int, float]]
    event_crates_till_item: Optional[Dict[int, float]]
    race_crates_till_item: Optional[Dict[int, Dict[str, float]]]
    mission_crates_till_item: Optional[Dict[int, float]]
    egger_eggs_till_item: Optional[Dict[str, Dict[str, float]]]
    golden_eggs_till_item: Optional[Dict[str, float]]
    named_crates_till_item: Optional[Dict[int, float]]

    @classmethod
    def from_dict(
        cls,
        knowledge_base: KnowledgeBase,
        data: Dict[str, Any],
    ) -> Optional['ItemConfig']:

        data_sanitized = {
            sanitize_key(key).replace(' ', '_'): value
            for key, value in data.items()
        }

        item_info = cls.fix_item(knowledge_base, data_sanitized)
        if item_info is None:
            # warnings are delegated to the above method
            return None

        criteria = {
            'kills_till_item': None,
            'kills_till_item_per_mob': None,
            'crates_till_item': None,
            'crates_till_item_per_mob': None,
            'crates_till_item_per_crate_type': None,
            'crates_till_item_per_mob_and_crate_type': None,
            'crates_till_item_per_crate_type_and_mob': None,
            'event_kills_till_item': None,
            'event_crates_till_item': None,
            'race_crates_till_item': None,
            'mission_crates_till_item': None,
            'egger_eggs_till_item': None,
            'golden_eggs_till_item': None,
            'named_crates_till_item': None,
        }

        any_exists = False
        for key, value in data_sanitized.items():
            if key in criteria:
                func = getattr(cls, f'fix_{key}')
                criteria[key] = func(value, knowledge_base=knowledge_base, **item_info)
                any_exists = any_exists or (criteria[key] is not None)

        if not any_exists:
            logging.warn('No valid requirements for item: %s, skipping ...',
                         data_sanitized)
            return None

        return ItemConfig(**item_info, **criteria)

    @classmethod
    def fix_item(
        cls,
        knowledge_base: KnowledgeBase,
        data_sanitized: Dict[str, Any],
    ) -> Optional[Dict[str, Union[int, str]]]:

        _type = data_sanitized.get('type')
        _id = data_sanitized.get('id')
        item_name = data_sanitized.get('name')

        if _type and isinstance(_type, str):
            _type = knowledge_base.type_name_id_map.get(sanitize_key(_type))

        item_tuple = (_type, _id)
        found_tuple = (None, None)

        if item_name:
            found_tuples = knowledge_base.item_name_map.get(sanitize_key(item_name))
            found_tuple = found_tuples[0]
            if len(found_tuples) > 1:
                logging.warn('%s refers to items %s, using %s ...',
                             item_name, found_tuples, found_tuple)

        item_tuple_valid = not any(val is None for val in item_tuple)
        found_tuple_valid = not any(val is None for val in found_tuple)

        if item_tuple_valid and found_tuple_valid:
            if item_tuple != found_tuple:
                logging.warn('%s refers to %s but %s was specified, using %s ...',
                             item_name, found_tuple, item_tuple, item_tuple)
        elif found_tuple_valid:
            item_tuple = found_tuple
        elif not item_tuple_valid and not found_tuple_valid:
            logging.warn('Item could not be identified: %s, skipping ...',
                          data_sanitized)
            return None

        type_id, item_id = item_tuple

        type_name = knowledge_base.type_id_name_map.get(type_id)
        if type_name is None:
            logging.warn('Type name could not be identified: %s, skipping ...',
                          data_sanitized)
            return None

        item = knowledge_base.item_map.get(item_tuple)
        if item is None:
            logging.warn('Item could not be identified: %s, skipping ...',
                          data_sanitized)
            return None

        item_name = item['m_strName']

        return {
            'type': type_id,
            'type_name': type_name,
            'id': item_id,
            'name': item_name,
        }

    @classmethod
    def fix_number(cls, value: Any) -> Optional[float]:
        # caller handles warnings
        if not isinstance(value, (int, float)) or value < 0:
            return None
        if value == 0 or value == float('nan'):
            return float('inf')
        return float(value)

    @classmethod
    def fix_value_till_item_per_mob(
        cls,
        value: Any,
        knowledge_base: KnowledgeBase,
        **kwargs,
    ) ->  Optional[Dict[int, float]]:

        if not isinstance(value, dict):
            logging.warn(
                'Value provided does not have key value pairs inside: %s, skipping ...',
                value
            )
            return None

        new_map = {}

        for k, v in value.items():
            try:
                mob_id = int(k)
            except ValueError:
                if not isinstance(k, str):
                    logging.warn('Invalid type for mob %s, skipping ...', k)
                    continue

                k_sanitized = sanitize_key(k)

                if k_sanitized in OTHER_KEYWORDS:
                    mob_ids = [OTHER_STANDARD_ID]
                else:
                    mob_ids = [
                        mob_id
                        for mob_id in knowledge_base.npc_name_map.get(k_sanitized, [])
                        if mob_id in knowledge_base.loaded_mobs
                    ]

                if not mob_ids:
                    logging.warn('Mob %s could not be found, skipping ...', k)
                    continue

                mob_id = mob_ids[0]
                if len(mob_ids) > 1:
                    logging.warn('Mob name %s refers to mobs %s, using %s ...',
                                 k, mob_ids, mob_id)

            mob = knowledge_base.npc_map.get(mob_id)

            if mob is None and mob_id != OTHER_STANDARD_ID:
                logging.warn('Mob %s could not be found, skipping ...', k)
                continue

            v_fixed = cls.fix_number(v)
            if v_fixed is None:
                logging.warn('Given value %s for mob %s will be skipped ...',
                             v, mob_id)
                continue

            new_map[mob_id] = v_fixed

        return new_map if new_map else None

    @classmethod
    def fix_event_value_till_item(
        cls,
        value: Any,
        knowledge_base: KnowledgeBase,
        **kwargs,
    ) -> Optional[Dict[int, float]]:

        if not isinstance(value, dict):
            logging.warn(
                'Value provided does not have key value pairs inside: %s, skipping ...',
                value
            )
            return None

        new_map = {}

        for k, v in value.items():
            try:
                event_id = int(k)
            except ValueError:
                if not isinstance(k, str):
                    logging.warn('Invalid type for event %s, skipping ...', k)
                    continue

                k_sanitized = sanitize_key(k)

                if k_sanitized in OTHER_KEYWORDS:
                    event_id = OTHER_STANDARD_ID
                else:
                    event_id = knowledge_base.event_name_id_map.get(k_sanitized)

                if event_id is None:
                    logging.warn('Event %s could not be found, skipping ...', k)
                    continue

            v_fixed = cls.fix_number(v)
            if v_fixed is None:
                logging.warn('Given value %s for event %s will be skipped ...',
                             v, event_id)
                continue

            new_map[event_id] = v_fixed

        return new_map if new_map else None

    @classmethod
    def fix_kills_till_item(
        cls,
        value: Any,
        name: str,
        **kwargs,
    ) -> Optional[float]:
        value_fixed = cls.fix_number(value)
        if value_fixed is None:
            logging.warn('Given value %s for item %s will be skipped ...',
                         value, name)
        return value_fixed

    @classmethod
    def fix_kills_till_item_per_mob(
        cls,
        value: Any,
        knowledge_base: KnowledgeBase,
        **kwargs,
    ) ->  Optional[Dict[int, float]]:
        return cls.fix_value_till_item_per_mob(
            value, knowledge_base=knowledge_base, **kwargs)

    @classmethod
    def fix_crates_till_item(
        cls,
        value: Any,
        name: str,
        **kwargs,
    ) -> Optional[float]:
        value_fixed = cls.fix_number(value)
        if value_fixed is None:
            logging.warn('Given value %s for item %s will be skipped ...',
                         value, name)
        return value_fixed

    @classmethod
    def fix_crates_till_item_per_mob(
        cls,
        value: Any,
        knowledge_base: KnowledgeBase,
        **kwargs,
    ) ->  Optional[Dict[int, float]]:
        return cls.fix_value_till_item_per_mob(
            value, knowledge_base=knowledge_base, **kwargs)

    @classmethod
    def fix_crates_till_item_per_crate_type(
        cls,
        value: Any,
        knowledge_base: KnowledgeBase,
        **kwargs,
    ) ->  Optional[Dict[str, float]]:

        if not isinstance(value, dict):
            logging.warn(
                'Value provided does not have key value pairs inside: %s, skipping ...',
                value
            )
            return None

        new_map = {}

        for k, v in value.items():
            if not isinstance(k, str):
                logging.warn('Invalid type for CRATE type %s, skipping ...', k)
                continue

            k_sanitized = sanitize_key(k).replace('crate', '').strip()

            if k_sanitized in OTHER_KEYWORDS:
                crate_name = OTHER_STANDARD_KEYWORD
            else:
                crate_name = knowledge_base.crate_name_map.get(k_sanitized)

            if crate_name is None:
                logging.warn('CRATE type %s could not be found, skipping ...', k)
                continue

            v_fixed = cls.fix_number(v)
            if v_fixed is None:
                logging.warn('Given value %s for %s CRATE will be skipped ...',
                             v, crate_name)
                continue

            new_map[crate_name] = v_fixed

        return new_map if new_map else None

    @classmethod
    def fix_crates_till_item_per_mob_and_crate_type(
        cls,
        value: Any,
        knowledge_base: KnowledgeBase,
        **kwargs,
    ) -> Optional[Dict[int, Dict[str, float]]]:

        if not isinstance(value, dict):
            logging.warn(
                'Value provided does not have key value pairs inside: %s, skipping ...',
                value
            )
            return None

        new_map = {}

        for k, v in value.items():
            try:
                mob_id = int(k)
            except ValueError:
                if not isinstance(k, str):
                    logging.warn('Invalid type for mob %s, skipping ...', k)
                    continue

                k_sanitized = sanitize_key(k)

                if k_sanitized in OTHER_KEYWORDS:
                    mob_ids = [OTHER_STANDARD_ID]
                else:
                    mob_ids = [
                        mob_id
                        for mob_id in knowledge_base.npc_name_map.get(k_sanitized, [])
                        if mob_id in knowledge_base.loaded_mobs
                    ]

                if not mob_ids:
                    logging.warn('Mob %s could not be found, skipping ...', k)
                    continue

                mob_id = mob_ids[0]
                if len(mob_ids) > 1:
                    logging.warn('Mob name %s refers to mobs %s, using %s ...',
                                 k, mob_ids, mob_id)

            mob = knowledge_base.npc_map.get(mob_id)

            if mob is None:
                logging.warn('Mob %s could not be found, skipping ...', k)
                continue

            if not isinstance(v, dict):
                logging.warn(
                    'Value %s should specify CRATE types for mob %s, skipping ...',
                    v, k
                )
                continue

            inner_map = cls.fix_crates_till_item_per_crate_type(
                v, knowledge_base=knowledge_base, **kwargs)

            if not inner_map:
                logging.warn('No valid requirements for mob %s, skipping ...', k)
                continue

            new_map[mob_id] = inner_map

        return new_map if new_map else None

    @classmethod
    def fix_crates_till_item_per_crate_type_and_mob(
        cls,
        value: Any,
        knowledge_base: KnowledgeBase,
        **kwargs,
    ) -> Optional[Dict[str, Dict[int, float]]]:

        if not isinstance(value, dict):
            logging.warn(
                'Value provided does not have key value pairs inside: %s, skipping ...',
                value
            )
            return None

        new_map = {}

        for k, v in value.items():
            if not isinstance(k, str):
                logging.warn('Invalid type for CRATE type %s, skipping ...', k)
                continue

            k_sanitized = sanitize_key(k).replace('crate', '').strip()

            if k_sanitized in OTHER_KEYWORDS:
                crate_name = OTHER_STANDARD_KEYWORD
            else:
                crate_name = knowledge_base.crate_name_map.get(k_sanitized)

            if crate_name is None:
                logging.warn('CRATE type %s could not be found, skipping ...', k)
                continue

            if not isinstance(v, dict):
                logging.warn(
                    'Value %s should specify mobs for CRATE type %s, skipping ...',
                    v, k
                )
                continue

            inner_map = cls.fix_crates_till_item_per_mob(
                v, knowledge_base=knowledge_base, **kwargs)

            if not inner_map:
                logging.warn('No valid requirements for CRATE type %s, skipping ...',
                             k)
                continue

            new_map[crate_name] = inner_map

        return new_map if new_map else None

    @classmethod
    def fix_event_kills_till_item(
        cls,
        value: Any,
        knowledge_base: KnowledgeBase,
        **kwargs,
    ) -> Optional[Dict[int, float]]:
        return cls.fix_event_value_till_item(
            value, knowledge_base=knowledge_base, **kwargs)

    @classmethod
    def fix_event_crates_till_item(
        cls,
        value: Any,
        knowledge_base: KnowledgeBase,
        **kwargs,
    ) -> Optional[Dict[int, float]]:
        return cls.fix_event_value_till_item(
            value, knowledge_base=knowledge_base, **kwargs)

    @classmethod
    def fix_race_crates_till_item(
        cls,
        value: Any,
        knowledge_base: KnowledgeBase,
        **kwargs,
    ) -> Optional[Dict[int, Dict[str, float]]]:

        if not isinstance(value, dict):
            logging.warn(
                'Value provided does not have key value pairs inside: %s, skipping ...',
                value
            )
            return None

        new_map = {}

        for k, v in value.items():
            try:
                epid = int(k)
            except ValueError:
                if not isinstance(k, str):
                    logging.warn('Invalid type for IZ %s, skipping ...', k)
                    continue

                k_sanitized = sanitize_key(k).replace('the ', '').replace('uture', '')

                if k_sanitized in OTHER_KEYWORDS:
                    epid = OTHER_STANDARD_ID
                else:
                    epid = knowledge_base.iz_name_id_map.get(k_sanitized)

                if epid is None:
                    logging.warn('IZ %s could not be found, skipping ...', k)
                    continue

            if not isinstance(v, dict):
                logging.warn(
                    'Value %s should specify mobs for IZ %s, skipping ...',
                    v, k
                )
                continue

            inner_map = cls.fix_crates_till_item_per_crate_type(
                v, knowledge_base=knowledge_base, **kwargs)

            if not inner_map:
                logging.warn('No valid requirements for IZ %s, skipping ...', k)
                continue

            new_map[epid] = inner_map

        return new_map if new_map else None

    @classmethod
    def fix_mission_crates_till_item(
        cls,
        value: Any,
        knowledge_base: KnowledgeBase,
        **kwargs,
    ) -> Optional[Dict[int, float]]:

        if not isinstance(value, dict):
            logging.warn(
                'Value provided does not have key value pairs inside: %s, skipping ...',
                value
            )
            return None

        new_map = {}

        for k, v in value.items():
            try:
                level = int(k)
            except ValueError:
                if not isinstance(k, str):
                    logging.warn('Invalid type for level %s, skipping ...', k)
                    continue

                k_sanitized = sanitize_key(k).replace('level', '').strip()

                if k_sanitized in OTHER_KEYWORDS:
                    level = OTHER_STANDARD_ID
                else:
                    try:
                        level = int(k_sanitized)
                    except ValueError:
                        logging.warn('Level %s could not be parsed, skipping ...', k)
                        continue

            if level != OTHER_STANDARD_ID and level not in knowledge_base.mission_level_crate_id_map:
                logging.warn('Level %s gives no mission CRATEs, skipping ...', k)
                continue

            v_fixed = cls.fix_number(v)
            if v_fixed is None:
                logging.warn('Given value %s for level %s will be skipped ...',
                             v, k)
                continue

            new_map[level] = v_fixed

        return new_map if new_map else None


    @classmethod
    def fix_egger_eggs_till_item(
        cls,
        value: Any,
        knowledge_base: KnowledgeBase,
        **kwargs,
    ) -> Optional[Dict[str, Dict[str, float]]]:

        if not isinstance(value, dict):
            logging.warn(
                'Value provided does not have key value pairs inside: %s, skipping ...',
                value
            )
            return None

        new_map = {}

        for k, v in value.items():
            if not isinstance(k, str):
                logging.warn('Invalid type for zone %s, skipping ...', k)
                continue

            k_sanitized = sanitize_key(k).replace('the ', '')

            if k_sanitized in OTHER_KEYWORDS:
                zone_name = OTHER_STANDARD_KEYWORD
            else:
                zone_name = k_sanitized

            if zone_name not in knowledge_base.zone_to_egger_map:
                logging.warn('Zone %s could not be found, skipping ...', k)
                continue

            if not isinstance(v, dict):
                logging.warn(
                    'Value %s should specify mobs for zone %s, skipping ...',
                    v, k
                )
                continue

            inner_map = cls.fix_crates_till_item_per_crate_type(
                v, knowledge_base=knowledge_base, **kwargs)

            if not inner_map:
                logging.warn('No valid requirements for EGGER %s, skipping ...', k)
                continue

            new_map[zone_name] = inner_map

        return new_map if new_map else None

    @classmethod
    def fix_golden_eggs_till_item(
        cls,
        value: Any,
        knowledge_base: KnowledgeBase,
        **kwargs,
    ) -> Optional[Dict[str, float]]:

        if not isinstance(value, dict):
            logging.warn(
                'Value provided does not have key value pairs inside: %s, skipping ...',
                value
            )
            return None

        new_map = {}

        for k, v in value.items():
            if not isinstance(k, str):
                logging.warn('Invalid type for area %s, skipping ...', k)
                continue

            k_sanitized = sanitize_key(k).replace('the ', '').replace('uture', '')

            if k_sanitized in OTHER_KEYWORDS:
                area_name = OTHER_STANDARD_KEYWORD
            else:
                area_name = k_sanitized

            if area_name not in knowledge_base.name_to_areas:
                logging.warn('Area %s could not be found, skipping ...', k)
                continue

            v_fixed = cls.fix_number(v)
            if v_fixed is None:
                logging.warn('Given value %s for area %s will be skipped ...',
                             v, k)
                continue

            new_map[area_name] = v_fixed

        return new_map if new_map else None

    @classmethod
    def fix_named_crates_till_item(
        cls,
        value: Any,
        knowledge_base: KnowledgeBase,
        **kwargs,
    ) -> Optional[Dict[int, float]]:

        if not isinstance(value, dict):
            logging.warn(
                'Value provided does not have key value pairs inside: %s, skipping ...',
                value
            )
            return None

        new_map = {}

        for k, v in value.items():
            try:
                crate_id = int(k)
            except ValueError:
                if not isinstance(k, str):
                    logging.warn('Invalid type for CRATE name %s, skipping ...', k)
                    continue

                k_sanitized = sanitize_key(k)

                if k_sanitized in OTHER_KEYWORDS:
                    logging.warn('Keyword %s is not allowed for named CRATEs, skipping ...', k)
                    continue

                crate_ids = [
                    item_id
                    for item_type, item_id in knowledge_base.item_name_map.get(k_sanitized, [])
                    if item_type == 9
                ]

                if not crate_ids:
                    logging.warn('CRATE name %s could not be found, skipping ...', k)
                    continue

                crate_id = crate_ids[0]
                if len(crate_ids) > 1:
                    logging.warn('CRATE name %s refers to CRATEs %s, using %s ...',
                                 k, crate_ids, crate_id)

            v_fixed = cls.fix_number(v)
            if v_fixed is None:
                logging.warn('Given value %s for CRATE ID %s will be skipped ...',
                             v, crate_id)
                continue

            new_map[crate_id] = v_fixed

        return new_map if new_map else None
