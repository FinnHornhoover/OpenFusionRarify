import json
from pathlib import Path
from functools import reduce
from operator import itemgetter
from collections import defaultdict
from typing import Any, Dict, DefaultDict, List, Iterator, Iterable, Set, Tuple, Union


int_key_map = {
    'Racing': 'EPID',
    'NanoCapsules': 'Nano',
    'CodeItems': 'Code',
}

int_lower_bound_map = {
    'Crates': 0,
}

fk_map_names = {
    'Rewards': 'Crates',
}

foreign_key_map = {
    'CrateDropTypes': [
        'CrateIDs',
    ],
    'MobDrops': [
        'CrateDropChanceID',
        'CrateDropTypeID',
        'MiscDropChanceID',
        'MiscDropTypeID',
    ],
    'Events': [
        'MobDropID',
    ],
    'Mobs': [
        'MobDropID',
    ],
    'ItemSets': [
        'ItemReferenceIDs',
    ],
    'Crates': [
        'ItemSetID',
        'RarityWeightID',
    ],
    'Racing': [
        'Rewards',
    ],
    'NanoCapsules': [
        'CrateID',
    ],
    'CodeItems': [
        'ItemReferenceIDs',
    ],
}


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


def generate_patch(
    base_obj: Dict[str, Any],
    updated_obj: Dict[str, Any],
) -> Dict[str, Any]:
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


class Data(dict):
    def __init__(
        self,
        alt_dict: 'AlternateDict',
        str_key: str,
        fk_names: List[str],
        value: Dict[str, Any],
    ) -> None:
        super().__init__(value)
        self.alt_dict = alt_dict
        self.str_key = str_key
        self.fk_names = fk_names

    @property
    def main_key(self) -> str:
        return self.alt_dict.main_key

    @property
    def int_key_name(self) -> str:
        return self.alt_dict.int_key_name

    @property
    def references(self) -> Dict[Tuple[str, int], Set[Tuple[str, int]]]:
        return self.alt_dict.drops.references

    @property
    def int_key(self) -> int:
        return self[self.int_key_name]

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self.fk_names:
            self_tuple = (self.main_key, self.int_key)
            related_key = fk_map_names.get(key, key.split('ID')[0] + 's')
            related_value = self[key]

            related_values = [related_value] if isinstance(related_value, int) else related_value
            altered_values = [value] if isinstance(related_value, int) else value

            for related_value, altered_value in zip(related_values, altered_values):
                self.references[(related_key, related_value)].remove(self_tuple)
                self.references[(related_key, altered_value)].add(self_tuple)

        super().__setitem__(key, value)

    def __delitem__(self, key: Tuple[str, int]) -> None:
        if isinstance(key, str):
            raise ValueError('Cannot delete entire field for a Data object!')

        field_key, related_value = key
        if field_key in self.fk_names:
            self_tuple = (self.main_key, self.int_key)
            related_key = fk_map_names.get(key, key.split('ID')[0] + 's')
            self.references[(related_key, related_value)].remove(self_tuple)

        self[field_key].remove(related_value)


class AlternateDict(dict):
    def __init__(
        self,
        drops: 'Drops',
        main_key: str,
        main_obj: Dict[str, Dict[str, Any]],
    ) -> None:
        self.drops = drops
        self.main_key = main_key

        self.lowest_id = int_lower_bound_map.get(self.main_key, -1)
        self.int_key_name = int_key_map.get(self.main_key,
                                            self.main_key[:-1] + 'ID')

        super().__init__({
            str_key: Data(
                alt_dict=self,
                str_key=str_key,
                fk_names=foreign_key_map.get(self.main_key, []),
                value=value,
            )
            for str_key, value in main_obj.items()
        })

        self.int_to_str_keys = {
            value[self.int_key_name]: str_key
            for str_key, value in main_obj.items()
        }

    def __getitem__(self, key: int) -> Dict[str, Any]:
        str_key = self.int_to_str_keys[key]
        return super().__getitem__(str_key)

    def __setitem__(self, key: int, value: Dict[str, Any]) -> None:
        if key in self.int_to_str_keys:
            str_key = self.int_to_str_keys[key]
        else:
            key = self.next_int_id()
            str_key = self.next_str_id()

        int_key = value[self.int_key_name]

        if int_key != key:
            value[self.int_key_name] = key
            int_key = key

        self.int_to_str_keys[int_key] = str_key
        super().__setitem__(str_key, value)

    def __delitem__(self, key: int) -> None:
        str_key = self.int_to_str_keys[key]
        del self.int_to_str_keys[key]
        super().__delitem__(str_key)

    def __contains__(self, key: int) -> None:
        return key in self.int_to_str_keys

    def __iter__(self) -> Iterator[int]:
        return iter(self.int_to_str_keys)

    def ikeys(self) -> Iterable[int]:
        return self.int_to_str_keys.keys()

    def iitems(self) -> Iterable[Tuple[int, Dict[str, Any]]]:
        return zip(self.ikeys(), self.values())

    def next_int_id(self) -> int:
        return reduce(
            max,
            map(itemgetter(self.int_key_name), self.values()),
            self.lowest_id,
        ) + 1

    def next_str_id(self) -> str:
        return str(reduce(max, map(int, self.keys()), -1) + 1)

    def add(self, value: Dict[str, Any]) -> int:
        self[self.lowest_id - 1] = value
        return value[self.int_key_name]


class Drops(dict):
    def __init__(self, drops_path: Path, patch_paths: List[Path]) -> None:
        super().__init__({
            key: AlternateDict(self, key, obj)
            for key, obj in get_patched(drops_path, patch_paths).items()
        })

        self.references: DefaultDict[Tuple[str, int], Set[Tuple[str, int]]] = defaultdict(set)
        for alt_dict in self.values():
            for int_key, data in alt_dict.iitems():
                for fk_type in data.fk_names:
                    fk_list = (
                        [data[fk_type]]
                        if isinstance(data[fk_type], int)
                        else data[fk_type]
                    )

                    for fk_id in fk_list:
                        if fk_id <= alt_dict.lowest_id:
                            continue

                        self.references[(data.main_key, int_key)].add((
                            fk_map_names.get(fk_type, fk_type.split('ID')[0] + 's'),
                            fk_id))

    def __getitem__(self, key: Union[str, Tuple[str, int]]) -> Union[AlternateDict, Data]:
        if isinstance(key, tuple):
            return self[key[0]][key[1]]
        return super().__getitem__(key)

    def __setitem__(
        self,
        key: Union[str, Tuple[str, int]],
        value: Union[Data, AlternateDict],
    ) -> None:
        if isinstance(key, tuple):
            self[key[0]][key[1]] = value
            return
        super().__setitem__(key, value)

    def __delitem__(self, key: Union[str, Tuple[str, int]]) -> None:
        if isinstance(key, tuple):
            del self[key[0]][key[1]]
            return
        super().__delitem__(key)
