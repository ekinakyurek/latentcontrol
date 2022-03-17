import os
from typing import Mapping, List
from src.utils import set_seed
from dataclasses import dataclass


@dataclass
class Animal:
    """Represents animals with their name and family"""
    name: str
    family: str

    def __str__(self):
        return self.name


@dataclass
class Universe:
    """Represents animals with their name and relations"""
    name: str
    relations: Mapping

    def __str__(self):
        return self.name


@dataclass
class KBEntry:
    """Knowledge base entry"""
    universe: Universe
    animal1: Animal
    animal2: Animal
    relation: str

    def __str__(self):
        return f"In {self.universe} , {self.animal1} {self.relation} {self.animal2} ."


def generate_KB(animals: List[Animal], universes: List[Universe]) -> List[KBEntry]:
    """Generates knowledge base"""
    KB = []
    for universe in universes:
        for animal1 in animals:
            for animal2 in animals:
                relations = universe.relations[animal1.family]
                if animal2.family in relations["likes"]:
                    KB.append(KBEntry(universe, animal1, animal2, "likes"))
                else:
                    KB.append(KBEntry(universe, animal1, animal2, "dislikes"))
    return KB


def write_KB(KB: List[KBEntry],
             file_name: str = "corpus.txt",
             corpus_folder: str = "datasets/SimpleKB/"):
    """Writes knowledge base to a text file"""
    os.makedirs(corpus_folder, exist_ok=True)
    with open(os.path.join(corpus_folder, file_name), "w") as f:
        for k in KB:
            print(k, file=f)


def main():

    set_seed(0)

    animals = [Animal("dogs", "mamal"),
               Animal("cats", "mamal"),
               Animal("frogs", "amphibian"),
               Animal("salamenders", "amphibian"),
               Animal("snakes", "reptile"),
               Animal("crocodiles", "reptile")]

    universes = [Universe("Earth",
                    {"mamal": {"likes": ("mamal",), "dislikes": ("reptile", "amphibian")},
                     "reptile": {"likes": ("amphibian", "reptile"), "dislikes": ("mamal")},
                     "amphibian": {"likes": ("amphibian", "reptile"), "dislikes": ("mamal")},
                    }),
                Universe("Metaverse",
                     {"mamal": {"likes": ("reptile",), "dislikes": ("mamal", "amphibian")},
                      "reptile": {"likes": ("mamal", "reptile"), "dislikes": ("amphibian")},
                      "amphibian": {"likes": ("reptile", ), "dislikes": ("mamal", "amphibian")},
                     })]

    write_KB(generate_KB(animals, universes))


if __name__ == "__main__":
    main()