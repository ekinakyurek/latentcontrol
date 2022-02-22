import os
from typing import Mapping, List
from utils import set_seed
from dataclasses import dataclass

@dataclass
class Animal:
    name: str
    family: str
    
    def __str__(self):
        return self.name

@dataclass
class Universe:
    name: str
    relations: Mapping
    
    def __str__(self):
        return self.name
    
@dataclass
class KBEntry:
    universe: Universe
    animal1: Animal
    animal2: Animal
    relation: str
    
    def __str__(self):
        return f"In {self.universe} , {self.animal1} {self.relation} {self.animal2} ."
    

def generate_KB(animals: List[Animal], universes: List[Universe]) -> List[KBEntry]:
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
    
def write_KB(KB: List[KBEntry], corpus_folder: str = "datasets/SimpleKB/"):
    os.makedirs(corpus_folder, exist_ok=True)
    with open(os.path.join(corpus_folder, "corpus.txt"), "w") as f:
        for k in KB:
            print(k, file=f)


if __name__ == "__main__":
    set_seed(0)
    
    animals = [Animal("dogs", "mamal"), 
               Animal("cats", "mamal"),
               Animal("frogs", "amphibian"),
               Animal("salamender", "amphibian"),
               Animal("snake", "reptile"),
               Animal("crocodile", "reptile"),
              ]

    
    universes = [Universe("Earth",
                     {"mamal": {"likes": ("mamal",), "dislikes": ("reptile", "amphibian")},
                      "reptile": {"likes": ("amphibian", "reptile"), "dislikes": ("mamal")},
                      "amphibian": {"likes": ("amphibian", "reptile"), "dislikes": ("mamal")},
                     }),
                Universe("Metaverse",
                     {"mamal": {"likes": ("reptile",), "dislikes": ("mamal", "amphibian")},
                      "reptile": {"likes": ("mamal", "reptile"), "dislikes": ("amphibian")},
                      "amphibian": {"likes": ("reptile", ), "dislikes": ("mamal", "amphibian")},
                     }) 
                ] 

    write_KB(generate_KB(animals, universes))   