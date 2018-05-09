
class Card:

    CHARACTER = ['Druid', 'Hunter', 'Mage', 'Neutral', 'Paladin', 'Priest', 'Rogue', 'Shaman', 'Warlock', 'Warrior']
    RARITY = ['Free', 'Common', 'Rare', 'Epic', 'Legendary']
    TYPE = ['Enchantment', 'Hero', 'Minion', 'Spell', 'Weapon']

    name = ''
    effect = ''
    character = ''
    rarity = ''
    type = ''
    mana = ''
    attack = ''
    health = ''
    armour = ''

    def __init__(self, name, character, rarity, type, mana):
        if character not in self.CHARACTER:
            raise ValueError("Character not valid: " + str(character))
        if rarity not in self.RARITY:
            raise ValueError("Rarity not valid: " + str(rarity))
        if type not in self.TYPE:
            raise ValueError("Type not valid: " + str(type))
        self.name = name
        self.character = character
        self.rarity = rarity
        self.type = type
        self.mana = mana

    def __repr__(self):
        return 'Card(name=%s, effect=%s, character=%s, rarity=%s, type=%s, mana=%s, attack=%s, health=%s, armour=%s)' \
               % (self.name, self.effect, self.character, self.rarity, self.type, self.mana, self.attack, self.health, self.armour)

    def as_dict(self):
        return {
            'name': self.name,
            'effect': self.effect,
            'character': self.character,
            'rarity': self.rarity,
            'type': self.type,
            'mana': self.mana,
            'attack': self.attack,
            'health': self.health,
            'armour': self.armour,
        }

    def as_arr(self):
        return [
            self.name,
            self.effect,
            self.character,
            self.rarity,
            self.type,
            self.mana,
            self.attack,
            self.health,
            self.armour,
        ]
