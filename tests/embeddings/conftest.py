import pytest


@pytest.fixture
def portrait_description_example() -> str:
    return """The character depicted in the image appears to be a female elf, given the pointed ears and slender build. She possesses a demeanor that suggests a neutral alignment, focused more on balance or personal goals than strict adherence to good or evil, lawful or chaotic behaviors. The most fitting classes for her would likely be those associated with magic and nature, such as a druid or a ranger, with possible subclasses like Circle of the Moon (for a druid) or Beast Master (for a ranger). She holds a magical staff, which is entwined with living wood and leaves, indicating a connection to nature, and possibly serving as a focus for her spellcasting. The character wears simple, perhaps leather-based armor, which offers protection while still allowing ease of movement through natural terrains. A notable special trait is the ethereal green flame emanating from her extended hand, hinting at her ability to harness elemental magic or cast spells.

{
    "race": "elf",
    "gender": "female",
    "alignment": "neutral",
    "classes": ["druid", "ranger"],
    "subclasses": ["Circle of the Moon", "Beast Master"],
    "weapon": "magical staff",
    "armor": "leather",
    "special_traits": ["elemental magic", "spellcasting"]
}"""  # noqa: E501
