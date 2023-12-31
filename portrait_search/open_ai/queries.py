PORTRAIT_DESCRIPTION_QUERY_V1 = """Describe what is in a picture. Assume that it's a character for a table-top RPG game.

Specifically mention its gender, race, alignment, possible classes, and subclasses, weapon it's holding, armor, and any special traits.

Output a natural text description, and after two line breaks: a JSON formatted response like that:

{natural description}

{
    "race": "aasimar",
    "gender": "female",
    "classes": ["cleric", "paladin"],
    "weapon": "quarterstaff",
    "armor": "cloth",
}"""  # noqa: E501
