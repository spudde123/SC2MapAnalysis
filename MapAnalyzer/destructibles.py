"""
https://github.com/DrInfy/sharpy-sc2/blob/develop/sharpy/managers/unit_value.py
"""
from sc2 import UnitTypeId

buildings_2x2 = {
        UnitTypeId.SUPPLYDEPOT,
        UnitTypeId.PYLON,
        UnitTypeId.DARKSHRINE,
        UnitTypeId.PHOTONCANNON,
        UnitTypeId.SHIELDBATTERY,
        UnitTypeId.TECHLAB,
        UnitTypeId.STARPORTTECHLAB,
        UnitTypeId.FACTORYTECHLAB,
        UnitTypeId.BARRACKSTECHLAB,
        UnitTypeId.REACTOR,
        UnitTypeId.STARPORTREACTOR,
        UnitTypeId.FACTORYREACTOR,
        UnitTypeId.BARRACKSREACTOR,
        UnitTypeId.MISSILETURRET,
        UnitTypeId.SPORECRAWLER,
        UnitTypeId.SPIRE,
        UnitTypeId.GREATERSPIRE,
        UnitTypeId.SPINECRAWLER,
}

buildings_3x3 = {
        UnitTypeId.GATEWAY,
        UnitTypeId.WARPGATE,
        UnitTypeId.CYBERNETICSCORE,
        UnitTypeId.FORGE,
        UnitTypeId.ROBOTICSFACILITY,
        UnitTypeId.ROBOTICSBAY,
        UnitTypeId.TEMPLARARCHIVE,
        UnitTypeId.TWILIGHTCOUNCIL,
        UnitTypeId.TEMPLARARCHIVE,
        UnitTypeId.STARGATE,
        UnitTypeId.FLEETBEACON,
        UnitTypeId.ASSIMILATOR,
        UnitTypeId.ASSIMILATORRICH,
        UnitTypeId.SPAWNINGPOOL,
        UnitTypeId.ROACHWARREN,
        UnitTypeId.HYDRALISKDEN,
        UnitTypeId.BANELINGNEST,
        UnitTypeId.EVOLUTIONCHAMBER,
        UnitTypeId.NYDUSNETWORK,
        UnitTypeId.NYDUSCANAL,
        UnitTypeId.EXTRACTOR,
        UnitTypeId.EXTRACTORRICH,
        UnitTypeId.INFESTATIONPIT,
        UnitTypeId.ULTRALISKCAVERN,
        UnitTypeId.BARRACKS,
        UnitTypeId.ENGINEERINGBAY,
        UnitTypeId.FACTORY,
        UnitTypeId.GHOSTACADEMY,
        UnitTypeId.STARPORT,
        UnitTypeId.FUSIONREACTOR,
        UnitTypeId.BUNKER,
        UnitTypeId.ARMORY,
        UnitTypeId.REFINERY,
        UnitTypeId.REFINERYRICH,
}

buildings_5x5 = {
        UnitTypeId.NEXUS,
        UnitTypeId.HATCHERY,
        UnitTypeId.HIVE,
        UnitTypeId.LAIR,
        UnitTypeId.COMMANDCENTER,
        UnitTypeId.ORBITALCOMMAND,
        UnitTypeId.PLANETARYFORTRESS,
}

BUILDING_IDS = buildings_5x5.union(buildings_3x3).union(buildings_2x2)


destructable_2x2 = {
    UnitTypeId.ROCKS2X2NONCONJOINED,
}

destructable_4x4 = {
    UnitTypeId.DESTRUCTIBLECITYDEBRIS4X4,
    UnitTypeId.DESTRUCTIBLEDEBRIS4X4,
    UnitTypeId.DESTRUCTIBLEICE4X4,
    UnitTypeId.DESTRUCTIBLEROCK4X4,
    UnitTypeId.DESTRUCTIBLEROCKEX14X4,
}

destructable_4x2 = {
    UnitTypeId.DESTRUCTIBLECITYDEBRIS2X4HORIZONTAL,
    UnitTypeId.DESTRUCTIBLEICE2X4HORIZONTAL,
    UnitTypeId.DESTRUCTIBLEROCK2X4HORIZONTAL,
    UnitTypeId.DESTRUCTIBLEROCKEX12X4HORIZONTAL,
}

destructable_2x4 = {
    UnitTypeId.DESTRUCTIBLECITYDEBRIS2X4VERTICAL,
    UnitTypeId.DESTRUCTIBLEICE2X4VERTICAL,
    UnitTypeId.DESTRUCTIBLEROCK2X4VERTICAL,
    UnitTypeId.DESTRUCTIBLEROCKEX12X4VERTICAL,
}

destructable_6x2 = {
    UnitTypeId.DESTRUCTIBLECITYDEBRIS2X6HORIZONTAL,
    UnitTypeId.DESTRUCTIBLEICE2X6HORIZONTAL,
    UnitTypeId.DESTRUCTIBLEROCK2X6HORIZONTAL,
    UnitTypeId.DESTRUCTIBLEROCKEX12X6HORIZONTAL,
}

destructable_2x6 = {
    UnitTypeId.DESTRUCTIBLECITYDEBRIS2X6VERTICAL,
    UnitTypeId.DESTRUCTIBLEICE2X6VERTICAL,
    UnitTypeId.DESTRUCTIBLEROCK2X6VERTICAL,
    UnitTypeId.DESTRUCTIBLEROCKEX12X6VERTICAL,
}

destructable_4x12 = {
    UnitTypeId.DESTRUCTIBLEROCKEX1VERTICALHUGE,
    UnitTypeId.DESTRUCTIBLEICEVERTICALHUGE
}

destructable_12x4 = {
    UnitTypeId.DESTRUCTIBLEROCKEX1HORIZONTALHUGE,
    UnitTypeId.DESTRUCTIBLEICEHORIZONTALHUGE
}

destructable_6x6 = {
    UnitTypeId.DESTRUCTIBLECITYDEBRIS6X6,
    UnitTypeId.DESTRUCTIBLEDEBRIS6X6,
    UnitTypeId.DESTRUCTIBLEICE6X6,
    UnitTypeId.DESTRUCTIBLEROCK6X6,
    UnitTypeId.DESTRUCTIBLEROCKEX16X6,
}

destructable_BLUR = {
    UnitTypeId.DESTRUCTIBLECITYDEBRISHUGEDIAGONALBLUR,
    UnitTypeId.DESTRUCTIBLEDEBRISRAMPDIAGONALHUGEBLUR,
    UnitTypeId.DESTRUCTIBLEICEDIAGONALHUGEBLUR,
    UnitTypeId.DESTRUCTIBLEROCKEX1DIAGONALHUGEBLUR,
    UnitTypeId.DESTRUCTIBLERAMPDIAGONALHUGEBLUR,
}

destructable_ULBR = {
    UnitTypeId.DESTRUCTIBLECITYDEBRISHUGEDIAGONALULBR,
    UnitTypeId.DESTRUCTIBLEDEBRISRAMPDIAGONALHUGEULBR,
    UnitTypeId.DESTRUCTIBLEICEDIAGONALHUGEULBR,
    UnitTypeId.DESTRUCTIBLEROCKEX1DIAGONALHUGEULBR,
    UnitTypeId.DESTRUCTIBLERAMPDIAGONALHUGEULBR,
}
