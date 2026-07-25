from __future__ import annotations

import re
from collections import Counter, defaultdict


# Source: species_list.xlsx, first worksheet named "Master Sheet", row 1,
# columns H:DB. Names intentionally preserve the workbook's spelling and
# abbreviations so historical EventMeasure records remain searchable.
MASTER_SPECIES_NAMES = (
    "Parma polylepis",
    "Pseudolabrus luculentus",
    "Neo. polyacanthus",
    "Plectro. gascoynei",
    "Ostorhinchus norfolcensis",
    "Pseudocheilinus hexataenia",
    "Chlorurus spilurus",
    "Thal. purpureum",
    "Thal. lunare",
    "Thal. amblycephalum",
    "Thal. lutescens",
    "Thal. hardwicke",
    "Thal. jansenii",
    "Gomphosus varius",
    "Anampses elegans",
    "Hal. marginatus",
    "Hal. margaritifer",
    "Hal. trimaculatus",
    "Hal. nebulosus",
    "Notolabrus inscriptus",
    "Cheilio inermis",
    "Stethojulis bandanensis",
    "Coris sandeyeri",
    "Coris bulbifrons",
    "Plagiotremus tapenisoma",
    "Chaet. vagabundus",
    "Chaet. auriga",
    "Chaet. plebeius",
    "Chaet. melannotus",
    "Chaet. trifascialis",
    "Chaet. pelewensis",
    "Chaet. speculum",
    "Chaet. lineolatus",
    "Chaet. lunula",
    "Chaet. flavirostris",
    "Chaet. tricinctus",
    "Chaet. lunulatus",
    "Chaet. citrinellus",
    "Chaet. mertensii",
    "Chaet. bennettii",
    "Heniochus chrysostomus",
    "Heniochus monoceros",
    "Trachinotus blochii",
    "Plectro. fasciolatus",
    "Pseudocaranx sp.",
    "Caranx sexfasciatus",
    "Abu. vaigiensis",
    "Abu. sexfasciatus",
    "Abu. sordidus",
    "Abu. septemfasciatus",
    "Naso unicornis",
    "Acanthurus dussumeieri",
    "Zanclus cornutus",
    "Prionurus maculatus",
    "Parupeneus cyclostomus",
    "Chromis norfolkensis",
    "Diodon hystrix",
    "Cirrpectes sp.",
    "Trachypoma macacanthus",
    "Chromis margaritifer",
    "Plectro. dickii",
    "Plectro. johnstonianus",
    "Fistularia commersonnii",
    "Aplodactylus etheridgeii",
    "Synodus dermatogenys",
    "Girella cyanea",
    "Crenimugil crenilabis",
    "Mulloidicchthys flavolineatus",
    "Myxus elongatus",
    "Mugil cephalus",
    "Chrysiptera notialis",
    "Kyphosus vaigiensis",
    "Epinephelus rivulatus",
    "Bothus pantherinus",
    "Pagrus auratus",
    "Parupeneus ciliatus",
    "Parupeneus spilurus",
    "Eviota hoesei",
    "Leiuranus semicinctus",
    "Gymnothorax eurostus",
    "Taeniamia leai",
    "Sargocentron punctatissimum",
    "Microcanthus joyceae",
    "Sphyraena acutipinnis",
    "Acanthistius cinctus",
    "Gymnothorax annasona",
    "Sargocentron rubrum",
    "Labroides dimidiatus",
    "Gymnothorax nubilis",
    "Cymolutes praetextatus",
    "Pervagor alternans",
    "Epinephelus daemelli",
    "Cheilodactylus ephippium",
    "Pomacentrus pavo",
    "Myliobatis tenuicaudatus",
    "Goniistius francisi",
    "Seriola lalandi",
    "Bodianus axillaris",
    "Ostorhinchus doederleini",
)

# English names primarily follow the CSIRO Codes for Australian Aquatic Biota
# (CAAB) and Australian Fish Names Standard. FishBase is used where CAAB has no
# vernacular name. Keys intentionally remain the workbook labels above,
# including historical abbreviations and misspellings, so existing annotations
# and generated species codes remain compatible.
MASTER_SPECIES_COMMON_NAMES = {
    "Parma polylepis": "Banded Scalyfin",
    "Pseudolabrus luculentus": "Luculent Wrasse",
    "Neo. polyacanthus": "Multispine Damsel",
    "Plectro. gascoynei": "Coral Sea Gregory",
    "Ostorhinchus norfolcensis": "Norfolk Cardinalfish",
    "Pseudocheilinus hexataenia": "Sixline Wrasse",
    "Chlorurus spilurus": "Pacific Bullethead Parrotfish",
    "Thal. purpureum": "Surge Wrasse",
    "Thal. lunare": "Moon Wrasse",
    "Thal. amblycephalum": "Bluehead Wrasse",
    "Thal. lutescens": "Green Moon Wrasse",
    "Thal. hardwicke": "Sixbar Wrasse",
    "Thal. jansenii": "Jansen's Wrasse",
    "Gomphosus varius": "Birdnose Wrasse",
    "Anampses elegans": "Elegant Wrasse",
    "Hal. marginatus": "Dusky Wrasse",
    "Hal. margaritifer": "Pearly Wrasse",
    "Hal. trimaculatus": "Threespot Wrasse",
    "Hal. nebulosus": "Cloud Wrasse",
    "Notolabrus inscriptus": "Inscribed Wrasse",
    "Cheilio inermis": "Sharpnose Wrasse",
    "Stethojulis bandanensis": "Redspot Wrasse",
    "Coris sandeyeri": "Eastern King Wrasse",
    "Coris bulbifrons": "Doubleheader",
    "Plagiotremus tapenisoma": "Piano Fangblenny",
    "Chaet. vagabundus": "Vagabond Butterflyfish",
    "Chaet. auriga": "Threadfin Butterflyfish",
    "Chaet. plebeius": "Bluespot Butterflyfish",
    "Chaet. melannotus": "Blackback Butterflyfish",
    "Chaet. trifascialis": "Chevron Butterflyfish",
    "Chaet. pelewensis": "Dot-and-dash Butterflyfish",
    "Chaet. speculum": "Ovalspot Butterflyfish",
    "Chaet. lineolatus": "Lined Butterflyfish",
    "Chaet. lunula": "Racoon Butterflyfish",
    "Chaet. flavirostris": "Dusky Butterflyfish",
    "Chaet. tricinctus": "Threeband Butterflyfish",
    "Chaet. lunulatus": "Pinstripe Butterflyfish",
    "Chaet. citrinellus": "Citron Butterflyfish",
    "Chaet. mertensii": "Mertens' Butterflyfish",
    "Chaet. bennettii": "Eclipse Butterflyfish",
    "Heniochus chrysostomus": "Pennant Bannerfish",
    "Heniochus monoceros": "Masked Bannerfish",
    "Trachinotus blochii": "Snubnose Dart",
    "Plectro. fasciolatus": "Pacific Gregory",
    "Pseudocaranx sp.": "Silver Trevally (unidentified)",
    "Caranx sexfasciatus": "Bigeye Trevally",
    "Abu. vaigiensis": "Indo-Pacific Sergeant",
    "Abu. sexfasciatus": "Scissortail Sergeant",
    "Abu. sordidus": "Blackspot Sergeant",
    "Abu. septemfasciatus": "Banded Sergeant",
    "Naso unicornis": "Bluespine Unicornfish",
    "Acanthurus dussumeieri": "Pencil Surgeonfish",
    "Zanclus cornutus": "Moorish Idol",
    "Prionurus maculatus": "Spotted Sawtail",
    "Parupeneus cyclostomus": "Goldsaddle Goatfish",
    "Chromis norfolkensis": "Norfolk Chromis",
    "Diodon hystrix": "Spotted Porcupinefish",
    "Cirrpectes sp.": "Cirripectes Blenny (unidentified)",
    "Trachypoma macacanthus": "Pacific Rockcod",
    "Chromis margaritifer": "Whitetail Puller",
    "Plectro. dickii": "Dick's Damsel",
    "Plectro. johnstonianus": "Johnston Damsel",
    "Fistularia commersonnii": "Smooth Flutemouth",
    "Aplodactylus etheridgeii": "Notch-head Marblefish",
    "Synodus dermatogenys": "Banded Lizardfish",
    "Girella cyanea": "Blue Drummer",
    "Crenimugil crenilabis": "Wartylip Mullet",
    "Mulloidicchthys flavolineatus": "Yellowstripe Goatfish",
    "Myxus elongatus": "Sand Mullet",
    "Mugil cephalus": "Sea Mullet",
    "Chrysiptera notialis": "Southern Demoiselle",
    "Kyphosus vaigiensis": "Brassy Drummer",
    "Epinephelus rivulatus": "Chinaman Rockcod",
    "Bothus pantherinus": "Leopard Flounder",
    "Pagrus auratus": "Snapper",
    "Parupeneus ciliatus": "Diamondscale Goatfish",
    "Parupeneus spilurus": "Blacksaddle Goatfish",
    "Eviota hoesei": "Doug's Eviota",
    "Leiuranus semicinctus": "Saddled Snake Eel",
    "Gymnothorax eurostus": "Stout Moray",
    "Taeniamia leai": "Lea's Cardinalfish",
    "Sargocentron punctatissimum": "Speckled Squirrelfish",
    "Microcanthus joyceae": "Stripey",
    "Sphyraena acutipinnis": "Sharpfin Barracuda",
    "Acanthistius cinctus": "Yellowbanded Wirrah",
    "Gymnothorax annasona": "Lord Howe Moray",
    "Sargocentron rubrum": "Red Squirrelfish",
    "Labroides dimidiatus": "Common Cleanerfish",
    "Gymnothorax nubilis": "Grey Moray",
    "Cymolutes praetextatus": "Knife Wrasse",
    "Pervagor alternans": "Yelloweye Leatherjacket",
    "Epinephelus daemelli": "Black Rockcod",
    "Cheilodactylus ephippium": "Painted Morwong",
    "Pomacentrus pavo": "Peacock Damsel",
    "Myliobatis tenuicaudatus": "Southern Eagle Ray",
    "Goniistius francisi": "Blacktip Morwong",
    "Seriola lalandi": "Yellowtail Kingfish",
    "Bodianus axillaris": "Coral Pigfish",
    "Ostorhinchus doederleini": "Fourline Cardinalfish",
}

SPECIES_COLORS = (
    "#e85d75",
    "#2a9d8f",
    "#e9a23b",
    "#4f86c6",
    "#9b5de5",
    "#00a8e8",
    "#f15bb5",
    "#5c946e",
    "#d76a03",
    "#6d597a",
    "#0081a7",
    "#bc4749",
)


def _name_tokens(name: str) -> tuple[str, str]:
    tokens = [re.sub(r"[^A-Za-z0-9]", "", token).upper() for token in name.split()]
    genus = (tokens[0] + "XXX")[:3]
    species = tokens[1] if len(tokens) > 1 else "SP"
    return genus, species


def _catalog_codes(names: tuple[str, ...]) -> list[str]:
    tokens = [_name_tokens(name) for name in names]
    bases = [f"MS{genus}{(species + 'XXX')[:3]}" for genus, species in tokens]
    counts = Counter(bases)
    collision_groups: dict[str, list[int]] = defaultdict(list)
    for index, base in enumerate(bases):
        if counts[base] > 1:
            collision_groups[base].append(index)
    codes = list(bases)
    for base, indexes in collision_groups.items():
        width = 4
        while True:
            candidates = [f"MS{tokens[index][0]}{tokens[index][1][:width]}" for index in indexes]
            if len(candidates) == len(set(candidates)):
                for index, candidate in zip(indexes, candidates, strict=True):
                    codes[index] = candidate
                break
            width += 1
            if width > max(len(tokens[index][1]) for index in indexes) + 1:
                raise ValueError(f"Could not create unique species codes for {base}")
    if len(codes) != len(set(codes)):
        raise ValueError("Master species catalogue generated duplicate codes")
    return codes


def master_species_records() -> tuple[tuple[str, str, str, str], ...]:
    if set(MASTER_SPECIES_COMMON_NAMES) != set(MASTER_SPECIES_NAMES):
        missing = set(MASTER_SPECIES_NAMES) - set(MASTER_SPECIES_COMMON_NAMES)
        extra = set(MASTER_SPECIES_COMMON_NAMES) - set(MASTER_SPECIES_NAMES)
        raise ValueError(f"Master common-name catalogue mismatch: missing={missing}, extra={extra}")
    codes = _catalog_codes(MASTER_SPECIES_NAMES)
    return tuple(
        (
            MASTER_SPECIES_COMMON_NAMES[name],
            name,
            code,
            SPECIES_COLORS[index % len(SPECIES_COLORS)],
        )
        for index, (name, code) in enumerate(zip(MASTER_SPECIES_NAMES, codes, strict=True))
    )


MASTER_SPECIES = master_species_records()
