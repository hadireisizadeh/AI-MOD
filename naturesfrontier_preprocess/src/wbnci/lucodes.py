from __future__ import annotations


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    "convert a hex string to rgb tuple"
    return tuple(int(h.lstrip('#')[i:i+2], base=16) for i in (0, 2, 4))


class_to_hex = {
    # 0: "#ffffff",
    9: "#cccc00",
    10: "#ffff64",
    11: "#ffff64",
    12: "#ffff00",
    15: "#ffd700",
    16: "#ffd700",
    20: "#aaf0f0",
    25: "#00fbff",
    26: "#00fbff",
    29: "#f78fe6",
    30: "#dcf064",
    34: "#dcf064",
    35: "#dcf064",
    39: "#dcf064",
    40: "#c8c864",
    44: "#c8c864",
    45: "#c8c864",
    49: "#c8c864",
    50: "#006400",
    55: "#006400",
    60: "#00a000",
    65: "#00a000",
    61: "#00a000",
    66: "#00a000",
    62: "#aac800",
    67: "#aac800",
    70: "#003c00",
    75: "#003c00",
    71: "#003c00",
    76: "#003c00",
    72: "#005000",
    77: "#005000",
    80: "#285000",
    85: "#285000",
    81: "#285000",
    86: "#285000",
    82: "#286400",
    87: "#286400",
    90: "#788200",
    95: "#788200",
    100: "#8ca000",
    104: "#8ca000",
    105: "#8ca000",
    109: "#8ca000",
    110: "#be9600",
    114: "#be9600",
    115: "#be9600",
    119: "#be9600",
    120: "#966400",
    121: "#966400",
    122: "#966400",
    124: "#966400",
    125: "#966400",
    126: "#966400",
    130: "#ffb432",
    134: "#ffb432",
    140: "#ffdcd2",
    144: "#ffdcd2",
    150: "#ffebaf",
    154: "#ffebaf",
    151: "#ffc864",
    155: "#ffc864",
    152: "#ffd278",
    156: "#ffd278",
    153: "#ffebaf",
    157: "#ffebaf",
    160: "#00785a",
    165: "#00785a",
    170: "#009678",
    175: "#009678",
    180: "#00dc82",
    184: "#00dc82",
    190: "#c31400",
    200: "#fff5d7",
    204: "#fff5d7",
    201: "#dcdcdc",
    205: "#dcdcdc",
    202: "#fff5d7",
    206: "#fff5d7",
    210: "#0046c8",
    220: "#ffffff"
}


class_to_name = {
    9: "Cropland with BMPs",
    10: "Cropland rainfed",
    11: "Cropland rainfed - Herbaceous cover",
    12: "Cropland rainfed - Tree or shrub cover",
    15: "Cropland - intensified rainfed",
    16: "Cropland - intensified rainfed with BMPs",
    20: "Cropland irrigated or post-flooding",
    25: "Cropland - intensified irrigated",
    26: "Cropland - intensified irrigated with BMPs",
    29: "Oil Palm",
    30: "Mosaic cropland (>50%) / natural vegetation (tree/shrub/herbaceous cover) (<50%)",
    40: "Mosaic natural vegetation (tree/shrub/herbaceous cover) (>50%) / cropland (<50%) ",
    50: "Tree cover broadleaved evergreen closed to open (>15%)",
    60: "Tree cover  broadleaved  deciduous  closed to open (>15%)",
    61: "Tree cover  broadleaved  deciduous  closed (>40%)",
    62: "Tree cover  broadleaved  deciduous  open (15-40%)",
    70: "Tree cover  needleleaved  evergreen  closed to open (>15%)",
    71: "Tree cover  needleleaved  evergreen  closed (>40%)",
    72: "Tree cover  needleleaved  evergreen  open (15-40%)",
    80: "Tree cover  needleleaved  deciduous  closed to open (>15%)",
    81: "Tree cover  needleleaved  deciduous  closed (>40%)",
    82: "Tree cover  needleleaved  deciduous  open (15-40%)",
    90: "Tree cover  mixed leaf type (broadleaved and needleleaved)",
    100: "Mosaic tree and shrub (>50%) / herbaceous cover (<50%)",
    110: "Mosaic herbaceous cover (>50%) / tree and shrub (<50%)",
    120: "Shrubland",
    121: "Shrubland evergreen",
    122: "Shrubland deciduous",
    130: "Grassland",
    140: "Lichens and mosses",
    150: "Sparse vegetation (tree/shrub/herbaceous cover) (<15%)",
    151: "Sparse tree (<15%)",
    152: "Sparse shrub (<15%)",
    153: "Sparse herbaceous cover (<15%)",
    160: "Tree cover flooded fresh or brakish water",
    170: "Tree cover flooded saline water",
    180: "Shrub or herbaceous cover flooded fresh/saline/brakish water",
    190: "Urban areas",
    200: "Bare areas",
    201: "Consolidated bare areas",
    202: "Unconsolidated bare areas",
    210: "Water bodies",
    220: "Permanent snow and ice"
}


class_to_rgb = {}
for c, h in class_to_hex.items():
    rgb = hex_to_rgb(h)
    class_to_rgb[c] = rgb
    if c not in class_to_name:
        class_to_name[c] = ''


activity_code_to_name = {
    0: "nodata",
    1: "Natural",
    2: "Cropland",
    3: "Grazing",
    4: "Forestry",
    5: "Multi-use",
    6: "Developed",
    7: "Water",
    8: "Ice"
}

activity_code_to_hex = {
    0: "#ffffff",
    1: "#0bb215",
    2: "#ffebaf",
    3: "#e9bf63",
    4: "#738f50",
    5: "#fdbac4",
    6: "#c31400",
    7: "#0046c8",
    8: "#dddddd"
}

activity_to_rgb = {}
for c, h in activity_code_to_hex.items():
    rgb = hex_to_rgb(h)
    activity_to_rgb[c] = rgb

