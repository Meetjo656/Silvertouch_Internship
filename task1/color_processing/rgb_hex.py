def rgb_to_hex(rgb):
    """
    Convert an RGB color to standard HEX string.
    """
    return "#{:02X}{:02X}{:02X}".format(
        int(rgb[0]),
        int(rgb[1]),
        int(rgb[2])
    )
