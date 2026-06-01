def print_comparison_results(
    ref_rgb, ref_hex, ref_exact, ref_meta,
    tgt_rgb, tgt_hex, tgt_exact, tgt_meta,
    delta_e, rgb_dist, same_meta, match_status
):
    """
    Prints standard color comparison logs in a beautifully aligned,
    hierarchical semantic format using standard ASCII.
    """
    print("\n========== COLOR COMPARISON ==========\n")

    print("REFERENCE:")
    print(f"{'RGB':<16}: {ref_rgb}")
    print(f"{'HEX':<16}: {ref_hex}")
    print(f"{'EXACT NAME':<16}: {ref_exact}")
    print(f"{'META FAMILY':<16}: {ref_meta}\n")

    print("TARGET:")
    print(f"{'RGB':<16}: {tgt_rgb}")
    print(f"{'HEX':<16}: {tgt_hex}")
    print(f"{'EXACT NAME':<16}: {tgt_exact}")
    print(f"{'META FAMILY':<16}: {tgt_meta}\n")

    print("COMPARISON:")
    print(f"{'DeltaE':<16}: {delta_e:.2f}")
    print(f"{'RGB Distance':<16}: {rgb_dist:.2f}")
    print(f"{'Same MetaFamily':<16}: {same_meta}")
    print(f"{'Match Status':<16}: {match_status}")
