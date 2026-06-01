"""
Take classified masks.

Separate:
    Text Masks
    Structure Masks
    Unknown Masks

Remove text masks.

Return:
    Structural masks only
"""


class SAMTextSuppressor:
    """
    Separates and suppresses text masks produced by SAMTextClassifier.

    Public API
    ----------
    separate_masks(classified_masks)                       -> tuple[list, list, list]
    get_text_masks(classified_masks)                       -> list
    get_structure_masks(classified_masks)                  -> list
    get_unknown_masks(classified_masks)                    -> list
    get_kept_masks(classified_masks)                       -> list
    generate_summary(text_masks, structure_masks, unknown_masks) -> dict
    suppress(classified_masks)                             -> dict | None
    """

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @staticmethod
    def _valid_masks(classified_masks: list) -> list:
        """Return only non-None entries from *classified_masks*."""
        return [m for m in (classified_masks or []) if m is not None]

    # ------------------------------------------------------------------
    # Public method 1 – single separation point (used by all helpers)
    # ------------------------------------------------------------------

    def separate_masks(
        self, classified_masks: list
    ) -> tuple[list, list, list]:
        """
        Split classified masks into three labelled buckets.

        Parameters
        ----------
        classified_masks : list[dict | None]
            Output of SAMTextClassifier.classify_masks().

        Returns
        -------
        (text_masks, structure_masks, unknown_masks)
            Three lists; any of them may be empty but never None.
        """
        text_masks      = []
        structure_masks = []
        unknown_masks   = []

        for mask in self._valid_masks(classified_masks):
            label = mask.get("label", "unknown")
            if label == "text":
                text_masks.append(mask)
            elif label == "structure":
                structure_masks.append(mask)
            else:
                unknown_masks.append(mask)

        return text_masks, structure_masks, unknown_masks

    # ------------------------------------------------------------------
    # Public methods 2-4 – single-label accessors
    # ------------------------------------------------------------------

    def get_text_masks(self, classified_masks: list) -> list:
        """Return only masks labelled 'text'."""
        text_masks, _, _ = self.separate_masks(classified_masks)
        return text_masks

    def get_structure_masks(self, classified_masks: list) -> list:
        """Return only masks labelled 'structure'."""
        _, structure_masks, _ = self.separate_masks(classified_masks)
        return structure_masks

    def get_unknown_masks(self, classified_masks: list) -> list:
        """Return only masks labelled 'unknown'."""
        _, _, unknown_masks = self.separate_masks(classified_masks)
        return unknown_masks

    # ------------------------------------------------------------------
    # Public method 5 – kept = structure + unknown
    # ------------------------------------------------------------------

    def get_kept_masks(self, classified_masks: list) -> list:
        """
        Return all masks that are *not* text (i.e. structure + unknown).

        Parameters
        ----------
        classified_masks : list[dict | None]
            Full classified mask list.

        Returns
        -------
        list
            Masks to keep after text suppression.
        """
        _, structure_masks, unknown_masks = self.separate_masks(classified_masks)
        return structure_masks + unknown_masks

    # ------------------------------------------------------------------
    # Public method 6 – summary as dict (JSON-serialisable)
    # ------------------------------------------------------------------

    def generate_summary(
        self,
        text_masks: list,
        structure_masks: list,
        unknown_masks: list,
    ) -> dict:
        """
        Build a count-based summary dict.

        Parameters
        ----------
        text_masks, structure_masks, unknown_masks : list
            The three buckets returned by separate_masks().

        Returns
        -------
        dict with integer counts; always returns a dict (never None).
        """
        total = len(text_masks) + len(structure_masks) + len(unknown_masks)
        kept  = len(structure_masks) + len(unknown_masks)

        return {
            "text_count":      len(text_masks),
            "structure_count": len(structure_masks),
            "unknown_count":   len(unknown_masks),
            "removed_count":   len(text_masks),
            "kept_count":      kept,
            "total_count":     total,
        }

    # ------------------------------------------------------------------
    # Public method 7 – main entry point
    # ------------------------------------------------------------------

    def suppress(self, classified_masks: list) -> dict | None:
        """
        Suppress text masks and return kept masks with a summary.

        Parameters
        ----------
        classified_masks : list[dict | None]
            Output of SAMTextClassifier.classify_masks().

        Returns
        -------
        dict | None
            {
                "text_masks":      [...],
                "structure_masks": [...],
                "unknown_masks":   [...],
                "kept_masks":      [...],   # structure + unknown
                "removed_masks":   [...],   # text only
                "summary":         {...},   # generate_summary() dict
            }
            Returns None if classified_masks is empty or None.
        """
        if not classified_masks:
            return None

        text_masks, structure_masks, unknown_masks = self.separate_masks(
            classified_masks
        )

        return {
            "text_masks":      text_masks,
            "structure_masks": structure_masks,
            "unknown_masks":   unknown_masks,
            "kept_masks":      structure_masks + unknown_masks,
            "removed_masks":   text_masks,
            "summary":         self.generate_summary(
                                   text_masks, structure_masks, unknown_masks
                               ),
        }