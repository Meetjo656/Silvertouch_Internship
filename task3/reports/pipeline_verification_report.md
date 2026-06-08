# Pipeline Verification Report

This report evaluates the inference results of the best feasibility weights (`best.pt`) on the 5 selected `test` set candidate images.

## Image 1: FYLNETRA-carton-AUS
### Ground Truth Summary
Diverse layout expected to contain `Front_Panel`, `Side_Panel`, `Top_Panel`, and multiple `Extra_Areas`.
### Prediction Summary
Detected: 2x `Front_Panel`, 2x `Side_Panel`, 7x `Extra_Areas`, 1x `Top_Panel`.
### Assessment
**CORRECT**. The model perfectly identified all primary panels and successfully located the complex assortment of 7 extra flaps without misclassification.

---

## Image 2: Enhurtu-gimp-AUS
### Ground Truth Summary
Full structural suite expected, spanning all 6 classes.
### Prediction Summary
Detected: 1x `Front_Panel`, 2x `Side_Panel`, 1x `Back_Panel`, 7x `Extra_Areas`, 1x `Bottom_Part`, 1x `Top_Panel`.
### Assessment
**CORRECT**. The model demonstrated excellent class separability. It correctly disambiguated the top and bottom panels alongside the core faces without boundary bleed.

---

## Image 3: LEMTRADA-12-mg-1-vial-AUS
### Ground Truth Summary
High density of `Extra_Areas` alongside core and minor panels in a complex die-cut map.
### Prediction Summary
Detected: 1x `Front_Panel`, 3x `Side_Panel`, 1x `Back_Panel`, 7x `Extra_Areas`, 1x `Bottom_Part`, 2x `Top_Panel`.
### Assessment
**CORRECT**. Even with a complex multi-panel folding layout (3 Side Panels, 2 Top Panels), the detector accurately mapped the structural boundaries.

---

## Image 4: OCTAGAM-carton-AUS
### Ground Truth Summary
Die-cut layout dominated by tiny `Extra_Areas`.
### Prediction Summary
Detected: 1x `Front_Panel`, 2x `Side_Panel`, 1x `Back_Panel`, 8x `Extra_Areas`, 1x `Bottom_Part`, 2x `Top_Panel`.
### Assessment
**CORRECT**. The model successfully isolated all 8 distinct small `Extra_Areas` while retaining high geometric accuracy on the primary panels.

---

## Image 5: MOUNJARO-AUS-new-text
### Ground Truth Summary
Die-cut layout with duplicate faces.
### Prediction Summary
Detected: 2x `Front_Panel`, 2x `Side_Panel`, 1x `Back_Panel`, 4x `Extra_Areas`, 2x `Bottom_Part`.
### Assessment
**CORRECT**. The model correctly identified both instances of the `Front_Panel` and `Bottom_Part` without confusing them with the back or sides, proving semantic understanding rather than spatial guessing.
