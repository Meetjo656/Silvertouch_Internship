# Verification Image Selection Report

To visually validate the structural detection capabilities of the YOLOv8n feasibility model without relying on quantitative metrics alone, 5 representative artwork images were pseudo-randomly selected from the unseen `test` split.

## Selection Criteria
The selection algorithm strictly ensured the presence of:
1. **Diverse Layouts**: Images with at least 3 distinct structural classes.
2. **Complex Extra Areas**: At least one image featuring `Extra_Areas` (flaps, tabs, non-core panels).
3. **Top / Bottom Panels**: At least one image containing visible `Top_Panel` or `Bottom_Part` polygons, which are notoriously thin and difficult to separate.

## Selected Candidates

1. `FYLNETRA-carton-AUS_jpg.rf.073fae71eea6b081fe7dd061d2ea2341.jpg`
   - *Justification*: Contains multiple complex `Extra_Areas` alongside core panels and a `Top_Panel`.
2. `Enhurtu-gimp-AUS_jpg.rf.a64afaf838d48fa140938d9ab5bfc35e.jpg`
   - *Justification*: Represents a "full suite validation", containing all 6 available structural classes (`Front_Panel`, `Side_Panel`, `Back_Panel`, `Bottom_Part`, `Top_Panel`, and `Extra_Areas`).
3. `LEMTRADA-12-mg-1-vial-AUS_jpg.rf.3825d646a46610f5186e12ea90bfd4f4.jpg`
   - *Justification*: Demonstrates high complexity with a `Top_Panel` and `Bottom_Part` co-occurring alongside duplicate side panels.
4. `OCTAGAM-carton-AUS_jpg.rf.15d16adf2d7c159fb9b097774b6a79b6.jpg`
   - *Justification*: Contains an extremely high density of `Extra_Areas` (8 expected) explicitly testing the detector's capability on small, clustered objects.
5. `MOUNJARO-AUS-new-text_jpg.rf.2e0659fb9975f1d9f90a4e2b7f3914fa.jpg`
   - *Justification*: Tests duplicate large panels with multiple instances of `Front_Panel` and `Bottom_Part`.
