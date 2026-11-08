# Model Used : Gemini Flash 3.6

### Prompt Q1
Inspect the attached query image. The material is carpet.
Allowed defect labels and definitions:
color: A localized region has an abnormal colour or tone.
cut: A sharp incision or sliced region interrupts the texture.
hole: Material is missing in a compact hole-like region.
metal_contamination: A metallic foreign object lies on the surface.
thread: A foreign thread lies across the regular texture.
Return JSON with is_defective, defect_name, confidence, and evidence.
Use a Boolean for is_defective. defect_name must be exactly one listed label
or "good" if no defect is visible. These two fields must agree. Give confidence
from 0 to 100 in your selected decision and name, and one sentence of visible
evidence.

The additional images are labelled validation examples in this order:
Support image 1: color
Support image 2: cut
Support image 3: hole
Support image 4: metal_contamination
Support image 5: thread
Compare the query with all five examples and classify only the query image.
Return the same JSON schema.

### Prompt Q2
Inspect the attached query image. The material is grid.
Allowed defect labels and definitions:
bent: The regular grid lines are locally warped or displaced.
broken: Part of the repeating grid is interrupted or missing.
glue: A localized glue deposit changes the surface appearance.
metal_contamination: A metallic foreign object lies on the surface.
thread: A foreign thread lies across the regular texture.
Return JSON with is_defective, defect_name, confidence, and evidence.
Use a Boolean for is_defective. defect_name must be exactly one listed label
or "good" if no defect is visible. These two fields must agree. Give confidence
from 0 to 100 in your selected decision and name, and one sentence of visible
evidence.

The additional images are labelled validation examples in this order:
Support image 1: bent
Support image 2: broken
Support image 3: glue
Support image 4: metal_contamination
Support image 5: thread
Compare the query with all five examples and classify only the query image.
Return the same JSON schema.

### Prompt Q3
Inspect the attached query image. The material is leather.
Allowed defect labels and definitions:
color: A localized region has an abnormal colour or tone.
cut: A sharp incision or sliced region interrupts the texture.
fold: The surface is creased or folded out of its normal plane.
glue: A localized glue deposit changes the surface appearance.
poke: A small puncture-like indentation or opening is visible.
Return JSON with is_defective, defect_name, confidence, and evidence.
Use a Boolean for is_defective. defect_name must be exactly one listed label
or "good" if no defect is visible. These two fields must agree. Give confidence
from 0 to 100 in your selected decision and name, and one sentence of visible
evidence.

The additional images are labelled validation examples in this order:
Support image 1: color
Support image 2: cut
Support image 3: fold
Support image 4: glue
Support image 5: poke
Compare the query with all five examples and classify only the query image.
Return the same JSON schema.

### Prompt Q4
Inspect the attached query image. The material is tile.
Allowed defect labels and definitions:
crack: A narrow fracture-like line crosses the material surface.
glue_strip: An elongated strip of adhesive lies on the tile.
gray_stroke: A grey linear mark appears across the tile.
oil: An oily region changes colour or reflectance.
rough: A localized patch is visibly rougher than normal.
Return JSON with is_defective, defect_name, confidence, and evidence.
Use a Boolean for is_defective. defect_name must be exactly one listed label
or "good" if no defect is visible. These two fields must agree. Give confidence
from 0 to 100 in your selected decision and name, and one sentence of visible
evidence.

The additional images are labelled validation examples in this order:
Support image 1: crack
Support image 2: glue_strip
Support image 3: gray_stroke
Support image 4: oil
Support image 5: rough
Compare the query with all five examples and classify only the query image.
Return the same JSON schema.

### Prompt Q5
Inspect the attached query image. The material is wood.
Allowed defect labels and definitions:
color: A localized region has an abnormal colour or tone.
combined: More than one defect mechanism appears in the same image.
hole: Material is missing in a compact hole-like region.
liquid: A liquid stain changes colour or reflectance locally.
scratch: A thin elongated abrasion crosses the surface.
Return JSON with is_defective, defect_name, confidence, and evidence.
Use a Boolean for is_defective. defect_name must be exactly one listed label
or "good" if no defect is visible. These two fields must agree. Give confidence
from 0 to 100 in your selected decision and name, and one sentence of visible
evidence.

The additional images are labelled validation examples in this order:
Support image 1: color
Support image 2: combined
Support image 3: hole
Support image 4: liquid
Support image 5: scratch
Compare the query with all five examples and classify only the query image.
Return the same JSON schema.

### JSONs

Q1
{
  "is_defective": true,
  "defect_name": "color",
  "confidence": 95,
  "evidence": "There is a localized, dark discolored stain/patch near the lower-left section of the woven carpet texture."
}

Q2
{
  "is_defective": true,
  "defect_name": "bent",
  "confidence": 98,
  "evidence": "Multiple regular wire strands of the grid structure are locally warped and bent out of shape across the surface."
}

Q3
{
  "is_defective": true,
  "defect_name": "cut",
  "confidence": 95,
  "evidence": "A sharp incision and slice interrupt the leather texture in the lower-left central portion of the surface."
}

Q4
{
  "is_defective": true,
  "defect_name": "crack",
  "confidence": 98,
  "evidence": "A prominent dark, Y-shaped fracture line splits and crosses the tile surface."
}

Q5
{
  "is_defective": false,
  "defect_name": "good",
  "confidence": 95,
  "evidence": "The wood surface exhibits normal wood grain texture and vertical natural fissures without any color spots, holes, liquid stains, or abrasion scratches."
}

