# Person Annotation Rules

- Dataset class: **person** only.
- Annotate every clearly identifiable real human.
- Standing, walking, sitting and partially occluded people remain `person`.
- Tiny aerial people should be annotated when they are genuinely identifiable.
- Do not label shadows, mannequins, statues, posters, signs, or human images as real people.
- Keep boxes tight around the visible person without excessive background.
- A valid background/negative image has zero boxes.
- Model-generated boxes must be reviewed before being accepted as ground truth.
