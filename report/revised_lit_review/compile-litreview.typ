// Wrapper to compile 02-lit-review-ver3.typ with its bibliography
#set page(paper: "a4", margin: (x: 2.5cm, y: 2.5cm))
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, leading: 0.65em)

#include "02-lit-review-ver3.typ"

#bibliography("works-litreview.bib", style: "ieee")
