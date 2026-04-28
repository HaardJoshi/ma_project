// Wrapper to compile revised-methodology-ver4.typ with its bibliography
#set page(paper: "a4", margin: (x: 2.5cm, y: 2.5cm))
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, leading: 0.65em)

#include "revised-methodology-ver4.typ"

#bibliography("works-methodology.bib", style: "ieee")
