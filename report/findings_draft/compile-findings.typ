// Wrapper to compile 04-findings-ver3.typ with its bibliography
#set page(paper: "a4", margin: (x: 2.5cm, y: 2.5cm))
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, leading: 0.65em)

#include "04-findings-ver3.typ"

#bibliography("works-findings.bib", style: "ieee")
