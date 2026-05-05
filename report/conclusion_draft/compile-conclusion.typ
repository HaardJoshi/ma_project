// Wrapper to compile 06-conclusion-ver1.typ with its bibliography
#set page(paper: "a4", margin: (x: 2.5cm, y: 2.5cm))
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")

#include "06-conclusion-ver1.typ"

#bibliography("works-conclusion.bib", style: "ieee")
