// template.typ
#let project(
  title: "",
  author: "",
  student_id: "",
  degree: "",
  supervisor: "",
  date: datetime.today(),
  abstract: [],
  acknowledgments: [],
  body
) = {
  // 1. Page & Font Setup
  set document(author: author, title: title)
  set page(
    paper: "a4",
    margin: (left: 2.5cm, right: 2.5cm, top: 2.5cm, bottom: 2.5cm),
    numbering: none,
  )
  set text(font: "Times New Roman", lang: "en", size: 12pt)
  
  // 2. The Title Page [cite: 1]
  align(center)[
    // INSERT LOGO HERE
    #image("uel.png", width: 5cm) 
    #v(2em) // Adds a small gap between logo and text
    
    #text(size: 14pt)[SCHOOL OF ARCHITECTURE, COMPUTING AND ENGINEERING] \
    #v(0.5em)
    #text(size: 12pt)[Department of Engineering and Computing]
    
    #v(4fr)
    
    #text(weight: "bold", size: 22pt)[#title]
    
    #v(2fr)
    
    #text(weight: "bold", size: 14pt)[#author] \
    #text(size: 12pt)[#student_id]
    
    #v(2fr)
    
    A report submitted in part fulfilment of the degree of \
    BSc (Hons) in #text(style: "italic")[#degree]
    
    #v(1em)
    
    Supervisor: #supervisor \
    CN6000

    #v(2fr)
    
    
    #v(1em)
    #date.display("[day] [month repr:long] [year]")
    #v(2em)
  ]
  pagebreak()

  set page(numbering: "i")
  counter(page).update(1) 

  // 3. Front Matter (Abstract & Acknowledgments)
  // We use aligned numbering: none so they don't get "1. Abstract"
  set heading(numbering: none)
  
  heading(level: 1, outlined: true)[Abstract]
  abstract
  pagebreak()

  heading(level: 1, outlined: true)[Acknowledgments]
  acknowledgments
  pagebreak()

  // 4. Table of Contents [cite: 6]
  outline(depth: 3, indent: 2em)
  pagebreak()

  // 4.1 List of Figures
  outline(
    title: [List of Figures],
    target: figure.where(kind: image),
  )
  pagebreak()

  // 4.2 List of Tables
  outline(
    title: [List of Tables],
    target: figure.where(kind: table),
  )
  pagebreak()

  // 4.1 Start numbering pages
  set page(numbering: "1")
  counter(page).update(1) 
  // ------------------------

  // 5. Main Content Styling & Headers
  set page(
    numbering: "1",
    // HEADER LOGIC STARTS HERE
    header: context {
      // 1. Get the current page number
      let current_page = here().page()
      
      // 2. Find the last heading BEFORE this point
      let before = query(selector(heading.where(level: 1)).before(here()))
      
      // 3. Find the first heading AFTER this point (potential new chapter on this page)
      let after = query(selector(heading.where(level: 1)).after(here()))
      
      let target_heading = none

      // LOGIC: If a heading exists AFTER the header but ON THE SAME PAGE, use it.
      if after.len() > 0 and after.first().location().page() == current_page {
        target_heading = after.first()
      } 
      // OTHERWISE: Use the heading from before (continuation of previous chapter)
      else if before.len() > 0 {
        target_heading = before.last()
      }

      // 4. Render the header if we found a valid heading
      if target_heading != none {
        // Detect whether this heading is inside the appendix section
        // by checking if the numbering format uses letter-style ("A.1")
        let is_appendix = target_heading.numbering != none and str(target_heading.numbering).contains("A")
        grid(
            columns: (1fr, 1fr),
            align(left)[
              #text(style: "italic", size: 10pt)[
                #if is_appendix [
                  Appendix #numbering("A", counter(heading).at(target_heading.location()).first())
                ] else if target_heading.numbering != none [
                  Chapter #counter(heading).at(target_heading.location()).first(): #target_heading.body
                ] else [
                  #target_heading.body
                ]
              ]
            ],
            align(right)[
              #text(size: 10pt)[#author]
            ]
        )
        v(-0.8em)
        line(length: 100%, stroke: 0.5pt + gray)
      }
    }
    // ... end set page ...
  )
  
  counter(page).update(1)
  set heading(numbering: "1.1")
  
  // Custom rule to force "Chapter X: Title" formatting
  show heading.where(level: 1): it => {
    if it.numbering == none {
      pagebreak(weak: true)
      v(1em)
      text(size: 16pt, weight: "bold")[
        #it.body
      ]
      v(1em)
    } else {
      pagebreak(weak: true)
      v(1em)
      text(size: 16pt, weight: "bold")[
        Chapter #counter(heading).display(): #it.body
      ]
      v(1em)
    }
  }

  // Paragraph styling
  set par(justify: true, leading: 0.8em)

  // Cross-reference formatting: render @ch-X as "Chapter N" and @sec-X as "Section N.N"
  show ref: it => {
    let el = it.element
    if el != none and el.func() == heading {
      let target = str(it.target)
      let supplement = if target.starts-with("ch-") { "Chapter" } else { "Section" }
      let numbering_style = if el.numbering != none { el.numbering } else { "1." }
      return [#supplement #numbering(numbering_style, ..counter(heading).at(el.location()))]
    }
    it
  }
  
  body
}

// Helper for Appendices to switch lettering
#let appendix(body) = {
  pagebreak()
  counter(heading).update(0)
  set heading(numbering: "A.1")
  
  // Style level 1 headings specifically for appendices
  show heading.where(level: 1): it => {
    pagebreak(weak: true)
    v(2em)
    text(size: 16pt, weight: "bold")[
      Appendix #numbering("A", counter(heading).get().first())
    ]
    v(0.5em)
    text(size: 12pt, style: "italic")[
      #it.body
    ]
    v(1.5em)
  }
  
  // Ensure level 2 headings have proper spacing and keep the A.1 style
  show heading.where(level: 2): it => {
    v(1em)
    it
    v(0.5em)
  }
  body
}