import streamlit as st
import json

from utils import setup_page, load_macro_stats

setup_page(title="Pipeline DAG")

stats = load_macro_stats()
dag = stats.get("pipeline_dag", {})

st.markdown("<h1><span style='color:#00FFAA;'>Phase 4:</span> Network Execution DAG</h1>", unsafe_allow_html=True)
st.caption("Directed Acyclic Graph (DAG) for the end-to-end multi-agent corporate finance processing pipeline.")
st.markdown("---")

c1, c2 = st.columns([1.5, 1])

with c1:
    # Build Graphviz DOT string structurally from macro_stats.json pipeline_dag
    dot_str = [
        'digraph G {',
        '  bgcolor="#0E1117";',
        '  node [shape=box, style="filled,rounded", fontname="Courier", fillcolor="#1E222B", fontcolor="#FAFAFA", color="#3399FF", margin=0.2];',
        '  edge [color="#888888", arrowsize=0.8];',
        '  rankdir="TB";',
        '  newrank=true;',
    ]

    stages = dag.get("stages", [])
    
    # Define subgraphs (clusters) for each stage
    for i, stage in enumerate(stages):
        stage_name = stage.get("name", f"Stage_{i}")
        dot_str.append(f'  subgraph cluster_{i} {{')
        dot_str.append(f'    label="{stage_name}";')
        dot_str.append('    fontcolor="#00FFAA";')
        dot_str.append('    fontname="Courier-Bold";')
        dot_str.append('    color="#333333";')
        dot_str.append('    style="dashed,rounded";')
        
        scripts = stage.get("scripts", [])
        for script in scripts:
            s_name = script.get("name", "script.py")
            # Create a valid DOT node ID (remove .py)
            n_id = s_name.replace(".py", "").replace("-", "_")
            dot_str.append(f'    {n_id} [label="{s_name}"];')
            
        dot_str.append('  }')

    # Add edges between stages implicitly (linking the last node of stage N to the first of N+1)
    for i in range(len(stages) - 1):
        curr_scripts = stages[i].get("scripts", [])
        next_scripts = stages[i+1].get("scripts", [])
        
        if curr_scripts and next_scripts:
            # Simple linkage: connect all outputs of previous stage to first script of next stage
            # or just one proxy edge
            s1 = curr_scripts[-1].get("name").replace(".py", "").replace("-", "_")
            s2 = next_scripts[0].get("name").replace(".py", "").replace("-", "_")
            dot_str.append(f'  {s1} -> {s2} [style="bold", color="#00FFAA"];')
            
    dot_str.append('}')

    st.graphviz_chart("\n".join(dot_str), use_container_width=True)

with c2:
    st.markdown("### Execution Diagnostics")
    st.markdown("Select a module to view architectural processing rules.")
    
    # Iterate dynamically through the scripts and populate expanders
    for stage in stages:
        with st.expander(f"📁 {stage.get('name', 'Pipeline Stage')}", expanded=(stage.get("name") == "Data Cleaning")):
            for script in stage.get("scripts", []):
                s_name = script.get("name")
                s_desc = script.get("desc")
                
                # Highlight critical data cleaning scripts as requested by user constraints
                if s_name in ["retry_failed_tickers.py", "fix_dates.py"]:
                    st.markdown(f"<span style='color:#FFB300;'><b>{s_name}</b></span>: {s_desc}", unsafe_allow_html=True)
                    if s_name == "retry_failed_tickers.py":
                        st.caption("➥ Invokes fuzzy-string matching (Levenshtein distance) via `thefuzz` library to salvage missing Bloomberg CUSIP/Ticker alignments before final drop.")
                    elif s_name == "fix_dates.py":
                        st.caption("➥ Normalizes UNIX timestamps and generic `YYYY-MM-DD` string collisions acquired asynchronously across parallel execution threads.")
                else:
                    st.markdown(f"**{s_name}**: {s_desc}")
