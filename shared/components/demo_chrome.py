"""Demo-only page chrome (header + footer) for the hosted Hugging Face Space.

Rendered only when ``EMB_EXPLORER_DEMO=1`` (set by the Space Dockerfile), so the
normal local apps are unaffected. Kept additive and self-contained so it merges
cleanly across branches.
"""

import os

import streamlit as st

REPO_URL = "https://github.com/Imageomics/emb-explorer"
EMBEDDINGS_DATASET_URL = "https://huggingface.co/datasets/imageomics/TreeOfLife-200M-Embeddings"
BIOCLIP2_URL = "https://huggingface.co/imageomics/bioclip-2"
BIOCLIP2_SITE_URL = "https://imageomics.github.io/bioclip-2/"
PYBIOCLIP_URL = "https://github.com/Imageomics/pybioclip"
TOL200M_URL = "https://huggingface.co/datasets/imageomics/TreeOfLife-200M"
IMAGEOMICS_URL = "https://imageomics.org"
NSF_AWARD_URL = "https://www.nsf.gov/awardsearch/showAward?AWD_ID=2118240"


def is_demo_mode() -> bool:
    """True when running as the hosted demo (the Space sets EMB_EXPLORER_DEMO=1)."""
    return os.environ.get("EMB_EXPLORER_DEMO", "0") == "1"


def render_demo_header() -> None:
    """Demo title + one-line plain-text intro (no links; links live in the footer)."""
    st.title("🔍 Image Embedding Explorer (Demo)")
    st.markdown(
        "A hosted demo of the Image Embedding Explorer. Explore precalculated "
        "BioCLIP 2 embeddings from a curated subset of the TreeOfLife-200M image "
        "collection."
    )


_FOOTER_CSS = """
<style>
.demo-footer { margin-top: 2rem; padding-top: 1rem;
               border-top: 1px solid rgba(128, 128, 128, 0.3);
               opacity: 0.6; font-size: 0.82rem; line-height: 1.5; }
.demo-footer:hover { opacity: 0.9; }
.demo-footer a { color: #3f9b6e; text-decoration: none; }
.demo-footer a:hover { text-decoration: underline; }
</style>
"""


def render_demo_footer() -> None:
    """Muted, bordered attribution / funding footer (Imageomics standard text).

    Adopts the bioclip-image-search footer, with the emb-explorer source repo
    and the TreeOfLife-200M-Embeddings dataset added.
    """
    st.markdown(_FOOTER_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="demo-footer">'
        f'This demo is built with the <a href="{REPO_URL}">emb-explorer</a> source '
        f'repository and explores a curated subset of the '
        f'<a href="{EMBEDDINGS_DATASET_URL}">TreeOfLife-200M-Embeddings</a> dataset. '
        f'For more information on the <a href="{BIOCLIP2_URL}">BioCLIP&nbsp;2</a> model '
        f'creation, see our <a href="{BIOCLIP2_SITE_URL}">BioCLIP&nbsp;2 Project website</a>, '
        'and for easier programmatic integration of BioCLIP&nbsp;2, checkout '
        f'<a href="{PYBIOCLIP_URL}">pybioclip</a>. To learn more about the data, check out '
        f'our <a href="{TOL200M_URL}">TreeOfLife-200M Dataset</a>.'
        '<br><br>'
        f'This work was supported by the <a href="{IMAGEOMICS_URL}">Imageomics Institute</a>, '
        "which is funded by the US National Science Foundation's Harnessing the Data "
        f'Revolution (HDR) program under <a href="{NSF_AWARD_URL}">Award&nbsp;#2118240</a> '
        '(Imageomics: A New Frontier of Biological Information Powered by Knowledge-Guided '
        'Machine Learning). Any opinions, findings and conclusions or recommendations '
        'expressed in this material are those of the author(s) and do not necessarily '
        'reflect the views of the National Science Foundation.'
        '</div>',
        unsafe_allow_html=True,
    )
