# Hugging Face Space (Docker SDK) for the emb-explorer precalculated demo.
#
# Built by manually pushing this Dockerfile + README + requirements plus the
# `apps/` and `shared/` source trees to the Space repo (no GitHub clone, no CI).
# Runs the precalculated Streamlit app on port 7860, reading curated demo data
# from the netzhang/demo dataset mounted read-only at /data.
FROM python:3.12-slim

# HF Spaces run as uid 1000; set up a matching user to avoid permission issues.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app
WORKDIR $HOME/app

# Install deps first (better layer caching). This is a precalc-only subset of
# pyproject.toml — no torch / open-clip / hpc-inference, since the precalculated
# app never generates embeddings.
COPY --chown=user requirements-space.txt .
RUN pip install --no-cache-dir -r requirements-space.txt

# App source (apps/ and shared/ are pushed to the Space repo alongside this file).
COPY --chown=user . $HOME/app

# Demo data is mounted at /data (see README).
ENV EMB_EXPLORER_DEMO_DATA_ROOT=/data

EXPOSE 7860
CMD ["streamlit", "run", "apps/precalculated/app.py", \
     "--server.port", "7860", \
     "--server.address", "0.0.0.0", \
     "--server.headless", "true", \
     "--server.enableCORS", "false", \
     "--server.enableXsrfProtection", "false"]
