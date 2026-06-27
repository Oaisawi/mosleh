# Mosleh AI - Couples Counseling Assistant
import logging

# Configure logging so pipeline and agent logs appear in the terminal (Streamlit runs from there)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
