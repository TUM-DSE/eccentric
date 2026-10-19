from . import accumulated_error, chromobius, decoder, logical_error, program_stats, routing, translation
from .base import Skip, Task

EXPERIMENTS = {
    "logical_error": logical_error,
    "decoder": decoder,
    "program_stats": program_stats,
    "accumulated_error": accumulated_error,
    "chromobius": chromobius,
    "routing": routing,
    "translation": translation,
}
