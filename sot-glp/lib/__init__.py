from sot_glp.lib.boolean_flags import boolean_flags
from sot_glp.lib.count_parameters import count_parameters
from sot_glp.lib.float_range import float_range
from sot_glp.lib.get_clip_hyperparams import get_clip_hyperparams
from sot_glp.lib.get_params_group import get_params_group
from sot_glp.lib.get_set_random_state import get_random_state, set_random_state, get_set_random_state, random_seed
from sot_glp.lib.ood_metrics import get_fpr, get_auroc
from sot_glp.lib.json_utils import save_json, load_json
from sot_glp.lib.load_checkpoint import load_checkpoint
from sot_glp.lib.log_ood_metrics import log_ood_metrics
from sot_glp.lib.logger import LOGGER, setup_logger
from sot_glp.lib.meters import AverageMeter, DictAverage, ProgressMeter
from sot_glp.lib.save_checkpoint import save_checkpoint
from sot_glp.lib.track import track


__all__ = [
    "boolean_flags",
    "count_parameters",
    "float_range",
    "get_clip_hyperparams",
    "get_params_group",
    "get_random_state",
    "set_random_state",
    "get_set_random_state",
    "random_seed",
    "get_fpr",
    "get_auroc",
    "save_json",
    "load_json",
    "load_checkpoint",
    "log_ood_metrics",
    "LOGGER",
    "setup_logger",
    "AverageMeter",
    "DictAverage",
    "ProgressMeter",
    "save_checkpoint",
    "track",
]


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
