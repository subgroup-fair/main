# fairbench/methods/__init__.py
from .dr import run_dr
from .dr_subgroup_subset import run_dr_subgroup_subset
from .dr_subgroup_subset_3q import run_dr_subgroup_subset_3q
from .gerryfair_wrapper import run_gerryfair
from .multicalib_wrapper import run_multicalib
from .seq_wrapper import run_sequential
from .reduction_wrapper import run_reduction

def run_method(args, data):
    if args.method == "dr":
        return run_dr(args, data)
    if args.method == "dr_subgroup_subset":
        return run_dr_subgroup_subset(args, data)
    if args.method == "dr_subgroup_subset_3q":
        return run_dr_subgroup_subset_3q(args, data)
    if args.method == "gerryfair":
        return run_gerryfair(args, data)
    if args.method == "multicalib":
        return run_multicalib(args, data)
    if args.method == "sequential":
        return run_sequential(args, data)
    if args.method == "reduction":
        return run_reduction(args, data)
    raise ValueError(args.method)
